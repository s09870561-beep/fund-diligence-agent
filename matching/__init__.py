"""Matching layer — investment mandate fit scoring.

This module compares a researched DiligenceBrief against an investment
mandate and produces a structured fit score with field-level reasoning.

  - InvestmentMandate — Pydantic model defining what the investor is
    looking for (sectors, stage, check size, geography, exclusions).

  - match_mandate(brief, mandate) -> dict
    Sends the brief and mandate to the LLM, asks it to score fit 0-100
    and explain the reasoning field by field.
"""

import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

from presentation import DiligenceBrief
from utils.retry import retry_with_backoff

load_dotenv(override=True)

# -- OpenCode Zen client -------------------------------------------------

_client = OpenAI(
    base_url="https://opencode.ai/zen/v1",
    api_key=os.getenv("OPENCODE_ZEN_API_KEY", "").strip(),
)

MODEL = "deepseek-v4-flash-free"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class InvestmentMandate(BaseModel):
    """What an investor is looking for in a potential investment."""

    sectors: list[str] = Field(description="Target sectors / industries for investment")
    stage: str = Field(description="Preferred funding stage (e.g. 'early-stage Series A', 'growth-stage', 'late-stage')")
    check_size_min: float = Field(default=0.0, description="Minimum check size in USD")
    check_size_max: float = Field(default=0.0, description="Maximum check size in USD")
    geography: list[str] = Field(description="Target geographic regions / countries")
    excluded_industries: list[str] = Field(default_factory=list, description="Industries the investor explicitly avoids")


# ---------------------------------------------------------------------------
# Match prompt
# ---------------------------------------------------------------------------

_MATCH_PROMPT = """You are an investment analyst evaluating how well a researched company or fund fits a specific investment mandate.

You will be given:
  1. A due-diligence brief about a company/fund.
  2. An investment mandate with specific criteria.

Your job is to score the fit on a scale of 0–100 and explain your reasoning.

Scoring guidelines:
  - 80–100: Strong fit — most mandate criteria clearly satisfied by the brief.
  - 50–79: Partial fit — some criteria match, others are unclear or missing.
  - 0–49: Poor fit — mandate criteria are not satisfied, or the brief contradicts them.

Rules (these are not suggestions):
  - If the brief does NOT mention a criterion (e.g. funding stage, check size), flag it as
    "unclear" — do NOT assume the fit is good. It goes into uncertain_fields.
  - If the brief explicitly contradicts a criterion (e.g. brief says "enterprise SaaS" but
    mandate says "fintech only"), score it as a mismatch and explain why.
  - Be specific. Quote the brief where possible.
  - The uncertain_fields list should contain every mandate dimension where the brief
    provides insufficient information to judge fit.

Respond ONLY with a single JSON object — no markdown fences, no explanation:

{
  "score": 0-100,
  "reasoning": [
    {
      "field": "sector",
      "verdict": "match | mismatch | unclear",
      "detail": "brief indicates fintech, mandate includes fintech — match"
    },
    {
      "field": "stage",
      "verdict": "match | mismatch | unclear",
      "detail": "brief doesn't specify funding stage — unclear"
    }
  ],
  "uncertain_fields": ["stage", "check_size"]
}"""


# ---------------------------------------------------------------------------
# Brief serialisation helper
# ---------------------------------------------------------------------------


def _brief_to_text(brief: DiligenceBrief) -> str:
    """Convert a DiligenceBrief to a readable text block for the LLM."""
    data = brief.model_dump()
    lines = [
        f"Entity: {data.get('entity_name', '?')}",
        f"Overview: {data.get('overview', '')}",
    ]
    ldr = data.get("leadership", [])
    if ldr:
        lines.append("Leadership:")
        for m in ldr:
            lines.append(f"  - {m.get('name', '?')} ({m.get('title', '?')})")
    ra = data.get("recent_activity", [])
    if ra:
        lines.append("Recent activity:")
        for a in ra[:8]:
            lines.append(f"  - {a}")
    pd = data.get("past_deals", [])
    if pd:
        lines.append("Past deals / investments:")
        for d in pd[:8]:
            lines.append(f"  - {d}")
    su = data.get("sources_used", [])
    lines.append(f"Sources used: {', '.join(su) if su else 'none'}")
    oq = data.get("open_questions", [])
    if oq:
        lines.append("Open questions / uncertainties:")
        for q in oq[:5]:
            lines.append(f"  - {q}")
    hv = data.get("human_verified_claims", [])
    if hv:
        lines.append("Human-verified claims:")
        for c in hv[:5]:
            lines.append(f"  - {c}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# match_mandate
# ---------------------------------------------------------------------------


def match_mandate(brief: DiligenceBrief, mandate: InvestmentMandate) -> dict:
    """Score how well a researched entity fits an investment mandate.

    Args:
        brief: The DiligenceBrief produced by the research pipeline.
        mandate: An InvestmentMandate defining the investor's criteria.

    Returns:
        A dict with keys:
          - score           (int)      Fit score 0–100.
          - reasoning       (list[dict]) Field-level explanations, each with
                            ``field``, ``verdict``, and ``detail``.
          - uncertain_fields (list[str]) Mandate dimensions the brief doesn't
                            sufficiently cover.
    """
    brief_text = _brief_to_text(brief)

    mandate_text = (
        f"Sectors: {', '.join(mandate.sectors)}\n"
        f"Stage: {mandate.stage}\n"
        f"Check size: ${mandate.check_size_min:,.0f} – ${mandate.check_size_max:,.0f}\n"
        f"Geography: {', '.join(mandate.geography)}\n"
        f"Excluded industries: {', '.join(mandate.excluded_industries) if mandate.excluded_industries else 'none'}"
    )

    user_content = (
        f"Due-diligence brief:\n{brief_text}\n\n"
        f"Investment mandate:\n{mandate_text}\n\n"
        "Score the fit and explain your reasoning."
    )

    response = retry_with_backoff(
        lambda: _client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": _MATCH_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
        ),
    )

    if isinstance(response, str):
        return {
            "score": 0,
            "reasoning": [{"field": "error", "verdict": "mismatch", "detail": f"LLM call failed: {response}"}],
            "uncertain_fields": [],
        }

    content = response.choices[0].message.content or ""
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1]
        content = content.rsplit("```", 1)[0].strip()

    try:
        result = json.loads(content)
        return {
            "score": max(0, min(100, result.get("score", 0))),
            "reasoning": result.get("reasoning", []),
            "uncertain_fields": result.get("uncertain_fields", []),
        }
    except (json.JSONDecodeError, ValueError) as e:
        return {
            "score": 0,
            "reasoning": [{"field": "error", "verdict": "mismatch", "detail": f"Parse error: {e}"}],
            "uncertain_fields": [],
        }
