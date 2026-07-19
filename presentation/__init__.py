"""Presentation layer — formatting the final structured brief.

This module takes the agent's raw findings and evidence and renders
them into a polished, structured due-diligence report suitable for
a human reader.  It owns:

  - DiligenceBrief — a Pydantic model defining the output schema.
  - synthesize_brief(goal, gathered_data) -> DiligenceBrief
    Sends gathered evidence to the LLM and returns a validated brief.
  - format_brief(brief) -> None
    Pretty-prints the brief as a clean, sectioned rich Panel.
"""

import json
import os
import time
from datetime import datetime, timezone

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.markdown import Markdown

from utils.retry import retry_with_backoff

load_dotenv(override=True)

console = Console()

# -- OpenCode Zen client -------------------------------------------------

_client = OpenAI(
    base_url="https://opencode.ai/zen/v1",
    api_key=os.getenv("OPENCODE_ZEN_API_KEY", "").strip(),
)

MODEL = "deepseek-v4-flash-free"


# ---------------------------------------------------------------------------
# Pydantic model
# ---------------------------------------------------------------------------


class LeadershipMember(BaseModel):
    name: str = Field(description="Full name of the leader")
    title: str = Field(description="Role or position at the firm")
    source_confidence: str = Field(
        description="How reliable the source is",
        pattern=r"^(high|medium|low)$",
    )


class DiligenceBrief(BaseModel):
    entity_name: str = Field(description="The fund or company being researched")
    overview: str = Field(description="High-level summary of the entity (2-4 sentences)")
    leadership: list[LeadershipMember] = Field(
        description="Key leadership figures identified"
    )
    recent_activity: list[str] = Field(
        description="Recent news, moves, or strategic shifts"
    )
    past_deals: list[str] = Field(
        description="Notable historical investments or transactions"
    )
    open_questions: list[str] = Field(
        description=(
            "Things that were unclear, contradictory, or unsupported by "
            "the evidence. Empty if everything is certain."
        ),
    )
    human_verified_claims: list[str] = Field(
        default=[],
        description=(
            "Claims that were flagged as uncertain by the model but have "
            "been reviewed and confirmed by a human expert."
        ),
    )
    sources_used: list[str] = Field(
        description="Which tools/sources contributed data (e.g. web_search, sec_edgar_lookup)"
    )
    generated_at: str = Field(description="ISO-8601 timestamp of generation")


# ---------------------------------------------------------------------------
# Synthesize prompt
# ---------------------------------------------------------------------------

_SYNTHESIS_PROMPT = """You are a due-diligence analyst. Your job is to produce a concise, evidence-grounded research brief.

You will be given:
  1. The original research goal.
  2. A list of retrieval results, each from a different source (web_search, sec_edgar_lookup, etc.).

Your task is to fill out the following JSON schema using ONLY the evidence provided.  Follow these rules — they are not suggestions:

  - OVERVIEW: Summarise what is known in 2-4 sentences.  Be precise, not promotional.
  - LEADERSHIP: Only include individuals explicitly named in the evidence.  If no leaders are mentioned, leave the list empty.
  - RECENT_ACTIVITY: Specific, dated events.  Do not fabricate.
  - PAST_DEALS: Specific deals, investments, or transactions mentioned in the evidence.
  - OPEN_QUESTIONS: This is the most important field.  If information is missing, contradictory, or comes from a low-quality source, put it here.  Do NOT guess or make up information to fill gaps.  An empty list means "everything above is fully certain and complete" — so if you are unsure about *anything*, note it.
  - SOURCES_USED: List the names of sources that actually contributed useful data (e.g. "web_search", "sec_edgar_lookup").  Skip sources that returned nothing relevant.
  - GENERATED_AT: Current UTC timestamp in ISO-8601 format.

Respond ONLY with a single JSON object matching this exact structure — no markdown fences, no explanation:

{
  "entity_name": "...",
  "overview": "...",
  "leadership": [
    {"name": "...", "title": "...", "source_confidence": "high|medium|low"}
  ],
  "recent_activity": ["..."],
  "past_deals": ["..."],
  "open_questions": ["..."],
  "human_verified_claims": [],
  "sources_used": ["..."],
  "generated_at": "..."
}"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Shared usage tracker  —  lets _attempt_synthesis communicate token counts
# back to synthesize_brief without changing any return signatures.
# ---------------------------------------------------------------------------

_last_usage: dict[str, int] = {"input": 0, "output": 0}


# ---------------------------------------------------------------------------
# synthesize_brief
# ---------------------------------------------------------------------------


def synthesize_brief(goal: str, gathered_data: list[dict], tracer=None) -> DiligenceBrief:
    """Synthesise gathered evidence into a structured DiligenceBrief.

    Args:
        goal: The original research goal.
        gathered_data: List of result dicts from ``execute_tool`` calls.
        tracer: Optional ``Tracer`` for logging the LLM call.

    Returns:
        A validated ``DiligenceBrief`` instance.

    Raises:
        RuntimeError: If the LLM fails to produce valid JSON after
        the retry attempt.
    """
    t0 = time.time()

    # Build the evidence block
    evidence_lines = []
    for i, rd in enumerate(gathered_data, 1):
        source = rd.get("source", "?")
        success = rd.get("success", False)
        data = rd.get("data", "") if success else rd.get("error", "No data")
        status = "✔" if success else "✘"
        label = rd.get("label", f"Result #{i}")
        evidence_lines.append(f"--- {label} [{status}] (source: {source}) ---")
        evidence_lines.append(data[:2000])
        evidence_lines.append("")

    user_content = (
        f"Research goal:\n{goal}\n\n"
        f"Evidence gathered:\n"
        + "\n".join(evidence_lines)
    )

    # ---- First attempt --------------------------------------------------
    brief = _attempt_synthesis(user_content, tracer)
    if brief is not None:
        synthesize_brief.last_usage = dict(_last_usage)
        if tracer:
            _log_synthesis_llm(tracer, t0)
        return brief

    # ---- Retry with stricter prompt -------------------------------------
    console.print("[yellow]First synthesis attempt failed. Retrying with stricter prompt ...[/]")
    strict_prompt = (
        _SYNTHESIS_PROMPT
        + "\n\nIMPORTANT: Your previous response could not be parsed as valid JSON "
        "matching the expected schema.  Ensure:\n"
        "  1. The response is a SINGLE valid JSON object.\n"
        "  2. NO markdown fences, NO trailing commas.\n"
        "  3. All string values are double-quoted.\n"
        "  4. leadership is an array, even if empty.\n"
        "  5. open_questions is an array (empty if certain).\n"
        "  6. generated_at is an ISO-8601 string.\n"
        "Respond with ONLY the JSON object."
    )
    brief = _attempt_synthesis(user_content, tracer, system_override=strict_prompt)
    if brief is not None:
        synthesize_brief.last_usage = dict(_last_usage)
        if tracer:
            _log_synthesis_llm(tracer, t0)
        return brief

    raise RuntimeError(
        "synthesize_brief: LLM failed to produce valid DiligenceBrief "
        "JSON after 2 attempts."
    )


def _attempt_synthesis(
    user_content: str,
    tracer=None,
    system_override: str | None = None,
) -> DiligenceBrief | None:
    """Try one synthesis pass.  Returns a DiligenceBrief or None."""
    system = system_override or _SYNTHESIS_PROMPT

    response = retry_with_backoff(
        lambda: _client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
        ),
        max_retries=2,
        tracer=tracer,
    )

    if isinstance(response, str):
        console.print(f"[red]Synthesis LLM call failed: {response}[/]")
        _last_usage["input"] = 0
        _last_usage["output"] = 0
        return None

    usage = getattr(response, "usage", None)
    if usage:
        _last_usage["input"] = usage.prompt_tokens or 0
        _last_usage["output"] = usage.completion_tokens or 0
    else:
        _last_usage["input"] = 0
        _last_usage["output"] = 0

    content = (response.choices[0].message.content or "").strip()

    # Strip markdown fences if present
    if content.startswith("```"):
        content = content.split("\n", 1)[-1]
        content = content.rsplit("```", 1)[0].strip()

    try:
        data = json.loads(content)
    except (json.JSONDecodeError, ValueError) as e:
        console.print(f"[red]Synthesis JSON parse failed: {e}[/]")
        console.print(f"[dim]Raw: {content[:500]}[/]")
        return None

    try:
        return DiligenceBrief(**data)
    except Exception as e:
        console.print(f"[red]Synthesis Pydantic validation failed: {e}[/]")
        return None


def _log_synthesis_llm(tracer, t0: float) -> None:
    """Log the LLM call via tracer (if available)."""
    if not tracer:
        return
    try:
        tracer.log_llm_call(
            model=MODEL,
            tokens_input=None,
            tokens_output=None,
            latency_sec=round(time.time() - t0, 3),
            purpose="synthesize_brief",
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# format_brief
# ---------------------------------------------------------------------------

_CONFIDENCE_STYLES = {
    "high": "[green]● high[/]",
    "medium": "[yellow]● medium[/]",
    "low": "[red]● low[/]",
}


def format_brief(brief: DiligenceBrief) -> None:
    """Pretty-print a DiligenceBrief as a clean, sectioned report."""
    # ── Entity name ──────────────────────────────────────────────────────
    lines = [
        f"[bold cyan]{brief.entity_name}[/]",
        "",
        f"[bold]Overview[/]",
        brief.overview,
    ]

    # ── Leadership ───────────────────────────────────────────────────────
    if brief.leadership:
        lines.append("")
        lines.append("[bold]Leadership[/]")
        for member in brief.leadership:
            style = _CONFIDENCE_STYLES.get(
                member.source_confidence, "[dim]● unknown[/]"
            )
            lines.append(f"  • [bold]{member.name}[/] — {member.title}  ({style})")
    else:
        lines.append("")
        lines.append("[bold]Leadership[/]  [dim](no specific names identified)[/]")

    # ── Recent activity ──────────────────────────────────────────────────
    if brief.recent_activity:
        lines.append("")
        lines.append("[bold]Recent Activity[/]")
        for item in brief.recent_activity:
            lines.append(f"  • {item}")
    else:
        lines.append("")
        lines.append("[bold]Recent Activity[/]  [dim](no recent activity found)[/]")

    # ── Past deals ───────────────────────────────────────────────────────
    if brief.past_deals:
        lines.append("")
        lines.append("[bold]Past Deals / Investments[/]")
        for deal in brief.past_deals:
            lines.append(f"  • {deal}")
    else:
        lines.append("")
        lines.append("[bold]Past Deals[/]  [dim](no past deals documented)[/]")

    # ── Human-verified claims ──────────────────────────────────────────
    if brief.human_verified_claims:
        lines.append("")
        lines.append("[bold green]Human-Verified Claims[/]")
        for claim in brief.human_verified_claims:
            lines.append(f"  [green]✓[/] {claim}")

    # ── Open questions ───────────────────────────────────────────────────
    if brief.open_questions:
        lines.append("")
        lines.append("[bold yellow]Open Questions & Uncertainties[/]")
        for q in brief.open_questions:
            lines.append(f"  [yellow]?[/] {q}")
    else:
        lines.append("")
        lines.append("[bold green]Open Questions[/]  [dim](none — all claims verified)[/]")

    # ── Footer ───────────────────────────────────────────────────────────
    lines.append("")
    sources_str = ", ".join(f"[cyan]{s}[/]" for s in brief.sources_used)
    lines.append(f"[dim]Sources used:[/] {sources_str}")
    lines.append(f"[dim]Generated:[/] {brief.generated_at}")

    report_text = "\n".join(lines)

    console.print()
    console.print(
        Panel(
            report_text,
            title="[bold]Due-Diligence Brief[/]",
            border_style="cyan",
            width=100,
            padding=(1, 2),
        )
    )
    console.print()


# ---------------------------------------------------------------------------
# format_ic_memo  —  one-page Investment Committee Memo (institutional LP)
# ---------------------------------------------------------------------------


def format_ic_memo(brief: DiligenceBrief, mandate_result: dict | None = None) -> str:
    """Reformat a DiligenceBrief into a one-page Investment Committee Memo.

    This is a pure reformatting function — it does NOT call the LLM.  It
    uses the data already gathered in the brief to produce an institutional-
    grade memo with sections for:
      - Executive Summary
      - Investment Thesis Considerations
      - Risk Factors
      - Mandate Fit (if mandate_result supplied)
      - Recommendation

    Recommendation rules (applied in order):
      1. 3+ unresolved open_questions → "Insufficient Information"
      2. mandate score < 50            → "Decline"
      3. mandate score 50–75           → "Proceed with Conditions"
      4. mandate score 75+             ��� "Proceed"

    Args:
        brief: A validated DiligenceBrief instance.
        mandate_result: Optional dict from match_mandate() with keys
            ``score``, ``reasoning``, ``uncertain_fields``.

    Returns:
        A plain-text memo string suitable for display and download.
    """
    lines: list[str] = []

    # ── Header ──────────────────────────────────────────────────────────────
    lines.append("=" * 72)
    lines.append("INVESTMENT COMMITTEE MEMO")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"Entity:               {brief.entity_name}")
    lines.append(f"Memo prepared:        {_now_iso()}")
    sources_str = ", ".join(brief.sources_used) if brief.sources_used else "N/A"
    lines.append(f"Data sources:          {sources_str}")
    lines.append("")

    # ── Executive Summary ───────────────────────────────────────────────────
    lines.append("-" * 72)
    lines.append("1. EXECUTIVE SUMMARY")
    lines.append("-" * 72)
    lines.append("")
    # Synthesise a 2–3 sentence summary from the overview
    overview = brief.overview.strip()
    if not overview.endswith("."):
        overview += "."
    # Pull recent-activity highlights
    recent_highlights = brief.recent_activity[:2] if brief.recent_activity else []
    thesis_hints = []
    if brief.past_deals:
        thesis_hints.append(
            f"The firm has a track record of {len(brief.past_deals)} "
            f"notable transaction(s) / investment(s)."
        )
    if brief.leadership:
        names = [m.name for m in brief.leadership[:3]]
        thesis_hints.append(
            f"Leadership includes {', '.join(names)}."
        )
    summary_parts = [overview] + thesis_hints
    lines.append(" ".join(summary_parts))
    lines.append("")

    # ── Investment Thesis Considerations ────────────────────────────────────
    lines.append("-" * 72)
    lines.append("2. INVESTMENT THESIS CONSIDERATIONS")
    lines.append("-" * 72)
    lines.append("")
    if brief.recent_activity:
        lines.append("Recent signal:")
        for item in brief.recent_activity:
            lines.append(f"  • {item}")
        lines.append("")
    else:
        lines.append("  (No recent activity recorded.)")
        lines.append("")
    if brief.past_deals:
        lines.append("Historical track record:")
        for deal in brief.past_deals:
            lines.append(f"  • {deal}")
    else:
        lines.append("  (No past deals documented.)")
    lines.append("")
    if brief.leadership:
        lines.append("Key team members:")
        for m in brief.leadership:
            conf_label = {"high": "verified", "medium": "corroborated", "low": "unverified"}.get(
                m.source_confidence, "unknown"
            )
            lines.append(f"  • {m.name} — {m.title}  [{conf_label}]")
    else:
        lines.append("  (No specific leaders identified.)")
    lines.append("")

    # ── Risk Factors ────────────────────────────────────────────────────────
    lines.append("-" * 72)
    lines.append("3. RISK FACTORS")
    lines.append("-" * 72)
    lines.append("")
    risk_factors: list[str] = []

    # Open questions
    if brief.open_questions:
        for q in brief.open_questions:
            risk_factors.append(f"• [Information Gap] {q}")
    else:
        risk_factors.append("• No material information gaps identified.")

    # Low-confidence leadership entries
    low_conf_leaders = [
        m for m in brief.leadership if m.source_confidence == "low"
    ]
    if low_conf_leaders:
        for m in low_conf_leaders:
            risk_factors.append(
                f"• [Unverified Team Data] Role of {m.name} ({m.title}) "
                f"has not been independently verified."
            )

    for rf in risk_factors:
        lines.append(f"  {rf}")
    if not risk_factors:
        lines.append("  No risk factors identified at this stage.")
    lines.append("")

    # ── Mandate Fit ─────────────────────────────────────────────────────────
    if mandate_result:
        lines.append("-" * 72)
        lines.append("4. MANDATE FIT ASSESSMENT")
        lines.append("-" * 72)
        lines.append("")
        score = mandate_result.get("score", 0)
        reasoning = mandate_result.get("reasoning", [])
        uncertain = mandate_result.get("uncertain_fields", [])
        lines.append(f"  Overall fit score:  {score}/100")
        lines.append("")
        if reasoning:
            lines.append("  Field-level breakdown:")
            for r in reasoning:
                field = r.get("field", "?")
                verdict = r.get("verdict", "?")
                detail = r.get("detail", "")
                lines.append(f"    {field}: {verdict.upper()}")
                lines.append(f"      {detail}")
            lines.append("")
        if uncertain:
            lines.append(
                f"  ⚠ Fields with insufficient data: "
                f"{', '.join(uncertain)}"
            )
        lines.append("")
    else:
        lines.append("-" * 72)
        lines.append("4. MANDATE FIT ASSESSMENT")
        lines.append("-" * 72)
        lines.append("")
        lines.append("  (No mandate criteria provided for assessment.)")
        lines.append("")

    # ── Recommendation ──────────────────────────────────────────────────────
    lines.append("-" * 72)
    lines.append("5. RECOMMENDATION")
    lines.append("-" * 72)
    lines.append("")

    unresolved_count = len(brief.open_questions)
    mandate_score = (mandate_result or {}).get("score", None)

    if unresolved_count >= 3:
        recommendation = "Insufficient Information"
        rationale = (
            f"{unresolved_count} unresolved open question(s) remain. "
            f"Additional research is required before the committee "
            f"can reach a decision."
        )
    elif mandate_score is not None:
        if mandate_score < 50:
            recommendation = "Decline"
            rationale = (
                f"Mandate fit score of {mandate_score}/100 is below "
                f"the minimum threshold. The researched entity does not "
                f"align with the stated investment mandate."
            )
        elif mandate_score < 75:
            recommendation = "Proceed with Conditions"
            rationale = (
                f"Mandate fit score of {mandate_score}/100 meets the "
                f"conditional threshold, but certain criteria require "
                f"closer review or additional data before full commitment."
            )
        else:
            recommendation = "Proceed"
            rationale = (
                f"Mandate fit score of {mandate_score}/100 is strong. "
                f"The entity aligns well with the stated investment mandate."
            )
    else:
        # No mandate provided — base recommendation on open questions only
        if unresolved_count > 0:
            recommendation = "Insufficient Information"
            rationale = (
                f"{unresolved_count} open question(s) remain unresolved. "
                f"A mandate assessment is also needed for a complete view."
            )
        else:
            recommendation = "Proceed with Conditions"
            rationale = (
                f"No significant information gaps identified, but a "
                f"mandate fit assessment is recommended before a final decision."
            )

    lines.append(f"  Recommendation:  {recommendation}")
    lines.append("")
    lines.append(f"  Rationale:       {rationale}")
    lines.append("")
    lines.append("=" * 72)
    lines.append("END OF MEMO")
    lines.append("=" * 72)

    return "\n".join(lines)
