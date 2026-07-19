"""Reasoning layer — planning, decision logic, and tool-selection.

This module is the agent's "brain". It owns:

  - create_plan(goal) -> dict
    Breaks a due-diligence question into concrete research steps.

  - select_tool(step, available_tools) -> (str, str)
    Decides which tool to invoke for a given plan step, using explicit
    logic where possible and falling back to the LLM only when the
    tool_hint is ambiguous.

  - synthesize(evidence, question) -> str        (stub)
  - critique(answer, evidence) -> dict             (stub)

This layer calls into retrieval/ and tools/ as needed, and uses the
model (OpenCode Zen) for all LLM-driven reasoning.  It does NOT
contain any I/O, search, or persistence logic directly.
"""

import json
import os
import time
from datetime import datetime, timezone

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from utils.retry import retry_with_backoff

console = Console()

# -- OpenCode Zen client -------------------------------------------------

_client = OpenAI(
    base_url="https://opencode.ai/zen/v1",
    api_key=os.getenv("OPENCODE_ZEN_API_KEY", "").strip(),
)

MODEL = "deepseek-v4-flash-free"

# ---------------------------------------------------------------------------
# Plan prompt
# ---------------------------------------------------------------------------

_PLAN_PROMPT = (
    "You are a fund-diligence planner. Break the user's research goal into "
    "concrete, actionable research steps. "
    "Each step must include:\n"
    '  - "step": an integer (starting at 1)\n'
    '  - "action": a clear one-sentence description of what to do\n'
    '  - "tool_hint": a guess at which tool this step will need — '
    'one of "web_search", "sec_edgar_lookup", or "none" '
    '(use "none" for synthesis/analysis steps that don\'t fetch new data)\n\n'
    "Respond ONLY with a valid JSON array of step objects, no markdown, no explanation.\n\n"
    "Example:\n"
    "[\n"
    '  {{"step": 1, "action": "Search for recent news and activity about the fund", '
    '"tool_hint": "web_search"}},\n'
    '  {{"step": 2, "action": "Look up SEC filings to verify AUM and fee structure", '
    '"tool_hint": "sec_edgar_lookup"}},\n'
    '  {{"step": 3, "action": "Synthesise findings into a risk assessment", '
    '"tool_hint": "none"}}\n'
    "]\n"
    "The array must have between 1 and {max_steps} elements."
)


# ---------------------------------------------------------------------------
# create_plan
# ---------------------------------------------------------------------------




def create_plan(goal: str, max_steps: int = 6, tracer=None) -> dict:
    """Break a due-diligence goal into concrete research steps.

    Args:
        goal: The user's research question (e.g. "Evaluate Fund X's strategy...").
        max_steps: Hard cap on the number of steps allowed (default 6).
        tracer: Optional ``Tracer`` instance for logging.

    Returns:
        A dict with keys:
          - goal:        The original goal string.
          - steps:       List of {"step": int, "action": str, "tool_hint": str}.
          - max_steps:   The cap that was enforced.
          - created_at:  ISO-8601 timestamp.

    The token usage from the LLM call is available on
    ``create_plan.last_usage`` (a dict with ``input`` and ``output`` keys,
    or ``None`` if the call failed or usage was unavailable).
    """
    t0 = time.time()

    response = retry_with_backoff(
        lambda: _client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": _PLAN_PROMPT.format(max_steps=max_steps)},
                {"role": "user", "content": goal},
            ],
            temperature=0.3,
        ),
        tracer=tracer,
    )
    latency = time.time() - t0

    if isinstance(response, str):
        console.print(f"[red]Plan LLM call failed: {response}[/]")
        create_plan.last_usage = None
        if tracer:
            tracer.log_error(f"Plan LLM call failed: {response}")
        # Fallback plan
        return {
            "goal": goal,
            "steps": [
                {"step": 1, "action": f"Search for information about: {goal}", "tool_hint": "web_search"},
                {"step": 2, "action": "Synthesise findings into a final answer", "tool_hint": "none"},
            ],
            "max_steps": max_steps,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    usage = getattr(response, "usage", None)
    if usage:
        create_plan.last_usage = {
            "input": usage.prompt_tokens or 0,
            "output": usage.completion_tokens or 0,
        }
        if tracer:
            tracer.log_llm_call(
                model=MODEL,
                tokens_input=usage.prompt_tokens,
                tokens_output=usage.completion_tokens,
                latency_sec=latency,
                purpose="create_plan",
            )
    else:
        create_plan.last_usage = None

    content = response.choices[0].message.content or ""
    content = content.strip()

    # Strip markdown fences if present
    if content.startswith("```"):
        content = content.split("\n", 1)[-1]
        content = content.rsplit("```", 1)[0].strip()

    try:
        steps = json.loads(content)
    except (json.JSONDecodeError, ValueError) as e:
        console.print(f"[red]Failed to parse plan JSON: {e}[/]")
        console.print(f"[dim]Raw response: {content}[/]")
        if tracer:
            tracer.log_error(f"Plan parse failed: {e}")
        steps = []

    # Validate structure
    valid_steps = []
    for s in steps:
        if isinstance(s, dict) and "action" in s:
            valid_steps.append({
                "step": s.get("step", len(valid_steps) + 1),
                "action": s["action"],
                "tool_hint": s.get("tool_hint", "web_search"),
            })
        else:
            console.print(f"[yellow]Skipping malformed step in plan: {s}[/]")

    # --- Hard cap enforcement ---
    if len(valid_steps) > max_steps:
        console.print(
            f"[yellow]Warning: model returned {len(valid_steps)} steps, "
            f"truncating to {max_steps}.[/]"
        )
        if tracer:
            tracer.log_error(
                f"Model returned {len(valid_steps)} steps, truncated to {max_steps}"
            )
        valid_steps = valid_steps[:max_steps]

    # If everything was malformed, use fallback
    if not valid_steps:
        valid_steps = [
            {"step": 1, "action": f"Search for information about: {goal}", "tool_hint": "web_search"},
            {"step": 2, "action": "Synthesise findings into a final answer", "tool_hint": "none"},
        ]

    # Re-number steps to ensure sequential 1..N
    for i, s in enumerate(valid_steps, 1):
        s["step"] = i

    if tracer:
        tracer.log_plan(valid_steps)

    return {
        "goal": goal,
        "steps": valid_steps,
        "max_steps": max_steps,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# select_tool  —  explicit logic with LLM fallback
# ---------------------------------------------------------------------------

_TOOL_HINT_MAP = {
    "web_search": "web_search",
    "sec_edgar_lookup": "sec_edgar_lookup",
    "none": "none",
}

_AMBIGUOUS_HINTS = {"search", "lookup", "research", "find", "fetch", "get", "check", "review", ""}


def select_tool(step: dict, available_tools: list[str], tracer=None) -> tuple[str, str]:
    """Decide which tool to use for a given plan step.

    Uses explicit lookup logic from the step's ``tool_hint`` when the
    mapping is unambiguous.  Falls back to an LLM call only when the
    hint is missing, empty, or does not directly name an available tool.

    Args:
        step: A plan step dict with at least ``tool_hint`` and ``action``.
        available_tools: List of tool names the agent can call.

    Returns:
        (tool_name: str, reason: str) — the selected tool and a
        one-line explanation of why it was chosen (or ``"none"`` for
        synthesis/analysis steps).
    """
    hint = step.get("tool_hint", "").strip().lower()

    # ---- Path A: explicit unambiguous mapping ---------------------------
    if hint in _TOOL_HINT_MAP:
        mapped = _TOOL_HINT_MAP[hint]
        if mapped == "none":
            return ("none", f"Step is synthesis/analysis (tool_hint='{hint}') — no tool needed.")
        if mapped in available_tools:
            return (mapped, f"tool_hint='{hint}' maps directly to available tool '{mapped}'.")
        # mapped tool isn't available — fall through to LLM
        console.print(
            f"[yellow]tool_hint='{hint}' maps to '{mapped}' but it's not in "
            f"available_tools={available_tools}. Falling back to LLM.[/]"
        )

    # ---- Path B: ambiguous or missing hint — ask the model -------------
    # This is the minority case; most steps have an explicit hint from the plan.
    if hint in _AMBIGUOUS_HINTS or hint not in _TOOL_HINT_MAP:
        action = step.get("action", "")
        tool_list = ", ".join(available_tools)
        prompt = (
            f"Given this research step:\n"
            f'  "{action}"\n\n'
            f"Which of the following available tools is the best fit?\n"
            f"  Available: {tool_list}\n\n"
            f'If none are needed (this is a synthesis/analysis step), answer "none".\n'
            f"Respond with ONLY the tool name — nothing else."
        )
        response = retry_with_backoff(
            lambda: _client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a tool-selection assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            ),
            tracer=tracer,
        )

        if isinstance(response, str):
            console.print(f"[red]Tool-selection LLM call failed: {response}[/]")
            return ("web_search", f"Fallback: tool selection LLM failed; defaulting to web_search.")

        chosen = (response.choices[0].message.content or "").strip().lower()
        reason = f"LLM fallback: tool_hint='{hint}' was ambiguous; model chose '{chosen}'."

        if chosen not in available_tools and chosen != "none":
            console.print(
                f"[yellow]LLM chose '{chosen}' which is not available. "
                f"Defaulting to web_search.[/]"
            )
            return ("web_search", f"LLM chose unavailable tool '{chosen}'; defaulted to web_search.")
        return (chosen, reason)

    # ---- Path C: hint mentions a specific tool by name -----------------
    # Check if the hint string contains any available tool name as a substring
    for tool in available_tools:
        if tool in hint:
            return (tool, f"tool_hint='{hint}' contains tool name '{tool}' — explicit match.")

    # Last resort
    return ("web_search", f"tool_hint='{hint}' not recognised; defaulting to web_search.")


# ---------------------------------------------------------------------------
# Stubs for future stages
# ---------------------------------------------------------------------------


def synthesize(evidence: list[str], question: str) -> str:
    """Combine retrieved evidence into a coherent answer or recommendation.

    .. todo:: Implement in a later stage.
    """
    return "[synthesize stub — not yet implemented]"


def critique(answer: str, evidence: list[str]) -> dict:
    """Self-review pass: identify gaps, uncertainty, or unsupported claims.

    .. todo:: Implement in a later stage.
    """
    return {"pass": True, "reason": "[critique stub — not yet implemented]", "fix": ""}
