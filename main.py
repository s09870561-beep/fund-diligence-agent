"""Fund Diligence Agent — main entry point and pipeline logic.

Orchestrates the full due-diligence research pipeline:
  guardrails/setup  →  reasoning/plan  →  retrieval/gather  →  tools/execute
  →  reasoning/synthesise  →  guardrails/check  →  presentation/format

Run:
    python main.py "Evaluate Fund XYZ: strategy, track record, and risks"

Uses OpenCode Zen (deepseek-v4-flash-free) for all LLM calls.
"""

import os
import re
import sys

from dotenv import load_dotenv

load_dotenv(override=True)

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from reasoning import create_plan, select_tool
from tools import execute_tool, TOOL_REGISTRY
from presentation import synthesize_brief, format_brief, DiligenceBrief
from guardrails import review_open_questions, RunGuard, RunLimitExceeded
from utils.tracer import Tracer

console = Console()

# ---------------------------------------------------------------------------
# Available tools
# ---------------------------------------------------------------------------

AVAILABLE_TOOLS = ["web_search", "sec_edgar_lookup", "recall_memory"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_entity(goal: str) -> str:
    """Extract the company/fund name from a research goal.

    Handles patterns like "Research Acme Corp: ...", "Evaluate Fund X: ...",
    or falls back to a reasonable prefix.
    """
    for prefix in ("Research ", "Evaluate ", "Analyze "):
        if goal.startswith(prefix):
            rest = goal[len(prefix):]
            m = re.match(r"([^:,\.\?]+)", rest)
            if m:
                return m.group(1).strip()
    return goal.split(":")[0].strip()


def _build_queries(entity: str) -> list[tuple[str, str]]:
    """Build search queries for a given entity name."""
    return [
        (f"{entity} recent news activity 2025 2026", "news"),
        (f"{entity} leadership partners team", "leadership"),
        (f"{entity} past deals investments portfolio", "past deals"),
    ]


def _show_retrieval(result: dict) -> None:
    """Print a retrieval result in a formatted Panel."""
    outcome_style = "[green]✔[/]" if result["success"] else "[red]✘[/]"
    source_style = "[cyan]" + result["source"] + "[/]"
    dur = result.get("duration_sec", None)
    dur_str = f"  [dim]Duration:[/] {dur:.1f}s" if dur is not None else ""
    fallback_str = (
        f"  [dim]Used fallback:[/] {'[yellow]Yes[/]' if result.get('used_fallback') else 'No'}"
        if "used_fallback" in result
        else ""
    )
    attempts_str = (
        f"  [dim]Attempts:[/] {result['attempts']}"
        if "attempts" in result
        else ""
    )

    data = result.get("data", "")
    data_preview = data[:1000] if data else "[dim](empty)[/]"

    retrieval_panel = Panel(
        f"{outcome_style}  "
        f"Answered by: {source_style}"
        f"{dur_str}"
        f"{fallback_str}"
        f"{attempts_str}\n\n"
        f"{'─' * 60}\n"
        f"{data_preview}\n"
        f"{'─' * 60}\n"
        + (f"\n[red]Error: {result['error']}[/]" if result.get("error") else ""),
        border_style="magenta",
        width=100,
    )
    console.print(retrieval_panel)
    console.print()


def _check_and_record(guard: RunGuard, max_steps: int, max_time: int, max_cost: float,
                      *, step: bool = False, tool: tuple[str, dict] | None = None):
    """Record a step and/or tool call, then check ceilings and loop detection.

    Raises ``RunLimitExceeded`` if any ceiling is hit.
    """
    if step:
        guard.record_step()
    if tool:
        guard.record_tool_call(*tool)
        if guard.detect_loop():
            console.print("  [yellow]⚠ Possible loop detected — same tool called repeatedly with same args.[/]")
    guard.check_all(max_steps, max_time, max_cost)


# ---------------------------------------------------------------------------
# run_pipeline  —  callable from main.py or evals/run_evals.py
# ---------------------------------------------------------------------------


def run_pipeline(
    goal: str,
    tracer: Tracer | None = None,
    max_steps: int = 6,
    max_time: int = 120,
    max_cost: float = 1.0,
    *,
    entity_override: str | None = None,
    quiet: bool = False,
) -> DiligenceBrief:
    """Execute the full due-diligence pipeline and return a DiligenceBrief.

    Args:
        goal: The research question (e.g. "Research Apple Inc.: ...").
        tracer: Optional ``Tracer`` for event logging.
        max_steps: Hard ceiling on pipeline steps (default 6).
        max_time: Hard ceiling on wall-clock seconds (default 120).
        max_cost: Hard ceiling on simulated cost in USD (default 1.0).
        entity_override: Explicit entity name (auto-extracted from goal if omitted).
        quiet: If True, suppress most terminal output (for evals).

    Returns:
        A validated ``DiligenceBrief`` (already run through review_open_questions
        with auto-approve).

    Raises:
        RunLimitExceeded: If a ceiling was hit.
    """
    guard = RunGuard()
    entity = entity_override or _extract_entity(goal)
    gathered = []
    brief_obj = None

    def _log(s: str):
        if not quiet:
            console.print(s)

    try:
        # -- Step 1: Plan ----------------------------------------------------
        _log("[dim]Creating research plan ...[/]")
        plan = create_plan(goal, max_steps=6, tracer=tracer)
        if create_plan.last_usage:
            guard.record_llm_usage(create_plan.last_usage["input"], create_plan.last_usage["output"])
        _check_and_record(guard, max_steps, max_time, max_cost, step=True)

        # -- Step 2: Gather evidence -----------------------------------------
        if not quiet:
            console.print()
            console.print(Panel("[bold]Gathering evidence …[/]", border_style="cyan"))

        queries = _build_queries(entity)
        for query, label in queries:
            _log(f"  [dim]→ {label} ...[/]")
            result = execute_tool("web_search", {"query": query}, tracer=tracer)
            gathered.append({**result, "label": label})
            if not quiet:
                _show_retrieval(result)
            _check_and_record(guard, max_steps, max_time, max_cost, step=True,
                              tool=("web_search", {"query": query[:40]}))

        # Also try SEC EDGAR
        _log(f"  [dim]→ SEC EDGAR ...[/]")
        edgar_result = execute_tool("sec_edgar_lookup", {"company_name": entity}, tracer=tracer)
        gathered.append({**edgar_result, "label": "SEC filings"})
        if not quiet:
            _show_retrieval(edgar_result)
        _check_and_record(guard, max_steps, max_time, max_cost, step=True,
                          tool=("sec_edgar_lookup", {"company_name": entity}))

        # -- Step 3: Synthesise into a DiligenceBrief ------------------------
        _log("[dim]Synthesizing research brief …[/]")
        brief_obj = synthesize_brief(goal, gathered, tracer=tracer)
        if synthesize_brief.last_usage:
            guard.record_llm_usage(synthesize_brief.last_usage["input"], synthesize_brief.last_usage["output"])
        _check_and_record(guard, max_steps, max_time, max_cost, step=True)

        # -- Step 4: Auto-review open questions (if env var set) -------------
        reviewed = review_open_questions(brief_obj, tracer=tracer)
        _check_and_record(guard, max_steps, max_time, max_cost, step=True)

        # Log RunGuard stats
        if tracer:
            tracer.log_run_guard(guard.summary())

        return reviewed

    except RunLimitExceeded:
        if tracer:
            tracer.log_run_guard(guard.summary())
        raise


# ---------------------------------------------------------------------------
# Main  (interactive / demo entry point)
# ---------------------------------------------------------------------------


def main():
    goal = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
        "Research Sequoia Capital: recent activity, leadership, and past deals."
    )

    max_steps = int(os.getenv("MAX_STEPS", "6"))
    max_time = int(os.getenv("MAX_TIME", "120"))
    max_cost = float(os.getenv("MAX_COST", "1.0"))

    tracer = Tracer(goal)
    brief = None

    try:
        console.print(
            Panel(
                f"[bold cyan]Fund Diligence Agent[/]\n\n"
                f"[dim]Goal:[/] {goal}\n"
                f"[dim]Limits:[/] steps={max_steps}, time={max_time}s, cost=${max_cost}",
                border_style="cyan",
            )
        )
        console.print()

        # Run the core pipeline
        reviewed_brief = run_pipeline(goal, tracer=tracer, max_steps=max_steps,
                                      max_time=max_time, max_cost=max_cost)

        brief = reviewed_brief

        # -- Display the final brief -----------------------------------------
        format_brief(brief)
        tracer.log_brief_summary(brief.model_dump())

        # -- Save to memory --------------------------------------------------
        console.print()
        console.print(Panel("[bold magenta]Memory Test — saving brief to vector store …[/]", border_style="magenta"))

        from memory import save_finding

        entity = brief.entity_name or _extract_entity(goal)
        save_result = save_finding(entity, brief)
        console.print(f"  [green]✓ Saved to memory:[/] {save_result['id']}")

        # -- Test semantic recall --------------------------------------------
        console.print()
        console.print(
            Panel(
                "[bold magenta]Memory Test — recall with semantically different query …[/]\n\n"
                "[dim]Query: \"What do we know about that major venture firm's "
                "recent fund and leadership changes?\"[/]\n"
                "[dim](Intentionally avoids the name 'Sequoia Capital' to "
                "test semantic similarity)[/]",
                border_style="magenta",
            )
        )
        memory_result = execute_tool(
            "recall_memory",
            {"query": f"What do we know about {entity}'s recent activity and leadership?"},
            tracer=tracer,
        )
        _show_retrieval(memory_result)

        # -- Match against sample mandate ------------------------------------
        console.print()
        console.print(Panel("[bold]Mandate Fit — testing against sample mandate …[/]", border_style="cyan"))

        from matching import InvestmentMandate, match_mandate

        sample_mandate = InvestmentMandate(
            sectors=["fintech", "financial services"],
            stage="early-stage (Series A and earlier)",
            check_size_min=1_000_000,
            check_size_max=5_000_000,
            geography=["United States"],
            excluded_industries=["cryptocurrency", "gambling"],
        )

        console.print(f"[dim]Mandate:[/] fintech / early-stage / $1–5M / US only")
        console.print()

        match_result = match_mandate(brief, sample_mandate)
        score = match_result.get("score", 0)
        reasoning = match_result.get("reasoning", [])
        uncertain = match_result.get("uncertain_fields", [])

        score_color = "green" if score >= 70 else "yellow" if score >= 40 else "red"
        console.print(
            Panel(
                f"[bold {score_color}]Fit score: {score}/100[/]\n\n"
                + "\n".join(
                    f"  [{('green' if r['verdict'] == 'match' else 'yellow' if r['verdict'] == 'unclear' else 'red')}] "
                    f"[bold]{r['field']}:[/] {r['verdict']}[/] — {r['detail']}"
                    for r in reasoning
                )
                + (f"\n\n[dim]Uncertain fields: {', '.join(uncertain)}[/]" if uncertain else ""),
                border_style=score_color,
                title="[bold]Mandate Fit Assessment[/]",
                width=100,
            )
        )

        # -- Find connections between entities -------------------------------
        console.print()
        console.print(Panel("[bold]Entity Connections — finding links between Sequoia Capital and Stripe …[/]", border_style="cyan"))

        from relationships import find_connections, print_connections

        conn_result = find_connections("Sequoia Capital", "Stripe")
        print_connections(conn_result)
        console.print()

        tracer.close()

        # -- Complete --------------------------------------------------------
        console.print(
            Panel(
                "[bold green]Run completed successfully[/]",
                border_style="green",
            )
        )

    except RunLimitExceeded as e:
        ceiling_labels = {
            "step_ceiling": "Step limit",
            "time_ceiling": "Time limit",
            "cost_ceiling": "Cost limit",
        }
        label = ceiling_labels.get(e.ceiling, e.ceiling)

        console.print()
        console.print(
            Panel(
                f"[bold red]⛔ Run stopped by {label}[/]\n\n"
                f"[yellow]Ceiling:[/]  {e.ceiling}\n"
                f"[yellow]Limit:[/]    {e.limit}\n"
                f"[yellow]Actual:[/]   {e.actual}",
                border_style="red",
                width=80,
            )
        )

        tracer.close()
        sys.exit(1)


if __name__ == "__main__":
    main()
