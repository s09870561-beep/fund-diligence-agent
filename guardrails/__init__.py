"""Guardrails layer — uncertainty flagging, run budgets, and ceilings.

This module enforces operational limits on agent runs and provides
explicit signalling when the agent encounters situations it should not
handle autonomously.

  - review_open_questions(brief) -> DiligenceBrief
    If the brief has open questions, presents each one to the user
    for confirm/reject/edit.  Returns a new brief with resolved
    uncertainties.

  - RunLimitExceeded(ceiling, limit, actual)
    Exception raised when a run hits a ceiling.

  - RunGuard
    Tracks step count, tool calls, token usage, cost, and elapsed time
    for one agent run.  Provides check_*_ceiling() methods and loop
    detection.
"""

import os
import time
from datetime import datetime, timezone

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from presentation import DiligenceBrief, format_brief

console = Console()

# ---------------------------------------------------------------------------
# RunLimitExceeded
# ---------------------------------------------------------------------------


class RunLimitExceeded(Exception):
    """Raised when a run exceeds one of its configured ceilings.

    Attributes:
        ceiling: Name of the ceiling hit (e.g. ``"step_ceiling"``).
        limit:   The configured limit that was exceeded.
        actual:  The actual value that exceeded the limit.
    """

    def __init__(self, ceiling: str, limit, actual):
        self.ceiling = ceiling
        self.limit = limit
        self.actual = actual
        super().__init__(f"Run limit exceeded: {ceiling} (limit={limit}, actual={actual})")


# ---------------------------------------------------------------------------
# RunGuard
# ---------------------------------------------------------------------------


class RunGuard:
    """Tracks run budgets and enforces ceilings for a single agent run.

    Usage::

        guard = RunGuard()
        # ... after each step or tool call ...
        guard.record_step()
        guard.record_tool_call("web_search", {"query": "..."})
        guard.check_step_ceiling(max_steps=6)
        guard.check_time_ceiling(max_seconds=120)
    """

    # Simulated per-token costs (the model is free; these let us exercise cost ceilings)
    COST_PER_INPUT_TOKEN = 0.000_001       # $1 / million input tokens
    COST_PER_OUTPUT_TOKEN = 0.000_002      # $2 / million output tokens

    def __init__(self):
        self.step_count = 0
        self.tool_call_count = 0
        self.total_tokens = 0
        self.estimated_cost = 0.0
        self.start_time = time.time()
        self._recent_tool_calls: list[tuple[str, dict]] = []
        self._stopped = False

    # -- Recording ----------------------------------------------------------

    def record_step(self) -> None:
        """Increment the step counter."""
        self.step_count += 1

    def record_tool_call(self, tool_name: str, args: dict) -> None:
        """Record a tool call for counting and loop detection."""
        self.tool_call_count += 1
        self._recent_tool_calls.append((tool_name, _args_key(args)))
        # Keep only the last 5 for loop detection
        while len(self._recent_tool_calls) > 5:
            self._recent_tool_calls.pop(0)

    def record_llm_usage(self, tokens_input: int, tokens_output: int) -> None:
        """Record token consumption from an LLM response and update cost."""
        self.total_tokens += tokens_input + tokens_output
        self.estimated_cost += tokens_input * self.COST_PER_INPUT_TOKEN
        self.estimated_cost += tokens_output * self.COST_PER_OUTPUT_TOKEN

    # -- Ceiling checks -----------------------------------------------------

    def check_step_ceiling(self, max_steps: int) -> None:
        """Raise ``RunLimitExceeded`` if step_count > *max_steps*."""
        if self.step_count > max_steps:
            raise RunLimitExceeded("step_ceiling", max_steps, self.step_count)

    def check_time_ceiling(self, max_seconds: int) -> None:
        """Raise ``RunLimitExceeded`` if elapsed time > *max_seconds*."""
        elapsed = time.time() - self.start_time
        if elapsed > max_seconds:
            raise RunLimitExceeded("time_ceiling", max_seconds, round(elapsed, 1))

    def check_cost_ceiling(self, max_cost: float) -> None:
        """Raise ``RunLimitExceeded`` if estimated cost > *max_cost*."""
        if self.estimated_cost > max_cost:
            raise RunLimitExceeded(
                "cost_ceiling", max_cost, round(self.estimated_cost, 6)
            )

    def check_all(self, max_steps: int, max_seconds: int, max_cost: float) -> None:
        """Convenience: check all three ceilings at once."""
        self.check_step_ceiling(max_steps)
        self.check_time_ceiling(max_seconds)
        self.check_cost_ceiling(max_cost)

    # -- Loop detection -----------------------------------------------------

    def detect_loop(self, recent_tool_calls: list[tuple[str, dict]] | None = None) -> bool:
        """Return ``True`` if the same tool was called with near-identical args 3+ times in a row.

        Args:
            recent_tool_calls: Optional override list; defaults to internal history.

        Returns:
            ``True`` if a repetitive pattern is detected.
        """
        calls = recent_tool_calls if recent_tool_calls is not None else self._recent_tool_calls
        if len(calls) < 3:
            return False
        last_three = calls[-3:]
        names = [c[0] for c in last_three]
        if len(set(names)) > 1:
            return False
        args_list = [c[1] for c in last_three]
        return all(a == args_list[0] for a in args_list)

    # -- Status -------------------------------------------------------------

    def elapsed(self) -> float:
        """Wall-clock seconds since the guard was created."""
        return time.time() - self.start_time

    def summary(self) -> dict:
        """Return a snapshot of all tracking fields."""
        return {
            "step_count": self.step_count,
            "tool_call_count": self.tool_call_count,
            "total_tokens": self.total_tokens,
            "estimated_cost": round(self.estimated_cost, 6),
            "elapsed_sec": round(self.elapsed(), 2),
        }

    def summary_table(self) -> Table:
        """Return a rich Table summarising the run state."""
        t = Table(title="[bold]RunGuard Summary[/]", border_style="cyan")
        t.add_column("Metric", style="bold")
        t.add_column("Value", style="cyan")
        t.add_row("Steps completed", str(self.step_count))
        t.add_row("Tool calls", str(self.tool_call_count))
        t.add_row("Total tokens", str(self.total_tokens))
        t.add_row("Estimated cost", f"${round(self.estimated_cost, 6)}")
        t.add_row("Elapsed", f"{round(self.elapsed(), 1)}s")
        return t


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _args_key(args: dict) -> str:
    """Normalise an args dict into a stable string for loop detection."""
    return str(sorted((k, str(v)) for k, v in sorted((args or {}).items())))


# ---------------------------------------------------------------------------
# review_open_questions  (unchanged from Stage 6)
# ---------------------------------------------------------------------------


def review_open_questions(brief: DiligenceBrief, tracer=None) -> DiligenceBrief:
    """Walk the user through each open question and apply their decisions.

    For each item in ``brief.open_questions`` the user is shown the
    uncertain claim and asked to:

      [c]  Confirm it is correct → moved to ``human_verified_claims``
      [r]  Reject / remove it    → deleted from the brief entirely
      [e]  Edit it with a correction → the corrected text replaces
           the original

    If ``brief.open_questions`` is empty, the brief is returned
    unchanged with an informational message.

    Args:
        brief: The ``DiligenceBrief`` to review.

    Returns:
        A new ``DiligenceBrief`` with all open questions resolved.
    """
    if not brief.open_questions:
        console.print()
        console.print(
            Panel(
                "[bold green]No open questions — brief is fully confirmed.[/]",
                border_style="green",
            )
        )
        console.print()
        return brief

    # -- Auto-approve for non-interactive runs (evals, piped) -----------
    auto = os.getenv("AUTO_APPROVE_REVIEW", "").strip().lower()
    if auto in ("y", "yes", "1"):
        updated = brief.model_copy(deep=True)
        for question in updated.open_questions:
            updated.human_verified_claims.append(question)
            if tracer:
                tracer.log_review_decision(question, "confirmed")
        updated.open_questions = []
        console.print("  [dim]All open questions auto-confirmed (AUTO_APPROVE_REVIEW=y).[/]")
        return updated

    # Work on a mutable copy
    updated = brief.model_copy(deep=True)

    console.print()
    console.print(
        Panel(
            "[bold yellow]Human Review Required[/]\n\n"
            f"The model flagged [bold]{len(updated.open_questions)}[/] "
            f"uncertain claim(s) in this brief.  Please review each one:",
            border_style="yellow",
        )
    )

    resolved = []
    remaining_questions = []

    for idx, question in enumerate(updated.open_questions, 1):
        console.print()
        console.print(
            Panel(
                f"[bold]Question {idx}/{len(updated.open_questions)}[/]\n\n"
                f"{question}\n\n"
                f"[dim]Type (c)onfirm, (r)eject, or (e)dit:[/]",
                title="[bold yellow]Uncertain Claim[/]",
                border_style="yellow",
                width=100,
            )
        )

        answer = Prompt.ask(
            "[bold]Action[/]",
            choices=["c", "r", "e", "confirm", "reject", "edit"],
            default="c",
        )

        if answer in ("c", "confirm"):
            updated.human_verified_claims.append(question)
            resolved.append(("confirmed", question, None))
            if tracer:
                tracer.log_review_decision(question, "confirmed")
            console.print(
                f"  [green]✓ Confirmed — added to human-verified claims.[/]"
            )

        elif answer in ("r", "reject"):
            resolved.append(("rejected", question, None))
            if tracer:
                tracer.log_review_decision(question, "rejected")
            console.print(f"  [dim]✗ Removed from brief.[/]")

        elif answer in ("e", "edit"):
            correction = Prompt.ask(
                "[bold cyan]Your correction[/]",
                default=question,
            )
            # The corrected text goes back into the relevant section.
            # We append it to human_verified_claims since it's now reviewed.
            updated.human_verified_claims.append(correction)
            resolved.append(("edited", question, correction))
            if tracer:
                tracer.log_review_decision(question, "edited", correction)
            console.print(
                f"  [green]✓ Corrected version saved to human-verified claims.[/]"
            )

    # --- Build the final brief ---------------------------------------------
    # Clear open_questions — everything was resolved
    updated.open_questions = []

    console.print()
    console.print(
        Panel(
            "[bold green]All open questions resolved — "
            "brief is now human-reviewed.[/]",
            border_style="green",
        )
    )

    return updated


def _tag_text(text: str, tag: str) -> str:
    """Wrap text in a tag badge (handy for confirmed / edited labels)."""
    return f"{text}  [dim][{tag}][/]"
