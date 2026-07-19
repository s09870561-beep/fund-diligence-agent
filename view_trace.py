"""Trace viewer — pretty-print a JSONL run log as a styled timeline.

Usage:
    python view_trace.py                  # latest log
    python view_trace.py logs/run_....jsonl  # specific file
"""

import json
import os
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.text import Text
from rich import box

console = Console()


def _events(path: str):
    """Yield parsed JSON events from a JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _ts(entry: dict) -> str:
    """Return a short timestamp for an entry."""
    raw = entry.get("timestamp", "")
    if "." in raw:
        raw = raw.split("T")[1][:12]
    else:
        raw = raw.split("T")[1][:8]
    return raw


def _preview(text: str, max_len: int = 300) -> str:
    """Shorten a string for display."""
    if len(text) > max_len:
        return text[:max_len] + " ..."
    return text


def view_trace(path: str) -> None:
    """Pretty-print a JSONL trace file."""
    entries = list(_events(path))
    if not entries:
        console.print("[red]No events found in that file.[/]")
        return

    console.print(
        Panel(
            f"[bold]Trace file:[/] {path}\n"
            f"[bold]Events:[/] {len(entries)}",
            border_style="blue",
            title="[bold blue]Fund Diligence — Run Timeline[/]",
        )
    )

    for i, entry in enumerate(entries):
        event = entry.get("event", "?")
        ts = _ts(entry)

        # -- goal -----------------------------------------------------------
        if event == "goal":
            console.print(
                Panel(
                    entry.get("goal", ""),
                    border_style="blue",
                    title=f"[bold blue]Goal @ {ts}[/]",
                )
            )

        # -- plan -----------------------------------------------------------
        elif event == "plan":
            steps = entry.get("steps", [])
            table = Table(box=box.SIMPLE, border_style="cyan", title=f"Plan @ {ts}")
            table.add_column("#", style="dim", width=4)
            table.add_column("Step", style="bold white")
            for s in steps:
                table.add_row(str(s.get("step", "?")), s.get("action", ""))
            console.print(table)

        # -- llm_call -------------------------------------------------------
        elif event == "llm_call":
            purpose = entry.get("purpose", "llm")
            inp = entry.get("tokens_input")
            out = entry.get("tokens_output")
            lat = entry.get("latency_sec", "?")
            tok_str = ""
            if inp is not None and out is not None:
                tok_str = f" | {inp} in / {out} out tokens"
            console.print(
                Panel(
                    f"[bold]Latency:[/] {lat}s{tok_str}",
                    border_style="green",
                    title=f"[green]LLM call ({purpose}) @ {ts}[/]",
                )
            )

        # -- tool_call ------------------------------------------------------
        elif event == "tool_call":
            tool = entry.get("tool", "?")
            args = entry.get("args", {})
            dur = entry.get("duration_sec", "?")
            result = entry.get("result_preview", "")

            args_str = json.dumps(args, ensure_ascii=False)[:200]
            preview = _preview(result)

            body = (
                f"[bold yellow]Tool:[/] {tool}\n"
                f"[bold yellow]Args:[/] {args_str}\n"
                f"[bold yellow]Duration:[/] {dur}s\n\n"
                f"{preview}"
            )
            console.print(
                Panel(
                    body,
                    border_style="yellow",
                    title=f"Tool call @ {ts}",
                    highlight=False,
                )
            )

        # -- finding --------------------------------------------------------
        elif event == "finding":
            fund = entry.get("fund", "")
            claim = entry.get("claim", "")
            source = entry.get("source", "")
            confidence = entry.get("confidence", "")
            style = {"high": "green", "medium": "yellow", "low": "red"}.get(confidence, "dim")
            console.print(
                Panel(
                    f"[bold]Fund:[/] {fund}\n"
                    f"[bold]Claim:[/] {_preview(claim)}\n"
                    f"[dim]Source:[/] {source}  [dim]Confidence:[/] [{style}]{confidence}[/]",
                    border_style="cyan",
                    title=f"Finding @ {ts}",
                )
            )

        # -- uncertainty ----------------------------------------------------
        elif event == "uncertainty":
            claim = entry.get("claim", "")
            reason = entry.get("reason", "")
            console.print(
                Panel(
                    f"[bold yellow]Claim:[/] {_preview(claim)}\n"
                    f"[yellow]Reason:[/] {_preview(reason)}",
                    border_style="yellow",
                    title=f"Uncertainty @ {ts}",
                )
            )

        # -- run_guard ------------------------------------------------------
        elif event == "run_guard":
            stats = entry.get("stats", {})
            table = Table(
                box=box.SIMPLE,
                border_style="cyan",
                title=f"RunGuard @ {ts}",
            )
            table.add_column("Metric", style="bold")
            table.add_column("Value", style="cyan")
            for k, v in stats.items():
                label = k.replace("_", " ").title()
                if k == "estimated_cost":
                    table.add_row(label, f"${v}")
                else:
                    table.add_row(label, str(v))
            console.print(table)

        # -- review_decision ------------------------------------------------
        elif event == "review_decision":
            question = entry.get("question", "")
            action = entry.get("action", "")
            corrected = entry.get("corrected_text", "")

            action_styles = {
                "confirmed": ("green", "Confirmed"),
                "rejected": ("red", "Rejected"),
                "edited": ("yellow", "Edited"),
            }
            style, label = action_styles.get(action, ("dim", action))

            body = (
                f"[bold]Action:[/] [{style}]{label}[/]\n"
                f"[bold]Question:[/] {_preview(question)}"
            )
            if corrected and action == "edited":
                body += f"\n[bold]Corrected:[/] {_preview(corrected)}"

            console.print(
                Panel(body, border_style=style, title=f"Human Review @ {ts}")
            )

        # -- brief_summary --------------------------------------------------
        elif event == "brief_summary":
            entity = entry.get("entity_name", "")
            overview = entry.get("overview", "")
            ldr = entry.get("leadership_count", 0)
            act = entry.get("recent_activity_count", 0)
            deals = entry.get("past_deals_count", 0)
            oq = entry.get("open_questions_count", 0)
            verified = entry.get("human_verified_claims_count", 0)

            body = (
                f"[bold cyan]{entity}[/]\n\n"
                f"{_preview(overview, 200)}\n\n"
                f"[dim]Leadership:[/] {ldr}    "
                f"[dim]Activity:[/] {act}    "
                f"[dim]Deals:[/] {deals}\n"
                f"[dim]Open questions:[/] {oq}    "
                f"[dim]Verified claims:[/] {verified}"
            )
            console.print(
                Panel(body, border_style="blue", title=f"Brief @ {ts}")
            )

        # -- summary --------------------------------------------------------
        elif event == "summary":
            console.print(
                Panel(
                    entry.get("summary", ""),
                    border_style="green",
                    title=f"[green]Summary @ {ts}[/]",
                )
            )

        # -- retry ----------------------------------------------------------
        elif event == "retry":
            attempt = entry.get("attempt", "?")
            max_r = entry.get("max_retries", "?")
            error = entry.get("error", "")
            console.print(
                Panel(
                    f"Attempt {attempt}/{max_r} failed: {error}",
                    border_style="yellow",
                    title=f"Retry @ {ts}",
                )
            )

        # -- error ----------------------------------------------------------
        elif event == "error":
            console.print(
                Panel(
                    entry.get("error", ""),
                    border_style="red",
                    title=f"[red]Error @ {ts}[/]",
                )
            )

        # -- fallback for unknown event types -------------------------------
        else:
            console.print(
                Panel(
                    json.dumps(entry, indent=2, ensure_ascii=False)[:500],
                    border_style="dim",
                    title=f"{event} @ {ts}",
                )
            )

    console.print(
        f"\n[dim]--- End of trace ({len(entries)} events) ---[/]"
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        if not os.path.isdir(log_dir):
            console.print("[red]No logs/ directory found. Run the agent first.[/]")
            sys.exit(1)
        files = sorted(
            [
                f
                for f in os.listdir(log_dir)
                if f.startswith("run_") and f.endswith(".jsonl")
            ],
            reverse=True,
        )
        if not files:
            console.print("[red]No run_*.jsonl files found in logs/[/]")
            sys.exit(1)
        path = os.path.join(log_dir, files[0])
        console.print(f"[dim]Using latest trace: {path}[/]\n")
    else:
        path = sys.argv[1]

    view_trace(path)
