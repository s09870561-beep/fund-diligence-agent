"""Run the fund-diligence agent against all test cases and score results.

Usage:
    python evals/run_evals.py

Sets AUTO_APPROVE_REVIEW=y so open questions are auto-confirmed.
Each test case is judged via LLM evaluator against its criteria.
The step-ceiling test (test 4) checks that RunLimitExceeded is raised.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv(override=True)

from openai import OpenAI
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from main import run_pipeline
from guardrails import RunLimitExceeded
from utils.tracer import Tracer
from utils.retry import retry_with_backoff

console = Console()

# -- LLM judge client (same model as the agent) -------------------------

_client = OpenAI(
    base_url="https://opencode.ai/zen/v1",
    api_key=os.getenv("OPENCODE_ZEN_API_KEY", "").strip(),
)
MODEL = "deepseek-v4-flash-free"


def judge_result(goal: str, criteria: str, brief_text: str) -> dict:
    """Ask the model whether *brief_text* satisfies *criteria* for *goal*.

    Returns {"pass": bool, "reason": str}.
    """
    judge_prompt = (
        "You are a strict but fair evaluator. Given a research goal, "
        "grading criteria, and the agent's due-diligence brief, determine "
        "whether the brief satisfies the criteria.\n\n"
        "Respond ONLY in JSON — no markdown, no explanation outside the "
        "JSON. Use this exact shape:\n"
        '{"pass": true/false, "reason": "brief explanation of your verdict"}'
    )

    user_msg = (
        f"Goal: {goal}\n\n"
        f"Criteria:\n{criteria}\n\n"
        f"Agent's brief:\n{brief_text}"
    )

    response = retry_with_backoff(
        lambda: _client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": judge_prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
        ),
    )

    if isinstance(response, str):
        return {"pass": False, "reason": f"Judge LLM error: {response}"}

    content = response.choices[0].message.content or ""
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1]
        content = content.rsplit("```", 1)[0].strip()

    try:
        result = json.loads(content)
        if not isinstance(result, dict) or "pass" not in result:
            raise ValueError("Missing 'pass' key")
        return result
    except (json.JSONDecodeError, ValueError) as e:
        return {"pass": False, "reason": f"Judge parse error: {e}"}


def brief_to_text(brief) -> str:
    """Serialize a DiligenceBrief (or None) to a readable string for the judge."""
    if brief is None:
        return "No brief was generated (pipeline stopped before synthesis)."
    try:
        data = brief.model_dump()
    except AttributeError:
        return str(brief)
    parts = [
        f"Entity: {data.get('entity_name', '?')}",
        f"Overview: {data.get('overview', '')[:500]}",
    ]
    leadership = data.get("leadership", [])
    if leadership:
        for m in leadership:
            parts.append(f"  Leader: {m.get('name', '?')} — {m.get('title', '?')} ({m.get('source_confidence', '?')})")
    ra = data.get("recent_activity", [])
    if ra:
        parts.append(f"Recent activity ({len(ra)} items):")
        for a in ra[:5]:
            parts.append(f"  • {a[:120]}")
    pd = data.get("past_deals", [])
    if pd:
        parts.append(f"Past deals ({len(pd)} items):")
        for d in pd[:5]:
            parts.append(f"  • {d[:120]}")
    oq = data.get("open_questions", [])
    if oq:
        parts.append(f"Open questions ({len(oq)} items):")
        for q in oq[:5]:
            parts.append(f"  ? {q[:150]}")
    su = data.get("sources_used", [])
    parts.append(f"Sources used: {', '.join(su) if su else '(none)'}")
    hv = data.get("human_verified_claims", [])
    if hv:
        parts.append(f"Human-verified claims ({len(hv)} items):")
        for c in hv[:5]:
            parts.append(f"  ✓ {c[:200]}")
    else:
        parts.append("Human-verified claims: 0")
    return "\n".join(parts)


def main():
    cases_path = os.path.join(os.path.dirname(__file__), "test_cases.json")
    with open(cases_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    console.print(
        Panel(
            f"[bold cyan]Fund Diligence Agent — Evals[/]\n"
            f"[dim]{len(cases)} test cases loaded[/]",
            border_style="cyan",
        )
    )
    console.print()

    # Auto-approve review questions for all test runs
    os.environ["AUTO_APPROVE_REVIEW"] = "y"

    results = []
    total = len(cases)

    for i, case in enumerate(cases, 1):
        cid = case["id"]
        label = case.get("label", f"Test {cid}")
        goal = case["goal"]
        criteria = case["criteria"]
        entity = case.get("entity", "")
        case_max_steps = case.get("max_steps", 15)
        expect_ceiling = case.get("expect_ceiling", False)

        short = f"[{cid}] {label}"
        console.rule(f"[bold]{short}[/]")
        console.print(f"[dim]Goal:[/] {goal[:80]}...")
        console.print()

        # Run the pipeline with per-case limits
        tracer = Tracer(goal)
        brief = None
        ceiling_hit = False
        ceiling_info = None
        error_msg = None

        try:
            brief = run_pipeline(
                goal,
                tracer=tracer,
                max_steps=case_max_steps,
                max_time=120,
                max_cost=1.0,
                entity_override=entity if entity else None,
                quiet=True,
            )
        except RunLimitExceeded as e:
            ceiling_hit = True
            ceiling_info = f"{e.ceiling}: limit={e.limit}, actual={e.actual}"
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
        finally:
            tracer.close()

        # Determine pass/fail
        if expect_ceiling:
            # Ceiling test: passing means the ceiling WAS triggered
            passed = ceiling_hit
            reason = ceiling_info if ceiling_hit else (error_msg or "Ceiling was NOT triggered (unexpected success)")
        elif ceiling_hit:
            passed = False
            reason = f"Unexpected ceiling hit: {ceiling_info}"
        elif error_msg:
            passed = False
            reason = f"Pipeline error: {error_msg}"
        else:
            # Judge the brief
            brief_text = brief_to_text(brief)
            verdict = judge_result(goal, criteria, brief_text)
            passed = verdict.get("pass", False)
            reason = verdict.get("reason", "No reason given")

        # Structure checks (bonus assertions)
        checks = []
        if brief is not None and not expect_ceiling:
            try:
                data = brief.model_dump()
                if not data.get("entity_name"):
                    checks.append("missing entity_name")
                if not data.get("overview"):
                    checks.append("missing overview")
                if not data.get("sources_used"):
                    checks.append("missing sources_used")
            except Exception:
                checks.append("invalid model")
        if checks:
            reason += " | Structural issues: " + "; ".join(checks)
            if passed:
                passed = False

        results.append({
            "id": cid,
            "label": label,
            "goal": goal,
            "pass": passed,
            "reason": reason[:200],
            "ceiling_hit": ceiling_hit,
            "brief_entity": brief.entity_name if brief else "N/A",
        })

        tag = "[green]PASS[/]" if passed else "[red]FAIL[/]"
        console.print(f"{tag} {reason[:120]}")
        console.print()

    # ------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------
    passed_count = sum(1 for r in results if r["pass"])
    score_pct = round(passed_count / total * 100)

    table = Table(
        title=f"[bold]Eval Results: {passed_count}/{total} passed ({score_pct}%)[/]",
        box=box.ROUNDED,
        border_style="blue",
    )
    table.add_column("ID", style="dim", width=4)
    table.add_column("Test", style="bold white", width=32)
    table.add_column("Entity", style="cyan", width=18)
    table.add_column("Ceiling?", width=9)
    table.add_column("Result", width=8)
    table.add_column("Reason", width=55)

    for r in results:
        label_short = r["label"][:30]
        entity_short = r["brief_entity"][:17]
        ceiling_str = "[yellow]yes[/]" if r["ceiling_hit"] else "no"
        outcome = "[green]PASS[/]" if r["pass"] else "[red]FAIL[/]"
        reason_short = r["reason"][:52] + "..." if len(r["reason"]) > 52 else r["reason"]
        table.add_row(str(r["id"]), label_short, entity_short, ceiling_str, outcome, reason_short)

    console.print()
    console.print(table)
    console.print()

    # Overall score panel
    color = "green" if score_pct >= 70 else "yellow" if score_pct >= 40 else "red"
    console.print(
        Panel(
            f"[bold {color}]{passed_count}/{total} passed ({score_pct}%)[/]",
            border_style=color,
            title="[bold]Overall Score[/]",
        )
    )


if __name__ == "__main__":
    main()
