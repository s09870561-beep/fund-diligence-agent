"""Observability logger — writes structured event logs in JSONL format."""

import json
import os
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")


def _run_timestamp() -> str:
    """Return a compact, sortable timestamp for the run filename."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class Tracer:
    """Structured event logger for one fund-diligence run.

    Events logged:
      - goal          The due-diligence question
      - plan          The structured research plan
      - llm_call      Every LLM request (model, tokens, latency)
      - tool_call     Every tool call with args and result
      - finding       A structured research finding (fund, claim, source, confidence)
      - uncertainty   A flagged uncertainty / red flag
      - summary       Final brief / answer
      - retry         Retry attempts on failures
      - error         Error events
    """

    def __init__(self, goal: str):
        os.makedirs(LOG_DIR, exist_ok=True)
        ts = _run_timestamp()
        self.path = os.path.join(LOG_DIR, f"run_{ts}.jsonl")
        self._file = open(self.path, "w", encoding="utf-8")
        self._write("goal", {"goal": goal})

    # ------------------------------------------------------------------
    def _write(self, event: str, data: dict) -> None:
        entry = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "event": event,
            **data,
        }
        self._file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._file.flush()

    # ------------------------------------------------------------------
    def log_plan(self, steps: list[dict]) -> None:
        self._write("plan", {"steps": steps})

    def log_llm_call(
        self,
        model: str,
        tokens_input: int | None,
        tokens_output: int | None,
        latency_sec: float,
        purpose: str = "",
    ) -> None:
        self._write(
            "llm_call",
            {
                "model": model,
                "tokens_input": tokens_input,
                "tokens_output": tokens_output,
                "latency_sec": round(latency_sec, 3),
                "purpose": purpose,
            },
        )

    def log_tool_call(
        self,
        tool: str,
        args: dict,
        result_preview: str,
        duration_sec: float,
    ) -> None:
        self._write(
            "tool_call",
            {
                "tool": tool,
                "args": args,
                "result_preview": result_preview[:500],
                "duration_sec": round(duration_sec, 3),
            },
        )

    def log_finding(
        self,
        fund: str,
        claim: str,
        source: str,
        confidence: str,
    ) -> None:
        self._write(
            "finding",
            {
                "fund": fund,
                "claim": claim,
                "source": source,
                "confidence": confidence,
            },
        )

    def log_uncertainty(self, claim: str, reason: str) -> None:
        self._write("uncertainty", {"claim": claim, "reason": reason})

    def log_summary(self, summary: str) -> None:
        self._write("summary", {"summary": summary})

    def log_error(self, message: str) -> None:
        self._write("error", {"error": message})

    def log_retry(self, attempt: int, max_retries: int, error: str) -> None:
        self._write(
            "retry",
            {
                "attempt": attempt,
                "max_retries": max_retries,
                "error": str(error),
            },
        )

    def log_run_guard(self, stats: dict) -> None:
        """Log a RunGuard summary snapshot."""
        self._write("run_guard", {"stats": stats})

    def log_review_decision(self, question: str, action: str, corrected_text: str | None = None) -> None:
        """Log a human review decision on an open question."""
        self._write(
            "review_decision",
            {
                "question": question,
                "action": action,
                "corrected_text": corrected_text or "",
            },
        )

    def log_brief_summary(self, brief: dict) -> None:
        """Log the final DiligenceBrief summary."""
        self._write(
            "brief_summary",
            {
                "entity_name": brief.get("entity_name", ""),
                "overview": brief.get("overview", "")[:300],
                "leadership_count": len(brief.get("leadership", [])),
                "recent_activity_count": len(brief.get("recent_activity", [])),
                "past_deals_count": len(brief.get("past_deals", [])),
                "open_questions_count": len(brief.get("open_questions", [])),
                "human_verified_claims_count": len(brief.get("human_verified_claims", [])),
            },
        )

    # ------------------------------------------------------------------
    def close(self) -> str:
        self._file.close()
        return self.path

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
