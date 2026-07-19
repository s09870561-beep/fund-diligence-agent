"""Tool layer — tool implementations and OpenAI function-calling schemas.

This module defines every action the agent can take, along with the
structured schemas exposed to the LLM via OpenAI function-calling:

  TOOL_REGISTRY — a dict mapping tool name → OpenAI tool schema dict,
  suitable for passing directly into ``tools`` parameter of chat
  completions.

  execute_tool(name, args, tracer) -> dict
    Dispatches tool calls to the correct implementation and returns a
    uniform result dict regardless of which tool ran.

Available tools:
  - web_search        — search the web (calls retrieval.web_search)
  - sec_edgar_lookup  — look up SEC EDGAR 10-K filings (calls
                        retrieval.sec_edgar_lookup)
  - recall_memory     — query past research sessions via vector memory
"""

import time

from dotenv import load_dotenv
from rich.console import Console

from retrieval import web_search, sec_edgar_lookup
from memory import recall_memory

load_dotenv(override=True)

console = Console()

# ---------------------------------------------------------------------------
# Tool schemas  (OpenAI function-calling format)
# ---------------------------------------------------------------------------

TOOL_REGISTRY: dict[str, dict] = {
    "web_search": {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current information about a company, "
                "fund, manager, or industry.  Uses the Tavily search API "
                "to return recent news, articles, and profiles."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string.",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    "sec_edgar_lookup": {
        "type": "function",
        "function": {
            "name": "sec_edgar_lookup",
            "description": (
                "Search SEC EDGAR for recent 10-K filings mentioning a "
                "specific company name.  Returns filing metadata including "
                "CIK, filing date, period ending, and a direct link to the "
                "filing on sec.gov."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "description": (
                            "The company name to search for (e.g. "
                            '"Apple Inc." or "Tesla, Inc.").'
                        ),
                    },
                },
                "required": ["company_name"],
                "additionalProperties": False,
            },
        },
    },
    "recall_memory": {
        "type": "function",
        "function": {
            "name": "recall_memory",
            "description": (
                "Retrieve findings, notes, or briefs saved during earlier "
                "research sessions about a fund or company. Useful for "
                "avoiding repeated lookups."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A topic or fund name to search past findings for.",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
}


# ---------------------------------------------------------------------------
# Dispatch map  —  tool name → callable
# ---------------------------------------------------------------------------

_DISPATCH: dict[str, callable] = {
    "web_search": lambda query: web_search(query),
    "sec_edgar_lookup": lambda company_name: sec_edgar_lookup(company_name),
    "recall_memory": lambda query, top_k=3: recall_memory(query, top_k=top_k),
}


# ---------------------------------------------------------------------------
# execute_tool
# ---------------------------------------------------------------------------

def execute_tool(tool_name: str, args: dict, tracer=None) -> dict:
    """Execute a named tool by dispatching to its implementation.

    Args:
        tool_name: One of the keys in ``TOOL_REGISTRY``.
        args: Keyword arguments to pass to the tool implementation.
        tracer: Optional ``Tracer`` instance for logging.

    Returns:
        A dict with keys:
          - source       — the tool name that ran
          - success      — bool
          - data         — result text (empty string on failure)
          - error        — error message, or ``None`` on success
          - duration_sec — wall-clock time spent in the tool

    Raises:
        ValueError: If ``tool_name`` is not in the registry.
    """
    if tool_name not in TOOL_REGISTRY:
        raise ValueError(
            f"Unknown tool: '{tool_name}'. "
            f"Available: {list(TOOL_REGISTRY.keys())}"
        )

    fn = _DISPATCH.get(tool_name)
    if fn is None:
        raise ValueError(
            f"Tool '{tool_name}' is in the registry but has no dispatch handler."
        )

    t0 = time.time()

    try:
        result = fn(**args)
        dur = time.time() - t0

        if tracer:
            tracer.log_tool_call(
                tool=tool_name,
                args=args,
                result_preview=(result.get("data") or result.get("error") or "")[:500],
                duration_sec=round(dur, 3),
            )

        return {
            **result,
            "duration_sec": round(dur, 3),
        }

    except Exception as e:
        dur = time.time() - t0
        error_msg = f"{type(e).__name__}: {e}"

        if tracer:
            tracer.log_tool_call(
                tool=tool_name,
                args=args,
                result_preview=f"ERROR: {error_msg[:300]}",
                duration_sec=round(dur, 3),
            )

        return {
            "source": tool_name,
            "success": False,
            "data": "",
            "error": error_msg,
            "duration_sec": round(dur, 3),
        }
