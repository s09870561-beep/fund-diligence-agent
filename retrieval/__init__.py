"""Retrieval layer — multi-source data gathering.

This module owns all data-collection from external sources:

  - web_search(query) -> dict
    Uses Tavily Search API to find current web results about a fund,
    manager, strategy, or holding.

  - sec_edgar_lookup(company_name) -> dict
    Searches SEC EDGAR's public full-text search API for recent 10-K
    filings mentioning the given company.

  - retrieve(step, tool_name, tracer) -> dict
    Orchestrator that calls the appropriate source function, wraps the
    call in exponential-backoff retries, falls back to the alternative
    source automatically if the primary fails, and returns metadata
    about which source answered and whether a fallback occurred.

All source functions return a uniform dict with keys:
  source, success, data, error
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

from dotenv import load_dotenv
from rich.console import Console

from utils.retry import retry_with_backoff

load_dotenv(override=True)

console = Console()

# Sentinel prefix used by retry_with_backoff on total exhaustion
_RETRY_ERROR_PREFIX = "The operation failed after"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    """Replace characters that can't be encoded on the current terminal."""
    encoding = sys.stdout.encoding or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding)


# ---------------------------------------------------------------------------
# web_search — Tavily Search API
# ---------------------------------------------------------------------------

def _web_search_raw(query: str) -> str:
    """Core web-search via Tavily.  Raises on network / API errors.

    Returns formatted result text on success (which may say "No results
    found"—that is a valid response, not an error).
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY environment variable is not set.")

    from tavily import TavilyClient

    client = TavilyClient(api_key=api_key)
    response = client.search(query=query, search_depth="basic")

    results = response.get("results", [])
    if not results:
        return f"No web results found for: {query}"

    parts = [f"Search results for: {query}\n"]
    for i, r in enumerate(results, 1):
        title = r.get("title", "No title")
        url = r.get("url", "")
        content = r.get("content", "No content")
        parts.append(f"{i}. {title}")
        parts.append(f"   URL: {url}")
        parts.append(f"   {content}\n")

    return _clean("\n".join(parts).strip())


def web_search(query: str) -> dict:
    """Search the web using Tavily.

    Args:
        query: The search query string.

    Returns:
        A dict with ``source``, ``success``, ``data``, and ``error`` keys.
    """
    try:
        data = _web_search_raw(query)
        return {"source": "web_search", "success": True, "data": data, "error": None}
    except Exception as e:
        return {"source": "web_search", "success": False, "data": "", "error": str(e)}


# ---------------------------------------------------------------------------
# sec_edgar_lookup — SEC EDGAR full-text search
# ---------------------------------------------------------------------------

_SEC_BASE = "https://efts.sec.gov/LATEST/search-index"
_SEC_UA = "FundDiligenceAgent/1.0 (research-internal@example.com)"


def _sec_edgar_raw(company_name: str) -> str:
    """Search SEC EDGAR for recent 10-K filings mentioning *company_name*.

    Uses the SEC's public full-text search API (no API key needed).

    Args:
        company_name: Name to search for (e.g. "Sequoia Capital").

    Returns:
        Formatted text describing matching filings, or a message
        indicating nothing was found.

    Raises:
        RuntimeError: On network / HTTP / parse errors (triggers retry).
    """
    import urllib.request
    import urllib.parse

    quoted = urllib.parse.quote(f'"{company_name}"')
    url = f"{_SEC_BASE}?q={quoted}&forms=10-K&dateRange=3y"

    req = urllib.request.Request(url, headers={"User-Agent": _SEC_UA})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
            data = json.loads(raw)
    except Exception as e:
        raise RuntimeError(f"SEC EDGAR request failed: {e}") from e

    hits = data.get("hits", {}).get("hits", [])

    if not hits:
        return (
            f'No 10-K filings found mentioning "{company_name}" '
            f"in the last 3 years."
        )

    parts = [
        f'SEC EDGAR — 10-K filings mentioning "{company_name}" '
        f"(showing top {min(len(hits), 10)} of {len(hits)}):\n"
    ]
    for i, hit in enumerate(hits[:10], 1):
        src = hit.get("_source", {})
        display_name = (src.get("display_names") or ["Unknown"])[0]
        cik = (src.get("ciks") or [""])[0]
        filed = src.get("file_date", "Unknown date")
        form = src.get("form", "10-K")
        period = src.get("period_ending", "")
        doc_type = src.get("file_type", "")
        description = src.get("file_description", "")

        # Build a direct link to the filing
        cik_no_lead = cik.lstrip("0")
        adsh_dashed = src.get("adsh", "")
        adsh_flat = adsh_dashed.replace("-", "")
        if cik_no_lead and adsh_flat:
            filing_url = (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{cik_no_lead}/{adsh_flat}/{adsh_dashed}-index.html"
            )
        else:
            filing_url = (
                f"https://www.sec.gov/cgi-bin/browse-edgar?"
                f"action=getcompany&CIK={cik}"
            )

        parts.append(f"{i}. {display_name} — {form} ({doc_type})")
        parts.append(f"   Filed: {filed}   Period: {period}")
        parts.append(f"   CIK: {cik}")
        parts.append(f"   {filing_url}")
        if doc_type == "10-K" or "10-K" in description:
            parts.append("   ⬆ Main filing document")

        parts.append("")

    return "\n".join(parts).strip()


def sec_edgar_lookup(company_name: str) -> dict:
    """Search SEC EDGAR for 10-K filings mentioning *company_name*.

    Args:
        company_name: Name to search for.

    Returns:
        A dict with ``source``, ``success``, ``data``, and ``error`` keys.
    """
    try:
        data = _sec_edgar_raw(company_name)
        return {
            "source": "sec_edgar_lookup",
            "success": True,
            "data": data,
            "error": None,
        }
    except Exception as e:
        return {
            "source": "sec_edgar_lookup",
            "success": False,
            "data": "",
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# retrieve — orchestrator with retry + auto-fallback
# ---------------------------------------------------------------------------

_RAW_SOURCES = {
    "web_search": _web_search_raw,
    "sec_edgar_lookup": _sec_edgar_raw,
}

_PUBLIC_SOURCES = {
    "web_search": web_search,
    "sec_edgar_lookup": sec_edgar_lookup,
}

_FALLBACK_NAME = {
    "web_search": "sec_edgar_lookup",
    "sec_edgar_lookup": "web_search",
}


def retrieve(step: dict, tool_name: str, tracer=None) -> dict:
    """Execute a research step by calling the appropriate source function.

    The primary source is called with exponential-backoff retries.
    If it fails entirely (all retries exhausted), the **other** source
    is tried once as an automatic fallback.  If the fallback also fails,
    the returned dict reflects that.

    Args:
        step: A plan step dict — ``step["action"]`` is used as the query.
        tool_name: ``"web_search"`` or ``"sec_edgar_lookup"``.
        tracer: Optional ``Tracer`` instance for logging.

    Returns:
        A dict with keys:
          - source        — name of the source that answered (or attempted)
          - success       — bool
          - data          — result text (empty string on failure)
          - error         — error message, or ``None`` on success
          - used_fallback — ``True`` if the fallback source was activated
          - attempts      — number of times the *primary* source was tried
    """
    t0 = time.time()
    query = step.get("action", "")

    raw_fn = _RAW_SOURCES.get(tool_name)
    if not raw_fn:
        return {
            "source": tool_name,
            "success": False,
            "data": "",
            "error": f"Unknown tool: {tool_name}",
            "used_fallback": False,
            "attempts": 0,
        }

    # --- Primary source with retry -----------------------------------------

    attempts: list[int] = [0]

    def _try():
        attempts[0] += 1
        return raw_fn(query)

    console.print(f"  [dim]→ Retrieving via [bold]{tool_name}[/] ...[/]")

    result = retry_with_backoff(_try, tracer=tracer)

    # Did retry exhaust? (retry_with_backoff returns an error string on total
    # failure; the raw functions return formatted text on success.)
    if isinstance(result, str) and result.startswith(_RETRY_ERROR_PREFIX):
        # ---- Primary failed — try the other source as fallback ----------
        fallback_name = _FALLBACK_NAME[tool_name]
        console.print(
            f"  [yellow]⚠ {tool_name} failed after {attempts[0]} attempt(s), "
            f"falling back to [bold]{fallback_name}[/] ...[/]"
        )

        if tracer:
            tracer.log_tool_call(
                tool=tool_name,
                args={"query": query},
                result_preview=f"FAILED after {attempts[0]} attempt(s): {result[:200]}",
                duration_sec=round(time.time() - t0, 3),
            )

        fb_fn = _PUBLIC_SOURCES[fallback_name]
        fb_t0 = time.time()
        fb_result = fb_fn(query)
        fb_dur = time.time() - fb_t0

        if tracer:
            tracer.log_tool_call(
                tool=fallback_name,
                args={"query": query},
                result_preview=(fb_result.get("data") or fb_result.get("error") or "")[:500],
                duration_sec=round(fb_dur, 3),
            )

        return {
            **fb_result,
            "used_fallback": True,
            "attempts": attempts[0],
        }

    # ---- Primary succeeded -----------------------------------------------
    dur = time.time() - t0
    if tracer:
        tracer.log_tool_call(
            tool=tool_name,
            args={"query": query},
            result_preview=result[:500],
            duration_sec=round(dur, 3),
        )

    return {
        "source": tool_name,
        "success": True,
        "data": result,
        "error": None,
        "used_fallback": False,
        "attempts": attempts[0],
    }
