"""Relationships layer — entity connection discovery.

This module searches for connections between two entities (companies, funds,
investors) and returns structured findings with sources.

  - find_connections(entity_a, entity_b) -> dict
    Runs targeted web searches for shared investors, board members,
    co-investments, and partnerships, then sends results to the LLM
    to extract only explicitly-sourced connections.
"""

import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich import box as rich_box
from rich.table import Table
from rich.panel import Panel

from retrieval import web_search
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
# Search queries
# ---------------------------------------------------------------------------

_CONNECTION_QUERIES = [
    "{a} {b} investor overlap",
    "{a} {b} board member",
    "{a} {b} co-invested",
    "{a} {b} funding deal partnership",
]


def _run_searches(entity_a: str, entity_b: str) -> list[dict]:
    """Run targeted web searches for connections between the two entities.

    Each search is wrapped in retry_with_backoff. Returns a list of
    result dicts with query, success, and data fields.
    """
    results = []
    for template in _CONNECTION_QUERIES:
        query = template.format(a=entity_a, b=entity_b)
        console.print(f"  [dim]Searching:[/] {query}")
        result = retry_with_backoff(
            lambda q=query: web_search(q),
        )
        if isinstance(result, str):
            # Web search returned an error string (retry exhausted)
            results.append({"query": query, "success": False, "data": result})
        else:
            # web_search returns a dict with source/success/data/error
            success = result.get("success", False)
            data = result.get("data", "")
            results.append({"query": query, "success": success, "data": data})
    return results


# ---------------------------------------------------------------------------
# LLM analysis prompt
# ---------------------------------------------------------------------------

_ANALYSIS_PROMPT = """You are a due-diligence analyst identifying connections between two entities.

You will be given:
  1. The names of two entities (Entity A and Entity B).
  2. Raw web search results for queries looking for connections between them.

Your task is to identify ONLY connections that are EXPLICITLY supported by the search results. Do NOT infer, guess, or use your own knowledge — if the search results don't mention a connection, it doesn't exist for the purposes of this analysis.

Types of connections to look for:
  - shared_investor: An investor that has invested in both entities.
  - shared_board: An individual who serves on both entities' boards.
  - co_investment: The two entities invested in the same company together.
  - partnership: A business partnership, integration, or collaboration.
  - funding: Entity A invested in Entity B, or vice versa.
  - other: Any other explicit connection mentioned in the search results.

For each connection you find, include:
  - type: one of the types above.
  - description: A 1-2 sentence description of the connection.
  - source: The specific URL or article title from the search results that supports this connection.

IMPORTANT — This is a hard rule:
  - If NO connections are found in the search results, set connections to an empty list and confidence to "low".
  - In searched_but_not_found, list each type of connection you looked for and didn't find (e.g. "shared board membership between Sequoia Capital and Stripe").

Respond ONLY with a single JSON object — no markdown fences, no explanation:

{
  "connections": [
    {
      "type": "shared_investor",
      "description": "Entity A and Entity B both received funding from ...",
      "source": "https://example.com/article-url"
    }
  ],
  "confidence": "high|medium|low",
  "searched_but_not_found": [
    "shared board membership between Entity A and Entity B",
    "co-investment deals between Entity A and Entity B in 2025"
  ]
}"""


# ---------------------------------------------------------------------------
# find_connections
# ---------------------------------------------------------------------------


def find_connections(entity_a: str, entity_b: str) -> dict:
    """Search for connections between two entities and return structured findings.

    Args:
        entity_a: First entity name (e.g. "Sequoia Capital").
        entity_b: Second entity name (e.g. "Stripe").

    Returns:
        A dict with keys:
          - connections       list[dict] — each with ``type``, ``description``,
                              and ``source``.
          - confidence        str — ``"high"``, ``"medium"``, or ``"low"``.
          - searched_but_not_found
                              list[str] — what was looked for and not found.
    """
    console.print(f"[bold]Finding connections between[/] [cyan]{entity_a}[/] and [cyan]{entity_b}[/] ...")
    t0 = time.time()

    # -- Step 1: Run searches ----------------------------------------------
    search_results = _run_searches(entity_a, entity_b)

    # -- Step 2: Build evidence text for the LLM ---------------------------
    evidence_parts = []
    for sr in search_results:
        query = sr["query"]
        success = sr["success"]
        data = sr.get("data", "")
        if success and data:
            evidence_parts.append(f"--- Search: {query} ---\n{data[:2000]}\n")
        else:
            evidence_parts.append(f"--- Search: {query} ---\n(no results or error)\n")

    evidence_text = "\n".join(evidence_parts)

    # -- Step 3: Send to LLM for analysis ----------------------------------
    user_content = (
        f"Entity A: {entity_a}\n"
        f"Entity B: {entity_b}\n\n"
        f"Search results:\n{evidence_text}\n\n"
        "Identify any connections between these two entities based ONLY on the search results above."
    )

    response = retry_with_backoff(
        lambda: _client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": _ANALYSIS_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
        ),
    )

    dur = time.time() - t0

    if isinstance(response, str):
        console.print(f"[red]Analysis LLM call failed: {response}[/]")
        return {
            "connections": [],
            "confidence": "low",
            "searched_but_not_found": [f"Analysis failed: {response}"],
        }

    content = response.choices[0].message.content or ""
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1]
        content = content.rsplit("```", 1)[0].strip()

    try:
        result = json.loads(content)
        connections = result.get("connections", [])
        confidence = result.get("confidence", "low")
        not_found = result.get("searched_but_not_found", [])

        console.print(f"  [dim]Found {len(connections)} connection(s), confidence={confidence} ({dur:.1f}s)[/]")
        return {
            "connections": connections,
            "confidence": confidence,
            "searched_but_not_found": not_found,
        }
    except (json.JSONDecodeError, ValueError) as e:
        console.print(f"[red]Failed to parse analysis: {e}[/]")
        return {
            "connections": [],
            "confidence": "low",
            "searched_but_not_found": [f"Parse error: {e}"],
        }


# ---------------------------------------------------------------------------
# check_conflicts  —  conflict-of-interest check against a portfolio
# ---------------------------------------------------------------------------


def check_conflicts(entity: str, portfolio_entities: list[str]) -> dict:
    """Check a researched entity for conflicts against an existing portfolio.

    Runs ``find_connections()`` between *entity* and each entry in
    *portfolio_entities*, then consolidates the results into a single
    report.  Any connection found is flagged as a potential conflict.

    Args:
        entity: The entity being researched (e.g. "Sequoia Capital").
        portfolio_entities: A list of existing portfolio entities to
            check against.  May be empty.

    Returns:
        A dict with keys:
          - entity              str — the entity that was checked.
          - portfolio_checked   list[str] — the portfolio list provided.
          - conflicts_found     list[dict] ��� each entry has the structure
            from ``find_connections()`` plus a ``portfolio_entity`` field
            indicating which portfolio item the conflict relates to.
          - no_conflicts        list[str] — portfolio entities with no
            connection found.
          - overall_confidence  str — the lowest confidence across all
            individual checks (``"high"``, ``"medium"``, or ``"low"``).
          - message             str — a human-readable summary.
    """
    if not portfolio_entities:
        return {
            "entity": entity,
            "portfolio_checked": [],
            "conflicts_found": [],
            "no_conflicts": [],
            "overall_confidence": "low",
            "message": (
                "No portfolio was provided to check against. "
                "Enter one or more portfolio company / fund names "
                "to run a conflict-of-interest check."
            ),
        }

    console.print(
        f"[bold]Checking conflicts[/] [cyan]{entity}[/] against "
        f"[cyan]{len(portfolio_entities)}[/] portfolio entit(ies) ..."
    )
    t0 = time.time()

    conflicts_found: list[dict] = []
    no_conflicts: list[str] = []
    confidences: list[str] = []

    for i, portfolio_entity in enumerate(portfolio_entities, 1):
        console.print(
            f"  [{i}/{len(portfolio_entities)}] Checking "
            f"[dim]{portfolio_entity}[/] ..."
        )
        result = find_connections(entity, portfolio_entity)

        conns = result.get("connections", [])
        conf = result.get("confidence", "low")
        confidences.append(conf)

        if conns:
            for conn in conns:
                conflicts_found.append({
                    **conn,
                    "portfolio_entity": portfolio_entity,
                })
        else:
            no_conflicts.append(portfolio_entity)

    # Lowest confidence across all checks
    rank = {"high": 3, "medium": 2, "low": 1}
    overall_confidence = min(confidences, key=lambda c: rank.get(c, 0)) if confidences else "low"

    dur = time.time() - t0

    if conflicts_found:
        message = (
            f"Found {len(conflicts_found)} potential conflict(s) between "
            f"{entity} and the provided portfolio "
            f"({len(no_conflicts)} entit(ies) with no conflicts). "
            f"Overall confidence: {overall_confidence}. "
            f"({dur:.1f}s)"
        )
    else:
        message = (
            f"No conflicts found between {entity} and "
            f"the {len(portfolio_entities)} portfolio entit(ies) checked. "
            f"Overall confidence: {overall_confidence}. "
            f"({dur:.1f}s)"
        )

    console.print(f"  [dim]{message}[/]")

    return {
        "entity": entity,
        "portfolio_checked": list(portfolio_entities),
        "conflicts_found": conflicts_found,
        "no_conflicts": no_conflicts,
        "overall_confidence": overall_confidence,
        "message": message,
    }


def print_connections(result: dict) -> None:
    """Pretty-print the connections result using rich."""
    connections = result.get("connections", [])
    confidence = result.get("confidence", "low")
    not_found = result.get("searched_but_not_found", [])

    conf_styles = {"high": "[green]high[/]", "medium": "[yellow]medium[/]", "low": "[red]low[/]"}
    conf_style = conf_styles.get(confidence, "[dim]unknown[/]")

    if connections:
        table = Table(
            title=f"[bold]Connections found (confidence: {conf_style})[/]",
            border_style="cyan",
            box=rich_box.ROUNDED,
        )
        table.add_column("#", style="dim", width=3)
        table.add_column("Type", style="bold cyan")
        table.add_column("Description", style="white", width=60)
        table.add_column("Source", style="dim", width=40)

        for i, conn in enumerate(connections, 1):
            table.add_row(
                str(i),
                conn.get("type", "?"),
                conn.get("description", ""),
                conn.get("source", "")[:38],
            )
        console.print(table)
    else:
        console.print(Panel(f"[yellow]No connections found (confidence: {conf_style})[/]", border_style="yellow"))

    if not_found:
        console.print()
        console.print("[bold dim]Also checked but didn't find:[/]")
        for item in not_found:
            console.print(f"  [dim]• {item}[/]")
