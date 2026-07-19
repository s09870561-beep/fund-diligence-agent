"""Memory layer — persistent storage for research findings.

Uses ChromaDB with sentence-transformers (all-MiniLM-L6-v2) for real
semantic embeddings.  The model downloads once from Hugging Face then
runs fully offline — no API keys, no paid services.

  - get_collection() -> persistent ChromaDB collection stored on disk
  - save_finding(entity_name, brief) -> stores a DiligenceBrief as a vector
  - recall_memory(query, top_k) -> semantic search, returns uniform dict
"""

import json
import os
import time
from datetime import datetime, timezone

from dotenv import load_dotenv
from rich.console import Console

from presentation import DiligenceBrief

load_dotenv(override=True)

console = Console()

MEMORY_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(MEMORY_DIR, "chroma_db")

_client = None
_collection = None
_embedding_fn = None

# ---------------------------------------------------------------------------
# ChromaDB collection  (with sentence-transformers embedding function)
# ---------------------------------------------------------------------------


def _get_embedding_function():
    """Lazy-init the sentence-transformers embedding function (all-MiniLM-L6-v2).

    The model is ~80 MB and downloads once from Hugging Face on first use.
    After that it is cached and runs fully offline.
    """
    global _embedding_fn
    if _embedding_fn is not None:
        return _embedding_fn

    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    console.print("[dim]Loading sentence-transformer model (all-MiniLM-L6-v2) …[/]")
    t0 = time.time()
    _embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    dur = time.time() - t0
    console.print(f"[dim]Model loaded in {dur:.1f}s[/]")
    return _embedding_fn


def _get_client():
    """Lazy-init the ChromaDB persistent client."""
    import chromadb
    from chromadb.config import Settings

    os.makedirs(CHROMA_DIR, exist_ok=True)
    return chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False),
    )


def get_collection():
    """Return the singleton 'findings' collection, creating it if needed.

    The collection uses ``SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")``
    so documents and queries are automatically embedded via a real transformer
    model — no manual embedding code needed.
    """
    global _client, _collection
    if _collection is not None:
        return _collection
    _client = _get_client()
    ef = _get_embedding_function()
    _collection = _client.get_or_create_collection(
        name="findings",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    return _collection


# ---------------------------------------------------------------------------
# save_finding
# ---------------------------------------------------------------------------


def _build_embedding_text(brief: DiligenceBrief) -> str:
    """Concatenate relevant brief fields into a single searchable text."""
    parts = [brief.overview]
    if brief.recent_activity:
        parts.append("Recent news: " + " | ".join(brief.recent_activity))
    if brief.past_deals:
        parts.append("Notable deals: " + " | ".join(brief.past_deals))
    if brief.leadership:
        names = [m.name for m in brief.leadership]
        parts.append("Leadership: " + ", ".join(names))
    return "\n".join(parts)


def save_finding(entity_name: str, brief: DiligenceBrief) -> dict:
    """Store a DiligenceBrief into the vector memory.

    The embedding is computed automatically by the collection's sentence-
    transformer embedding function — no manual vector computation needed.

    Args:
        entity_name: The fund/company name (used as metadata tag).
        brief: The DiligenceBrief to store.

    Returns:
        A dict with ``id``, ``entity_name``, ``timestamp``.
    """
    collection = get_collection()
    text = _build_embedding_text(brief)

    timestamp = datetime.now(timezone.utc).isoformat()
    doc_id = f"{entity_name}__{int(time.time())}"

    metadata = {
        "entity_name": entity_name,
        "timestamp": timestamp,
        "overview": brief.overview[:1000],
        "recent_activity": json.dumps(brief.recent_activity, ensure_ascii=False)[:1000],
        "past_deals": json.dumps(brief.past_deals, ensure_ascii=False)[:1000],
        "leadership": json.dumps(
            [m.model_dump() for m in brief.leadership], ensure_ascii=False
        )[:1000],
        "sources_used": json.dumps(brief.sources_used, ensure_ascii=False),
    }

    collection.add(
        documents=[text],
        metadatas=[metadata],
        ids=[doc_id],
    )

    return {
        "id": doc_id,
        "entity_name": entity_name,
        "timestamp": timestamp,
    }


# ---------------------------------------------------------------------------
# recall_memory
# ---------------------------------------------------------------------------


def recall_memory(query: str, top_k: int = 3) -> dict:
    """Search past findings via semantic (vector) similarity.

    Uses the all-MiniLM-L6-v2 sentence-transformer model for real semantic
    embeddings — captures meaning, not just keyword overlap.

    Args:
        query: Free-text search query — does NOT need to match exact keywords.
        top_k: Max results to return (default 3).

    Returns:
        Uniform result dict with keys:
          source, success, data, error
    """
    t0 = time.time()

    try:
        collection = get_collection()
    except Exception as e:
        return {
            "source": "recall_memory",
            "success": False,
            "data": "",
            "error": f"Failed to open ChromaDB collection: {e}",
        }

    try:
        count = collection.count()
    except Exception:
        count = 0

    if count == 0:
        return {
            "source": "recall_memory",
            "success": True,
            "data": "No findings have been saved to memory yet.",
            "error": None,
        }

    try:
        results = collection.query(
            query_texts=[query],
            n_results=min(top_k, count),
        )
    except Exception as e:
        return {
            "source": "recall_memory",
            "success": False,
            "data": "",
            "error": f"ChromaDB query failed: {e}",
        }

    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not ids:
        return {
            "source": "recall_memory",
            "success": True,
            "data": "No relevant findings found for the query.",
            "error": None,
        }

    lines = [f"Memory search results for: \"{query}\"\n"]
    for i, (doc_id, doc_text, dist, meta) in enumerate(
        zip(ids, documents, distances, metadatas), 1
    ):
        entity = (meta or {}).get("entity_name", "Unknown")
        ts = (meta or {}).get("timestamp", "?")
        score = round(1.0 - (dist or 0), 4)

        lines.append(f"{i}. (entity: {entity})  (similarity: {score})")
        lines.append(f"   Saved: {ts}")
        lines.append(f"   {doc_text[:400]}...")
        lines.append("")

    return {
        "source": "recall_memory",
        "success": True,
        "data": "\n".join(lines).strip(),
        "error": None,
    }
