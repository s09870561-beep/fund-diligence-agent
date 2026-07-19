# Fund Diligence Agent

An AI-powered research assistant for **family offices, institutional LPs, and investment committees** — producing structured **Investment Committee Memos** with IC-confirmed findings, flagged risks, mandate fit scoring, and conflict-of-interest checks.

## What it does

Give it a research goal like _"Research Sequoia Capital: recent activity, leadership, and past deals"_ and the agent:

1. **Plans** — breaks your goal into concrete research steps (news search, leadership lookup, deal analysis, SEC filings check)
2. **Gathers** — searches the web via Tavily and checks SEC EDGAR for recent 10-K filings
3. **Synthesises** — feeds everything to an LLM that produces a structured `DiligenceBrief`
4. **Flags uncertainty** — the model is *instructed* to put anything unclear into `open_questions` rather than guess
5. **Human review** — lets you confirm, reject, or edit each uncertain claim (IC Review workflow)
6. **IC Memo** ��� one-page Investment Committee Memo with Executive Summary, Thesis, Risk Factors, Mandate Fit, and Recommendation
7. **Conflict-of-Interest check** — scans portfolio holdings for potential conflicts
8. **Remembers** — saves finished briefs to a local vector database (ChromaDB with all-MiniLM-L6-v2 embeddings) for natural-language recall

## Architecture (7 layers)

| Layer | Responsibility |
|-------|---------------|
| `reasoning/` | Planning (`create_plan`), tool selection, LLM orchestration |
| `retrieval/` | Data gathering: web search (Tavily) and SEC EDGAR full-text lookup |
| `tools/` | Unified `TOOL_REGISTRY` + `execute_tool()` dispatcher with OpenAI function-calling schemas |
| `memory/` | Vector memory via ChromaDB + sentence-transformers (all-MiniLM-L6-v2) for semantic recall |
| `presentation/` | `DiligenceBrief` Pydantic model, LLM-based synthesis, `format_brief()` rendering, and `format_ic_memo()` for one-page Investment Committee Memos |
| `matching/` | Investment mandate scoring (`match_mandate`) with field-level reasoning against the brief |
| `relationships/` | Entity connection discovery (`find_connections`) and portfolio conflict-of-interest checks (`check_conflicts`) |
| `guardrails/` | `RunGuard` (step/time/cost ceilings with `RunLimitExceeded`), loop detection, `review_open_questions()` |
| `utils/` | `retry_with_backoff()` (exponential-backoff), `Tracer` (JSONL run logging) |

Each layer imports only from the layers below it — no circular dependencies.

## Setup

```bash
git clone <repo> fund-diligence-agent
cd fund-diligence-agent

# Create a virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

You'll need a `.env` file in the project root:

```env
# OpenCode Zen API key (free, no billing needed)
OPENCODE_ZEN_API_KEY=sk-...

# Tavily Search API key — get one free at https://tavily.com/
TAVILY_API_KEY=tvly-...
```

Both keys are free. The sentence‑transformer model (`all-MiniLM-L6-v2`) downloads once from Hugging Face (~80 MB) and then runs fully offline.

## How to run

```bash
# Interactive mode — runs the full pipeline with human review
python main.py "Research Sequoia Capital: recent activity, leadership, and past deals."

# Non-interactive mode — auto-confirms all open questions
set AUTO_APPROVE_REVIEW=y
python main.py "Research Stripe: business model, growth, and competitors."

# Env-var ceilings (override defaults: steps=6, time=120s, cost=$1.00)
set MAX_STEPS=10
set MAX_TIME=300
python main.py "Research Y Combinator: structure, track record, and influence."
```

## View a trace

Every run logs structured events to `logs/run_<timestamp>.jsonl`. View the latest trace:

```bash
python view_trace.py                    # latest run
python view_trace.py logs/run_2026.jsonl  # specific file
```

This prints a color-coded timeline showing every goal, plan step, tool call (with duration and result preview), LLM usage, human review decision, and the final RunGuard summary.

## Run evals

```bash
python evals/run_evals.py
```

This runs 4 test cases against the full pipeline, auto-confirms open questions, and scores each result via an LLM judge against criteria in `evals/test_cases.json`. Output is a pass/fail table and an overall score.

---

## Design decisions

### 1. Explicit tool-selection logic before LLM fallback

The `select_tool()` function in `reasoning/` uses a **three-path** strategy:

- **Path A** — the plan step's `tool_hint` is checked against an explicit map (`web_search` → `web_search`, `sec_edgar_lookup` → `sec_edgar_lookup`, `none` → `none`). This handles ~90 % of cases in one dictionary lookup.
- **Path B** — only when the hint is missing, empty, or ambiguous (`"search"`, `"lookup"`, `"research"`, etc.) does the function fall back to an LLM call.
- **Path C** — if the hint string contains an available tool name as a substring, it's matched directly.

**Why not always ask the LLM?** An LLM call for every tool selection would add latency and cost for a task that is almost always a direct lookup from the plan. The explicit map resolves instantly. The LLM fallback is there for the rare ambiguous case — not as the primary path. This also means the agent works (most of the time) even if the LLM is unavailable or slow.

### 2. Fact-level human checkpoints instead of only action-approval

The agent has two kinds of checkpoints:

1. **`review_open_questions()`** — after the brief is synthesized, every uncertainty the model flagged is presented to the user for **confirm / reject / edit**. This is a fact-level review, not an action-level one.
2. **`RunGuard` ceilings** — automatic hard stops on step count, wall-clock time, and simulated cost.

**Why fact-level review instead of action-approval?** Action-level approval ("do you want to call web_search?") would be unbearably chatty for a pipeline that makes 4–6 tool calls. More importantly, the human's expertise is in evaluating *claims*, not approving tool invocations. The model is good at *finding* information; the human is good at *judging* whether the resulting claims are reliable. By surfacing only the uncertainties, the review step stays small (usually 3–7 items) and each one is a substantive judgment call.

### 3. Real sentence-transformer embeddings (replacing n-gram hashing)

The memory layer initially used a **hand-rolled character n-gram hashing** approach: extract character 2/3/4-grams from the text, hash each to a position in a 768‑dimensional vector with MD5, and L2-normalize. This ran fully locally, required zero downloads, and took ~10 lines of code.

**Why replace it?** The n-gram approach was fast and zero-dependency, but it could only match *substring-level* patterns. A query like _"PE firm co-steward transition"_ would find a document about Sequoia Capital only through literal n-gram overlap between "transition" and "co-steward" — it had no understanding that "VC firm" and "venture capital" are semantically equivalent.

Replacing it with `all-MiniLM-L6-v2` via sentence-transformers improved the recall query from a similarity of ~0.48 (n-gram) to ~0.49 (transformer) for the reworded query, and from failing entirely to ~0.33 for the much harder _"PE firm co-steward transition"_ query. The transformer model correctly maps near-synonyms to nearby positions in embedding space because it was trained on millions of sentence pairs.

**Why not use it from the start?** The n-gram approach was built first to prove the architecture without adding PyTorch (a ~2 GB dependency). Once the memory layer was validated and working, replacing the embedding function was a 10‑minute change. The story is: *start simple, iterate when the metrics tell you to.*

### 4. Hard ceilings with real cost tracking instead of just prompting

The `RunGuard` class tracks **step_count**, **tool_call_count**, **total_tokens**, **estimated_cost**, and **elapsed_time** for every run, and raises `RunLimitExceeded` when any ceiling is exceeded.

**Why hard ceilings?** Asking the LLM to "be efficient" is a soft constraint with no guarantees. A model in a loop can easily call the same tool 50 times before it decides to stop — the prompt said "be efficient" but the model interpreted that differently. Hard ceilings catch this immediately: `detect_loop()` flags the 3rd repeated call, `check_step_ceiling()` stops the run at the configured limit.

**Why simulated cost tracking?** The underlying model (OpenCode Zen / deepseek-v4-flash-free) is free, so there's nothing to track. But the architecture treats cost as a first-class constraint because real deployments would use paid models. The simulated per-token constants (`$1/M input`, `$2/M output`) are close to real GPT-4o pricing, so switching to a paid model requires only changing the constants, not the architecture.

---

## Known limitations

- **LLM output variability** — the plan and brief quality depend on the model, which is non-deterministic. Evals show 60–80 % pass rates partly due to this variation.
- **Tavily free tier** — the free Tavily plan limits search depth and rate. For production use, upgrade to a paid Tavily plan or swap in a different search provider.
- **SEC EDGAR parsing** — the current `sec_edgar_lookup` returns filing *metadata* (CIK, date, form type, direct link) but does not fetch or parse the filing *content*. For deeper analysis, the extraction step needs a PDF/HTML parser.
- **Single-entity focus** — the pipeline researches one entity per run. Cross-fund comparisons or portfolio-wide analysis would require running multiple pipelines and merging results.
- **Memory deduplication** — `save_finding()` stores every brief independently. There is no deduplication by entity name — if you research Sequoia Capital twice, both briefs are stored as separate documents.
- **No tool retry for non-LLM tools** — `execute_tool()` wraps its call in a try/except but uses the `retry_with_backoff` only inside `retrieval/` for the raw source calls. Network errors during tool dispatch are caught and returned as error dicts, not retried automatically.
- **Windows symlinks** — Hugging Face's cache system uses symlinks, which require either Developer Mode or administrator privileges on Windows. The model still works without symlinks (the library prints a non-fatal warning and falls back to copying).
