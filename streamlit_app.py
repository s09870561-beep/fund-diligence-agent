"""Streamlit web app wrapping the fund-diligence-agent pipeline.

Usage:
    streamlit run streamlit_app.py

Tabs: Diligence Brief (default), Mandate Match, Find Connections.
"""

import json
import os
import re
import time

import streamlit as st

# ---------------------------------------------------------------------------
# API keys: st.secrets first, then os.environ / .env
# ---------------------------------------------------------------------------
for _fkey in ("OPENCODE_ZEN_API_KEY", "TAVILY_API_KEY"):
    try:
        if _fkey in st.secrets:
            os.environ[_fkey] = st.secrets[_fkey]
    except Exception:
        pass

from dotenv import load_dotenv

load_dotenv(override=True)

from reasoning import create_plan
from tools import execute_tool
from presentation import synthesize_brief, DiligenceBrief, format_ic_memo
from guardrails import RunLimitExceeded
from utils.tracer import Tracer

from matching import InvestmentMandate, match_mandate
from relationships import find_connections, check_conflicts

# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------

_INITIAL = {
    "phase": "input",
    "goal": "",
    "plan": None,
    "gathered": None,
    "brief": None,
    "final_brief": None,
    "trace_events": [],
    "tracer": None,
    "error": None,
    "entity": "",
    "review_questions": [],
    "review_idx": 0,
    "review_decisions": [],
    "review_editing": False,
    "review_edit_value": "",
    "step_count": 0,
    "tool_call_count": 0,
    "started_at": None,
    # Mandate match state
    "mandate_result": None,
    "mandate_running": False,
    # Connections state
    "conn_result": None,
    "conn_running": False,
    # IC Memo state
    "ic_memo": None,
    # Conflict check state
    "conflict_result": None,
    "conflict_running": False,
}

for _fk, _fv in _INITIAL.items():
    if _fk not in st.session_state:
        st.session_state[_fk] = _fv


def _reset():
    for _fk in _INITIAL:
        st.session_state[_fk] = _INITIAL[_fk]


def _add_trace(evt: str, label: str, data, duration: float | None = None):
    st.session_state.trace_events.append({
        "event": evt,
        "label": label,
        "data": data,
        "duration": duration,
        "time": time.strftime("%H:%M:%S"),
    })


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Fund Diligence Agent", page_icon=":bar_chart:", layout="wide")

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.agent-header {
    padding: 2rem 0 0 0;
}
.agent-header h1 {
    font-size: 2.25rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #F1F5F9;
    margin: 0 0 0.25rem 0;
}
.agent-header .accent {
    color: #2DD4BF;
}
.agent-header .tagline {
    font-size: 1rem;
    font-weight: 400;
    color: #8899AA;
    margin: 0 0 1.5rem 0;
}

.result-card {
    background: #1A1E23;
    border: 1px solid #2A2F37;
    border-radius: 12px;
    padding: 1.75rem 2rem;
    margin: 1rem 0 1.5rem 0;
    box-shadow: 0 4px 20px rgba(0,0,0,0.25);
}
.result-card h3 {
    font-size: 1.1rem;
    font-weight: 600;
    color: #2DD4BF;
    margin: 0 0 0.75rem 0;
}
.result-card p, .result-card div, .result-card li {
    font-size: 0.95rem;
    line-height: 1.7;
    color: #D1D8E0;
}

.step-label {
    font-size: 0.85rem;
    font-weight: 500;
    color: #8899AA;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 0.15rem;
}

.stButton > button {
    border-radius: 8px;
    font-weight: 500;
    transition: all 0.15s ease;
}
.stButton > button:active {
    transform: scale(0.97);
}

.streamlit-expanderHeader {
    font-weight: 600;
    font-size: 0.9rem;
    color: #D1D8E0;
}

.stAlert {
    border-radius: 8px;
    border-left-width: 4px;
}

div[data-testid="stMetricValue"] {
    font-weight: 700;
    color: #2DD4BF;
}
div[data-testid="stMetricLabel"] {
    font-weight: 500;
    color: #8899AA;
}
"""
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
st.markdown(
    '<div class="agent-header"><h1><span class="accent">Fund Diligence</span> Agent</h1>'
    '<p class="tagline">AI-powered research for family offices, institutional LPs, and investment committees — '
    'producing structured IC memos with verified claims and flagged uncertainties.</p></div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Controls")
    if st.button("New research", use_container_width=True, type="primary"):
        _reset()
        st.rerun()
    st.divider()
    st.markdown("### API Keys")
    ok_z = os.getenv("OPENCODE_ZEN_API_KEY") and os.getenv("OPENCODE_ZEN_API_KEY") != "sk-your-key-here"
    ok_t = os.getenv("TAVILY_API_KEY") and os.getenv("TAVILY_API_KEY") != "tvly-your-key-here"
    if ok_z:
        st.success("OpenCode Zen key found")
    else:
        st.error("OpenCode Zen key missing")
    if ok_t:
        st.success("Tavily key found")
    else:
        st.error("Tavily key missing")
    if st.session_state.brief:
        st.divider()
        st.caption(f"Latest memo: {st.session_state.brief.entity_name}")


# ---------------------------------------------------------------------------
# Phase functions (unchanged)
# ---------------------------------------------------------------------------


def _extract_entity_name(goal: str) -> str:
    for pfx in ("Research ", "Evaluate ", "Analyze "):
        if goal.startswith(pfx):
            rest = goal[len(pfx):]
            m = re.match(r"([^:,\.\?]+)", rest)
            if m:
                return m.group(1).strip()
    return goal.split(":")[0].strip()


def _phase_input():
    goal = st.text_area(
        "What company or fund would you like researched?",
        height=100,
        placeholder="e.g. Research Sequoia Capital: recent activity, leadership, and past deals.",
    )
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        run_clicked = st.button("Run Diligence", type="primary", use_container_width=True)
    with col2:
        if st.button("Example", use_container_width=True):
            goal = "Research Sequoia Capital: recent activity, leadership, and past deals."
            st.rerun()
    if run_clicked and goal.strip():
        st.session_state.goal = goal.strip()
        st.session_state.started_at = time.time()
        st.session_state.entity = _extract_entity_name(goal.strip())
        st.session_state.tracer = Tracer(st.session_state.goal)
        st.session_state.trace_events = []
        st.session_state.step_count = 0
        st.session_state.tool_call_count = 0
        st.session_state.phase = "planning"
        st.rerun()


def _phase_planning():
    status = st.status("Creating research plan ...", expanded=True)
    with status:
        plan = create_plan(st.session_state.goal, tracer=st.session_state.tracer)
        st.session_state.plan = plan
        st.session_state.step_count += 1
        _add_trace("plan", "Research Plan", plan)
        for s in plan["steps"]:
            st.write(f"**Step {s['step']}:** {s['action']}")
        st.success(f"Plan created with {len(plan['steps'])} steps")
    status.update(state="complete", label=f"Plan created ({len(plan['steps'])} steps)")
    st.session_state.phase = "gathering"
    st.rerun()


def _phase_gathering():
    status = st.status("Gathering evidence ...", expanded=True)
    entity = st.session_state.entity
    gathered = list(st.session_state.gathered or [])
    queries = [
        (f"{entity} recent news activity 2025 2026", "news"),
        (f"{entity} leadership partners team", "leadership"),
        (f"{entity} past deals investments portfolio", "past deals"),
    ]
    with status:
        for query, label in queries:
            if any(g.get("label") == label for g in gathered):
                continue
            status.write(f"Web search: {label} ...")
            result = execute_tool("web_search", {"query": query}, tracer=st.session_state.tracer)
            gathered.append({**result, "label": label})
            st.session_state.tool_call_count += 1
            _add_trace("tool", f"web_search: {label}", result.get("data", "")[:200], result.get("duration_sec"))
            src = result.get("source", "?")
            dur = result.get("duration_sec", 0) or 0
            fb = " (fallback)" if result.get("used_fallback") else ""
            ok = "OK" if result.get("success") else "FAIL"
            st.write(f"  {ok} {label}: {src}{fb} - {dur:.1f}s")
            st.session_state.step_count += 1
        if not any(g.get("label") == "SEC filings" for g in gathered):
            status.write(f"SEC EDGAR: {entity} ...")
            result = execute_tool("sec_edgar_lookup", {"company_name": entity}, tracer=st.session_state.tracer)
            gathered.append({**result, "label": "SEC filings"})
            st.session_state.tool_call_count += 1
            _add_trace("tool", f"sec_edgar_lookup: {entity}", result.get("data", "")[:200], result.get("duration_sec"))
            src = result.get("source", "?")
            dur = result.get("duration_sec", 0) or 0
            fb = " (fallback)" if result.get("used_fallback") else ""
            ok = "OK" if result.get("success") else "FAIL"
            st.write(f"  {ok} SEC EDGAR: {src}{fb} - {dur:.1f}s")
            st.session_state.step_count += 1
    st.session_state.gathered = gathered
    status.update(state="complete", label="Evidence gathered (4 sources)")
    st.session_state.phase = "synthesis"
    st.rerun()


def _phase_synthesis():
    status = st.status("Synthesizing research brief ...", expanded=True)
    brief = None
    with status:
        try:
            brief = synthesize_brief(
                st.session_state.goal,
                st.session_state.gathered,
                tracer=st.session_state.tracer,
            )
            st.session_state.step_count += 1
            _add_trace("synthesis", "Brief synthesized", f"Entity: {brief.entity_name}")
            st.success(f"Brief created for {brief.entity_name}")
            status.update(state="complete", label=f"Brief synthesized for {brief.entity_name}")
        except Exception as e:
            st.error(f"Synthesis failed: {e}")
            st.session_state.error = str(e)
            status.update(state="error", label="Synthesis failed")
    st.session_state.brief = brief
    if brief and brief.open_questions:
        st.session_state.review_questions = list(brief.open_questions)
        st.session_state.review_idx = 0
        st.session_state.review_decisions = []
        st.session_state.review_editing = False
        st.session_state.phase = "review"
    else:
        st.session_state.phase = "done"
    st.rerun()


def _phase_review():
    brief = st.session_state.brief
    questions = st.session_state.review_questions
    idx = st.session_state.review_idx
    if idx >= len(questions):
        _finalize_review()
        st.rerun()
        return
    question = questions[idx]
    st.markdown("## IC Review Required")
    st.markdown(f"The model flagged {len(questions)} item(s) needing committee review in this memo.")
    st.divider()
    st.markdown(f"### Question {idx + 1} of {len(questions)}")
    st.warning(question)
    if st.session_state.review_editing:
        correction = st.text_area(
            "Your correction:",
            value=st.session_state.review_edit_value or question,
            height=100,
            key="edit_text",
        )
        col_s, col_c = st.columns(2)
        with col_s:
            if st.button("Save correction", type="primary", use_container_width=True):
                st.session_state.review_decisions.append({
                    "question": question, "action": "edited", "correction": correction,
                })
                st.session_state.review_idx += 1
                st.session_state.review_editing = False
                st.session_state.review_edit_value = ""
                _add_trace("review", f"Edited: {question[:60]}...", correction)
                st.rerun()
        with col_c:
            if st.button("Cancel", use_container_width=True):
                st.session_state.review_editing = False
                st.rerun()
        return
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Confirm", type="primary", use_container_width=True):
            st.session_state.review_decisions.append({
                "question": question, "action": "confirmed", "correction": None,
            })
            st.session_state.review_idx += 1
            _add_trace("review", f"Confirmed: {question[:60]}...", None)
            st.rerun()
    with col2:
        if st.button("Reject", use_container_width=True):
            st.session_state.review_decisions.append({
                "question": question, "action": "rejected", "correction": None,
            })
            st.session_state.review_idx += 1
            _add_trace("review", f"Rejected: {question[:60]}...", None)
            st.rerun()
    with col3:
        if st.button("Edit", use_container_width=True):
            st.session_state.review_editing = True
            st.session_state.review_edit_value = question
            st.rerun()


def _finalize_review():
    brief = st.session_state.brief
    decisions = st.session_state.review_decisions
    final = brief.model_copy(deep=True)
    final.open_questions = []
    final.human_verified_claims = list(brief.human_verified_claims)
    for dec in decisions:
        action = dec["action"]
        if action == "confirmed":
            final.human_verified_claims.append(dec["question"])
        elif action == "edited":
            final.human_verified_claims.append(dec["correction"] or dec["question"])
    st.session_state.final_brief = final
    st.session_state.phase = "done"


def _render_brief_section(label: str, items: list, icon: str = ""):
    if not items:
        st.markdown(f"**{label}**  *None*")
        return
    st.markdown(f"**{label}**")
    for item in items:
        st.markdown(f"- {icon} {item}")


def _phase_done():
    brief = st.session_state.final_brief or st.session_state.brief
    if brief is None:
        st.error("No brief was generated.")
        if st.session_state.error:
            st.code(st.session_state.error)
        return
    data = brief.model_dump()
    st.divider()
    st.markdown(f"## Investment Committee Memo: {data['entity_name']}")
    st.markdown("### Overview")
    st.markdown(data["overview"])
    st.markdown("### Leadership")
    ldr = data.get("leadership", [])
    if ldr:
        for m in ldr:
            conf = m.get("source_confidence", "low")
            badges = {"high": "high", "medium": "medium", "low": "low"}
            st.markdown(f"- **{m['name']}** - {m['title']}  ({badges.get(conf, '?')})")
    else:
        st.markdown("*No specific leaders identified.*")
    col_a, col_b = st.columns(2)
    with col_a:
        _render_brief_section("Recent Activity", data.get("recent_activity", []))
    with col_b:
        _render_brief_section("Past Deals / Investments", data.get("past_deals", []))
    hv = data.get("human_verified_claims", [])
    if hv:
        st.markdown("### IC-Confirmed Findings")
        for c in hv:
            st.markdown(f"- OK {c}")
    oq = data.get("open_questions", [])
    if oq:
        st.markdown("### Items for IC Review")
        for q in oq:
            st.markdown(f"- ? {q}")
    else:
        st.markdown("### Items for IC Review")
        st.markdown("*None — all claims sufficient for review.*")
    su = data.get("sources_used", [])
    gen = data.get("generated_at", "")
    st.caption(f"Sources: {', '.join(su) if su else 'N/A'} | Generated: {gen}")
    with st.expander("Show reasoning trace", expanded=False):
        st.markdown("#### Research Plan")
        if st.session_state.plan:
            for s in st.session_state.plan["steps"]:
                st.markdown(f"**Step {s['step']}:** {s['action']}")
        st.markdown("#### Events")
        for evt in st.session_state.trace_events:
            dur_str = f" ({evt['duration']:.1f}s)" if isinstance(evt.get("duration"), (int, float)) else ""
            st.markdown(f"**[{evt['time']}]** {evt['label']}{dur_str}")
        st.markdown("#### RunGuard Summary")
        st.markdown(f"- Steps: {st.session_state.step_count}  Tool calls: {st.session_state.tool_call_count}")
    if st.session_state.tracer:
        st.caption(f"Trace saved to {st.session_state.tracer.path}")
    st.divider()

    # ── IC Memo generation ──────────────────────────────────────────────
    col_m1, col_m2 = st.columns([1, 5])
    with col_m1:
        if st.button("Generate IC Memo", type="primary", use_container_width=True):
            mandate_result = st.session_state.mandate_result
            memo = format_ic_memo(brief, mandate_result=mandate_result)
            st.session_state.ic_memo = memo
            st.rerun()

    if st.session_state.ic_memo:
        st.markdown("### Investment Committee Memo")
        with st.container(border=True):
            st.text(st.session_state.ic_memo)
        st.download_button(
            label="Download IC Memo (.md)",
            data=st.session_state.ic_memo,
            file_name=f"IC_Memo_{brief.entity_name.replace(' ', '_')}.md",
            mime="text/markdown",
            use_container_width=True,
        )
    st.divider()

    if st.button("Research something else", type="primary"):
        _reset()
        st.rerun()


# ===========================================================================
# Tabs
# ===========================================================================

tab1, tab2, tab3 = st.tabs(["Investment Committee Memo", "Mandate Fit Assessment", "Find Connections"])

# ---------------------------------------------------------------------------
# Tab 1 — existing diligence pipeline (unchanged)
# ---------------------------------------------------------------------------

with tab1:
    phase = st.session_state.phase
    if phase != "input" and st.session_state.goal:
        st.info(f"Researching: {st.session_state.goal}")

    try:
        if phase == "input":
            _phase_input()
        elif phase == "planning":
            _phase_planning()
        elif phase == "gathering":
            _phase_gathering()
        elif phase == "synthesis":
            _phase_synthesis()
        elif phase == "review":
            _phase_review()
        elif phase == "done":
            _phase_done()
    except RunLimitExceeded as e:
        labels = {"step_ceiling": "Step limit", "time_ceiling": "Time limit", "cost_ceiling": "Cost limit"}
        label = labels.get(e.ceiling, e.ceiling)
        st.error(f"Run stopped by {label}")
        st.markdown(f"- **Ceiling:** {e.ceiling}")
        st.markdown(f"- **Limit:** {e.limit}")
        st.markdown(f"- **Actual:** {e.actual}")
        st.session_state.phase = "done"
    except Exception as e:
        st.error(f"Unexpected error: {type(e).__name__}: {e}")
        st.session_state.error = str(e)
        st.session_state.phase = "done"

# ---------------------------------------------------------------------------
# Tab 2 — Mandate Fit Assessment
# ---------------------------------------------------------------------------

with tab2:
    st.markdown("## Mandate Fit Assessment")
    brief = st.session_state.final_brief or st.session_state.brief

    if brief is None:
        st.info("No diligence brief available yet. Run a research goal in the **Investment Committee Memo** tab first.")
    else:
        st.success(f"Using brief for: **{brief.entity_name}**")

        with st.expander("Mandate criteria", expanded=True):
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                sectors_text = st.text_input("Sectors (comma-separated)", value="fintech, financial services")
                stage = st.text_input("Preferred stage", value="early-stage (Series A and earlier)")
                geo_text = st.text_input("Geography (comma-separated)", value="United States")
            with col_s2:
                check_min = st.number_input("Min check size ($)", min_value=0, value=1_000_000, step=500_000)
                check_max = st.number_input("Max check size ($)", min_value=0, value=5_000_000, step=500_000)
                excl_text = st.text_input("Excluded industries (comma-separated)", value="cryptocurrency, gambling")

        if st.button("Run Match", type="primary", use_container_width=True):
            st.session_state.mandate_running = True
            sectors = [s.strip() for s in sectors_text.split(",") if s.strip()]
            geography = [g.strip() for g in geo_text.split(",") if g.strip()]
            excluded = [e.strip() for e in excl_text.split(",") if e.strip()]

            mandate = InvestmentMandate(
                sectors=sectors,
                stage=stage,
                check_size_min=check_min,
                check_size_max=check_max,
                geography=geography,
                excluded_industries=excluded,
            )

            with st.status("Running mandate match ...", expanded=True) as s:
                result = match_mandate(brief, mandate)
                st.session_state.mandate_result = result
                s.update(state="complete", label="Match complete")

            st.session_state.mandate_running = False

        # Display results
        if st.session_state.mandate_result:
            result = st.session_state.mandate_result
            score = result.get("score", 0)
            reasoning = result.get("reasoning", [])
            uncertain = result.get("uncertain_fields", [])

            score_color = "green" if score >= 70 else "orange" if score >= 40 else "red"
            st.markdown("### Fit Score")
            st.markdown(
                f"<h1 style='color: {score_color}; text-align: center;'>{score}/100</h1>",
                unsafe_allow_html=True,
            )

            st.markdown("### Field-by-field reasoning")
            for r in reasoning:
                field = r.get("field", "?")
                verdict = r.get("verdict", "?")
                detail = r.get("detail", "")
                is_uncertain = verdict == "unclear" or field in uncertain
                icon = {"match": "✅", "unclear": "⚠️", "mismatch": "❌"}.get(verdict, "❓")
                box_style = "warning" if is_uncertain else "success" if verdict == "match" else "error"
                st.markdown(
                    f"<div style='padding: 8px; margin: 4px 0; border-radius: 6px; "
                    f"background-color: {'#fff3cd' if is_uncertain else '#d4edda' if verdict == 'match' else '#f8d7da'};'>"
                    f"<strong>{icon} {field}</strong>: {verdict}<br>{detail}</div>",
                    unsafe_allow_html=True,
                )

            if uncertain:
                st.warning(f"Uncertain fields: {', '.join(uncertain)}")

# ---------------------------------------------------------------------------
# Tab 3 — Find Connections
# ---------------------------------------------------------------------------

with tab3:
    st.markdown("## Entity Connections")
    st.markdown("Search for connections between two companies, funds, or investors.")

    col_a, col_b = st.columns(2)
    with col_a:
        entity_a = st.text_input("Entity A", value="Sequoia Capital", key="conn_a")
    with col_b:
        entity_b = st.text_input("Entity B", value="Stripe", key="conn_b")

    if st.button("Search Connections", type="primary", use_container_width=True):
        if entity_a.strip() and entity_b.strip():
            st.session_state.conn_running = True

            with st.status(f"Searching connections between {entity_a} and {entity_b} ...", expanded=True) as s:
                result = find_connections(entity_a.strip(), entity_b.strip())
                st.session_state.conn_result = result
                s.update(state="complete", label=f"Found {len(result.get('connections', []))} connection(s)")

            st.session_state.conn_running = False
        else:
            st.error("Please enter both entity names.")

    # Display results
    if st.session_state.conn_result:
        result = st.session_state.conn_result
        connections = result.get("connections", [])
        confidence = result.get("confidence", "low")
        not_found = result.get("searched_but_not_found", [])

        conf_badge = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(confidence, "⚪")
        st.markdown(f"### Results  {conf_badge} confidence: {confidence}")

        if connections:
            st.markdown("#### Connections found")
            for i, conn in enumerate(connections, 1):
                ctype = conn.get("type", "?")
                desc = conn.get("description", "")
                src = conn.get("source", "")
                type_icons = {
                    "shared_investor": "🏦",
                    "shared_board": "👥",
                    "co_investment": "💰",
                    "partnership": "🤝",
                    "funding": "💵",
                    "other": "🔗",
                }
                icon = type_icons.get(ctype, "🔗")
                with st.container(border=True):
                    st.markdown(f"**{icon} Connection {i}** — {ctype}")
                    st.markdown(desc)
                    if src:
                        st.markdown(f"*Source: {src}*")
        else:
            st.info("No direct connections found between these entities.")

        if not_found:
            st.markdown("#### Checked but not found")
            for item in not_found:
                st.markdown(f"- ❌ {item}")
            st.caption("These searches returned no relevant results — included for transparency.")

    # -------------------------------------------------------------------
    # Conflict-of-Interest Check
    # -------------------------------------------------------------------
    st.divider()
    st.markdown("## Conflict-of-Interest Check")
    st.markdown(
        "Check whether the researched entity has any conflicts of interest "
        "with your existing portfolio. Enter one portfolio entity per line."
    )

    # Use the brief entity if available, otherwise free-text
    brief_for_conflict = st.session_state.final_brief or st.session_state.brief
    default_entity = brief_for_conflict.entity_name if brief_for_conflict else ""

    col_e, col_p = st.columns(2)
    with col_e:
        conflict_entity = st.text_input(
            "Entity to check",
            value=default_entity,
            key="conflict_entity",
        )
    with col_p:
        portfolio_text = st.text_area(
            "Existing Portfolio (one per line)",
            height=100,
            placeholder="BlackRock\nGoldman Sachs\nAndreessen Horowitz",
            key="portfolio_entities",
        )

    if st.button("Check for Conflicts", type="primary", use_container_width=True):
        if conflict_entity.strip():
            portfolio_list = [
                p.strip() for p in portfolio_text.split("\n")
                if p.strip()
            ]
            st.session_state.conflict_running = True

            with st.status(
                f"Checking {conflict_entity} against "
                f"{len(portfolio_list)} portfolio entit(ies) ...",
                expanded=True,
            ) as s:
                result = check_conflicts(conflict_entity.strip(), portfolio_list)
                st.session_state.conflict_result = result
                n_conflicts = len(result.get("conflicts_found", []))
                n_clean = len(result.get("no_conflicts", []))
                s.update(
                    state="complete",
                    label=f"Found {n_conflicts} conflict(s), "
                           f"{n_clean} entit(ies) with no conflicts",
                )

            st.session_state.conflict_running = False
        else:
            st.error("Please enter an entity name to check.")

    # Display conflict results
    if st.session_state.conflict_result:
        result = st.session_state.conflict_result
        message = result.get("message", "")
        conflicts = result.get("conflicts_found", [])
        no_conflicts = result.get("no_conflicts", [])
        overall_confidence = result.get("overall_confidence", "low")

        conf_badge = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(overall_confidence, "⚪")

        if not result.get("portfolio_checked"):
            st.info(message)
        elif conflicts:
            st.markdown(
                f"### ⚠ Potential Conflicts  {conf_badge} "
                f"confidence: {overall_confidence}"
            )
            for i, conf in enumerate(conflicts, 1):
                ctype = conf.get("type", "?")
                desc = conf.get("description", "")
                src = conf.get("source", "")
                port_entity = conf.get("portfolio_entity", "?")
                type_icons = {
                    "shared_investor": "🏦",
                    "shared_board": "👥",
                    "co_investment": "💰",
                    "partnership": "🤝",
                    "funding": "💵",
                    "other": "🔗",
                }
                icon = type_icons.get(ctype, "🔗")
                with st.container(border=True):
                    st.markdown(
                        f"**{icon} Conflict {i}** — {ctype} "
                        f"_(with {port_entity})_"
                    )
                    st.markdown(desc)
                    if src:
                        st.markdown(f"*Source: {src}*")
        else:
            st.success(
                f"✅ No conflicts found between **{result.get('entity', '')}** "
                f"and the {len(no_conflicts)} portfolio entit(ies) checked."
            )

        st.caption(f"Overall confidence: {overall_confidence} | {message}")
