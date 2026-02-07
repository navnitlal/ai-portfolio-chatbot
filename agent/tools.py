"""Declared tools for the portfolio chatbot. The LLM chooses which tool to call from these definitions."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal, Tuple

import pandas as pd
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

from core.analytics import allocation_on_date, max_drawdown, time_weighted_return, held_asset_classes_recent
from core.prompts import RISK_QUESTIONS, RISK_QUESTION_OPTIONS, NUM_RISK_QUESTIONS
from core.risk import score_risk_answers, infer_risk_from_allocation, TARGET_ALLOCATION_BY_RISK
from agent.trends import search_trends, summarize_and_recommend

class GetPerformanceArgs(BaseModel):
    """Input for get_performance."""

    start_date: str = Field(description="Start date in YYYY-MM-DD format")
    end_date: str = Field(description="End date in YYYY-MM-DD format")
    asset_class: Optional[Literal["Stock", "Bond", "Cash", "ETF"]] = Field(
        default=None,
        description="Optional: compute return for this asset class only; omit for total portfolio",
    )


class SubmitRiskAnswerArgs(BaseModel):
    """Input for submit_risk_answer."""

    question_number: Literal["1", "2", "3", "4"] = Field(
        description="Which risk question (1-4) the user is answering"
    )
    answer_text: str = Field(description="The user's answer in their own words")


class FetchMarketTrendsArgs(BaseModel):
    """Input for fetch_market_trends."""

    asset_classes: List[Literal["Stock", "Bond", "Cash", "ETF"]] = Field(
        default_factory=lambda: ["Stock", "Bond", "Cash", "ETF"],
        description="Which asset classes to search; default all four if user has no portfolio",
    )


class SuggestRebalanceArgs(BaseModel):
    """Input for suggest_rebalance."""

    target_risk: Optional[Literal["Conservative", "Moderate", "Aggressive"]] = Field(
        default=None,
        description="If the user asks for a rebalance to a specific risk category (e.g. 'rebalance to aggressive', 'target for conservative'), pass that category here. Omit to use stored or inferred risk.",
    )


# --- Tools for binding to the LLM ---


def get_binding_tools() -> List[StructuredTool]:
    """Tools to pass to model.bind_tools(). Only name, description, and args_schema are used by the LLM."""
    return [
        StructuredTool.from_function(
            name="get_capabilities",
            description="Return a short description of what the assistant can do (performance calculation, risk questionnaire, market trends, rebalance suggestion). The assistant is read-only and cannot modify any data; portfolio updates happen via CSV upload in the sidebar. Use when the user asks what you can do or for help.",
            func=lambda: "",
        ),
        StructuredTool.from_function(
            name="get_edit_help",
            description="Respond when the user asks to edit, add, update, or remove portfolio data. First time: suggest uploading the latest portfolio CSV via the sidebar. If they ask again without having uploaded: advise contacting a human advisor. Use when the user wants to change or edit portfolio data.",
            func=lambda: "",
        ),
        StructuredTool.from_function(
            name="get_current_risk",
            description="Return the user's current risk category inferred from their latest portfolio allocation. Always calculates from the most recent portfolio data (Stock/Bond/Cash/ETF ratios), not from a stored profile. Call this whenever the user asks 'what is my risk', 'current risk category of the portfolio', 'risk profile', etc. After calling, output ONLY the tool result—no extra explanation, no 'I labeled', no next steps. Do NOT call get_next_risk_question for a simple risk-category question.",
            func=lambda: "",
        ),
        StructuredTool.from_function(
            name="suggest_rebalance",
            description="Suggest a rebalance ratio and how to achieve it (read-only — does NOT change any data). If the user asks for a specific risk category (e.g. 'rebalance to aggressive', 'target for conservative'), pass target_risk with that category; otherwise uses stored or inferred risk. Returns current allocation, target allocation, and steps. Do not offer to apply or execute the rebalance.",
            args_schema=SuggestRebalanceArgs,
            func=lambda: "",
        ),
        StructuredTool.from_function(
            name="get_next_risk_question",
            description="Start the risk questionnaire or get the next question. Call ONLY this (no other tools) when the user says they want to run or redo the questionnaire (e.g. 'run the questionnaire', 'set my risk profile'). Do not call submit_risk_answer in the same turn—that would skip question 1.",
            func=lambda: "",
        ),
        StructuredTool.from_function(
            name="submit_risk_answer",
            description="Process the user's answer to a risk question (1-4) in the current session and return the next question or final inferred profile. This does NOT persist anything to the database — the risk profile is computed in-session only. Call only when the user has just answered a question (e.g. they replied 'Hold', 'Buy more', or chose an option after seeing 'Question N of 4'). Do not call this when the user is only asking to start the questionnaire.",
            args_schema=SubmitRiskAnswerArgs,
            func=lambda question_number, answer_text: "",
        ),
        StructuredTool.from_function(
            name="get_performance",
            description="Compute portfolio performance from stored SQLite data. Uses start_date and end_date (YYYY-MM-DD) to load the date range from the database; start value = total portfolio on start date, end value = total portfolio on end date (from SQLite). Returns period return and max drawdown. Prefer this whenever the user has loaded portfolio data and asks for return/performance over a period — do not ask the user for start/end market values; they are read from the store. If no data in that range, say so and direct the user to use the sidebar to upload a CSV file with their portfolio data.",
            args_schema=GetPerformanceArgs,
            func=lambda start_date, end_date, asset_class=None: "",
        ),
        StructuredTool.from_function(
            name="fetch_market_trends",
            description="Fetch recent market trends/news via Tavily for the given asset classes. Call immediately when the user asks for latest trends/news/outlook/market updates (no permission/consent step). After calling, output ONLY the tool result (it includes a Sources section with URLs).",
            args_schema=FetchMarketTrendsArgs,
            func=lambda asset_classes=None: "",
        ),
        StructuredTool.from_function(
            name="apply_risk_profile_change",
            description="Note the user's preferred risk profile for the current session based on trend analysis or their explicit request. This is session-only and does NOT persist to the database. The assistant cannot make permanent changes. Call only when the user explicitly states they want to note a different risk category for this session.",
            func=lambda: "",
        ),
    ]


# --- Handlers: (args, state, config) -> (result_str, state_updates) ---


def _handler_get_capabilities(
    _args: Dict[str, Any], _state: Dict[str, Any], _config: Dict[str, Any]
) -> Tuple[str, Dict[str, Any]]:
    result = (
        "I am a read-only portfolio assistant. I can: "
        "(1) Calculate performance (return and max drawdown) over a date range from your uploaded data; "
        "(2) Run a 4-question risk questionnaire to infer your risk profile; "
        "(3) Fetch latest market trends and news via web search; "
        "(4) Suggest a rebalance ratio based on your risk category. "
        "I cannot edit, save, or modify any data. To update your portfolio, upload a CSV file via the sidebar."
    )
    return result, {}


def _handler_get_edit_help(
    _args: Dict[str, Any], state: Dict[str, Any], _config: Dict[str, Any]
) -> Tuple[str, Dict[str, Any]]:
    if state.get("already_suggested_upload_for_edit"):
        return (
            "I don't have permission to edit your portfolio. Please contact a human advisor for changes."
        ), {}
    return (
        "I cannot edit your portfolio. Please upload your latest portfolio file (CSV) using the upload button in the sidebar to update your data."
    ), {"already_suggested_upload_for_edit": True}


def _handler_get_current_risk(
    _args: Dict[str, Any], state: Dict[str, Any], config: Dict[str, Any]
) -> Tuple[str, Dict[str, Any]]:
    """Always infer risk from latest portfolio allocation (not stored profile)."""
    store = config.get("configurable", {}).get("store")
    if not store:
        return (
            "No portfolio data available. "
            "Use the sidebar to upload a CSV file with your portfolio data to calculate your risk category from your allocation.",
            {},
        )
    df = store.load_all()
    if df.empty:
        return (
            "No portfolio data loaded. "
            "Use the sidebar to upload a CSV file to calculate your risk category from your allocation.",
            {},
        )
    last_date = df["Date"].max()
    alloc = allocation_on_date(df, last_date)
    inferred = infer_risk_from_allocation(
        alloc["Stock"], alloc["Bond"], alloc["Cash"], alloc["ETF"]
    )
    s = round(alloc["Stock"] * 100, 1)
    b = round(alloc["Bond"] * 100, 1)
    c = round(alloc["Cash"] * 100, 1)
    e = round(alloc["ETF"] * 100, 1)
    stored = state.get("risk_profile")
    stored_note = f" (Stored profile: **{stored}**)" if stored and stored != inferred else ""
    return (
        f"Based on your current portfolio allocation (as of {last_date}): "
        f"Stock **{s}%**, Bond **{b}%**, Cash **{c}%**, ETF **{e}%** → inferred risk category: **{inferred}**{stored_note}. "
        "Would you like to know how much you'd need to rebalance for a different risk category (Conservative, Moderate, or Aggressive)? Ask me to suggest a rebalance.",
        {},
    )


def _handler_suggest_rebalance(
    args: Dict[str, Any], state: Dict[str, Any], config: Dict[str, Any]
) -> Tuple[str, Dict[str, Any]]:
    """Suggest rebalance ratio and how to achieve it. Uses target_risk if provided and valid, else stored or inferred risk."""
    store = config.get("configurable", {}).get("store")
    if not store:
        return "Portfolio data is not available. Use the sidebar to upload a CSV file to get a rebalance suggestion.", {}

    df = store.load_all()
    if df.empty:
        return "No portfolio data loaded. Use the sidebar to upload a CSV file to get a rebalance suggestion based on your risk category.", {}

    last_date = df["Date"].max()
    alloc = allocation_on_date(df, last_date)
    s_cur = alloc["Stock"]
    b_cur = alloc["Bond"]
    c_cur = alloc["Cash"]
    e_cur = alloc["ETF"]
    inferred = infer_risk_from_allocation(s_cur, b_cur, c_cur, e_cur)
    stored = state.get("risk_profile")
    requested = (args or {}).get("target_risk")
    # Use requested category if valid, else stored, else inferred
    risk = (requested if requested in TARGET_ALLOCATION_BY_RISK else None) or (stored if stored in TARGET_ALLOCATION_BY_RISK else None) or inferred
    t = TARGET_ALLOCATION_BY_RISK.get(risk)
    if not t:
        return "Could not determine a target allocation for your risk category.", {}

    s_tgt, b_tgt, c_tgt, e_tgt = t
    # Percentage-point deltas
    ds = round((s_tgt - s_cur) * 100, 1)
    db = round((b_tgt - b_cur) * 100, 1)
    dc = round((c_tgt - c_cur) * 100, 1)
    de = round((e_tgt - e_cur) * 100, 1)

    # One-line context: requested vs stored vs inferred (avoid "no stored" when user asked for a specific target)
    if requested and requested in TARGET_ALLOCATION_BY_RISK and requested == risk:
        risk_note = f"Target you asked for: **{risk}**. (Current allocation implies **{inferred}**.)"
    elif stored:
        risk_note = f"Stored risk: **{stored}**."
    else:
        risk_note = f"Inferred from allocation: **{inferred}**."
    lines = [
        f"**As of {last_date}**: Stock **{round(s_cur*100, 1)}%**, Bond **{round(b_cur*100, 1)}%**, Cash **{round(c_cur*100, 1)}%**, ETF **{round(e_cur*100, 1)}%**.",
        risk_note,
        f"**Target allocation for {risk}** (cash kept to a minimum so more is in Stock/Bond/ETF): Stock **{round(s_tgt*100, 0)}%**, Bond **{round(b_tgt*100, 0)}%**, Cash **{round(c_tgt*100, 0)}%**, ETF **{round(e_tgt*100, 0)}%**.",
        "**How to achieve it:**",
    ]
    steps = []
    if abs(ds) >= 0.5:
        steps.append(f"Move **{abs(ds):.1f}** percentage points {'into' if ds > 0 else 'out of'} Stock.")
    if abs(db) >= 0.5:
        steps.append(f"Move **{abs(db):.1f}** percentage points {'into' if db > 0 else 'out of'} Bond.")
    if abs(dc) >= 0.5:
        steps.append(f"Move **{abs(dc):.1f}** percentage points {'into' if dc > 0 else 'out of'} Cash.")
    if abs(de) >= 0.5:
        steps.append(f"Move **{abs(de):.1f}** percentage points {'into' if de > 0 else 'out of'} ETF.")
    if not steps:
        lines.append("Your allocation is already close to the target; no change needed.")
    else:
        lines.append(" " + " ".join(steps))
        # Clarify why we move into/out of each: only assets below target get inflows; above target get outflows.
        below = [("Stock", ds)] if ds > 0.5 else []
        if db > 0.5:
            below.append(("Bond", db))
        if dc > 0.5:
            below.append(("Cash", dc))
        if de > 0.5:
            below.append(("ETF", de))
        if below:
            names = " and ".join(b[0] for b in below)
            lines.append(f"\n*Why into {names}? Your {' and '.join(n.lower() for n in names.split(' and '))} {'is' if len(below) == 1 else 'are'} below the target; the rest are above target, so rebalance by reducing the overweight and adding to the underweight.*")
    lines.append("\n*I only suggest — I cannot change your portfolio; any trades or edits require human advisor approval.*")
    return "\n\n".join(lines), {}


def _handler_get_next_risk_question(
    _args: Dict[str, Any], state: Dict[str, Any], _config: Dict[str, Any]
) -> Tuple[str, Dict[str, Any]]:
    risk_answers = dict(state.get("risk_answers") or {})
    question_keys = [str(i) for i in range(1, NUM_RISK_QUESTIONS + 1)]
    # If the questionnaire was already completed, starting it again should restart from Q1.
    if all(k in risk_answers for k in question_keys):
        risk_answers = {}
        q = next((q for q in RISK_QUESTIONS if q.startswith("1)")), "")
        q_text = q.split(" (")[0].strip() if " (" in q else q
        opts = RISK_QUESTION_OPTIONS[0] if RISK_QUESTION_OPTIONS else ""
        return (
            f"**Question 1 of {NUM_RISK_QUESTIONS}**\n\n{q_text}\n\n{opts}",
            {"risk_answers": {}, "risk_profile": None},
        )
    for i, k in enumerate(question_keys):
        if k not in risk_answers:
            q = next((q for q in RISK_QUESTIONS if q.startswith(k + ")")), "")
            # Question text without the parenthetical options; we add options line below
            q_text = q.split(" (")[0].strip() if " (" in q else q
            opts = RISK_QUESTION_OPTIONS[i] if i < len(RISK_QUESTION_OPTIONS) else ""
            return f"**Question {i + 1} of {NUM_RISK_QUESTIONS}**\n\n{q_text}\n\n{opts}", {}
    profile = score_risk_answers(risk_answers)
    return f"Your inferred risk profile is: **{profile}**.", {"risk_profile": profile}


def _handler_submit_risk_answer(
    args: Dict[str, Any], state: Dict[str, Any], _config: Dict[str, Any]
) -> Tuple[str, Dict[str, Any]]:
    qnum = args.get("question_number", "")
    answer = (args.get("answer_text") or "").strip()
    risk_answers = dict(state.get("risk_answers") or {})
    # Safeguard: if no prior answers and no real answer text, LLM likely called submit when user only asked to start—show Q1 and do not record.
    if not risk_answers and not answer:
        q = next((q for q in RISK_QUESTIONS if q.startswith("1)")), "")
        q_text = q.split(" (")[0].strip() if " (" in q else q
        opts = RISK_QUESTION_OPTIONS[0] if RISK_QUESTION_OPTIONS else ""
        return f"**Question 1 of {NUM_RISK_QUESTIONS}**\n\n{q_text}\n\n{opts}", {}
    risk_answers[qnum] = answer
    question_keys = [str(i) for i in range(1, NUM_RISK_QUESTIONS + 1)]
    for i, k in enumerate(question_keys):
        if k not in risk_answers:
            q = next((q for q in RISK_QUESTIONS if q.startswith(k + ")")), "")
            q_text = q.split(" (")[0].strip() if " (" in q else q
            opts = RISK_QUESTION_OPTIONS[i] if i < len(RISK_QUESTION_OPTIONS) else ""
            return f"**Question {i + 1} of {NUM_RISK_QUESTIONS}**\n\n{q_text}\n\n{opts}", {"risk_answers": risk_answers}
    profile = score_risk_answers(risk_answers)
    return f"All done. Your inferred risk profile is: **{profile}**.", {"risk_answers": risk_answers, "risk_profile": profile}


def _handler_get_performance(
    args: Dict[str, Any], _state: Dict[str, Any], config: Dict[str, Any]
) -> Tuple[str, Dict[str, Any]]:
    store = config.get("configurable", {}).get("store")
    if not store:
        return (
            "No portfolio data available. Use the sidebar to upload a CSV file with your daily portfolio values (Date, StockValue, BondValue, ETFValue, CashValue) to calculate performance.",
            {},
        )

    # Require data loaded before asking/using dates
    df_all = store.load_all()
    if df_all.empty:
        return (
            "No portfolio data loaded. Use the sidebar to upload a CSV file with your daily portfolio values (Date, StockValue, BondValue, ETFValue, CashValue) to calculate performance.",
            {},
        )

    req_start = pd.to_datetime(args["start_date"]).date()
    req_end = pd.to_datetime(args["end_date"]).date()
    asset_class = args.get("asset_class")

    min_d = df_all["Date"].min()
    max_d = df_all["Date"].max()
    # Clamp requested dates to available range (nearest available dates)
    start = min(max(req_start, min_d), max_d)
    end = min(max(req_end, min_d), max_d)
    if start > end:
        start, end = min_d, max_d

    df_range = store.load_range(start, end)
    if df_range.empty:
        return "I don't have any stored daily values yet. Use the sidebar to upload a CSV file with your portfolio data.", {}

    pr = time_weighted_return(df_range, start, end, asset_class=asset_class)
    n = pr.get("sub_periods", 0)
    method_note = f" (Time-Weighted Return{f', {n} sub-periods' if n > 1 else ''})"

    mdd = max_drawdown(df_range, start, end) if asset_class is None else None
    what = "Total portfolio" if asset_class is None else f"{asset_class} asset class"
    why_note = ""
    if (req_start, req_end) != (start, end):
        why_note = (
            f"\n(Note: requested range is not found exactly in the stored data  {req_start.isoformat()} → {req_end.isoformat()}, "
            f"but stored data is from {min_d.isoformat()} → {max_d.isoformat()}, "
            f"so the nearest available dates {start.isoformat()} → {end.isoformat()} were used to calculate the performance.)\n"
        )
    msg = (
        f"Performance for **{what}** from **{start.isoformat()}** to **{end.isoformat()}**:{why_note}\n"
        f"- Start value on **{start.isoformat()}**: **{pr['start_value']:.2f}**\n"
        f"- End value on **{end.isoformat()}**: **{pr['end_value']:.2f}**\n"
        f"- Return for this period: **{pr['return_pct']:.2f}%**{method_note}\n"
    )
    if mdd is not None:
        msg += f"- Max drawdown (total): **{mdd:.2f}%**\n"
    return msg, {"did_compute_metrics": True}


def _handler_fetch_market_trends(
    args: Dict[str, Any], state: Dict[str, Any], config: Dict[str, Any]
) -> Tuple[str, Dict[str, Any]]:

    store = config.get("configurable", {}).get("store")
    main_llm = config.get("configurable", {}).get("main_llm")
    if not main_llm:
        return "LLM not available for summarizing trends.", {}

    # Determine topics for trend search.
    # - If caller passes asset_classes explicitly, respect that.
    # - Otherwise, include BOTH a broad "Global markets" topic AND the user's held asset classes.
    base_asset_classes = args.get("asset_classes")
    if base_asset_classes:
        topics = list(base_asset_classes)
    else:
        topics = ["Global markets"]
        default_assets = ["Stock", "Bond", "Cash", "ETF"]
        if store:
            df_all = config.get("configurable", {}).get("store").load_all()
            held_assets = held_asset_classes_recent(df_all) or default_assets
        else:
            held_assets = default_assets
        topics.extend(held_assets)

    # Deduplicate while preserving order
    seen_topics: set[str] = set()
    held = []
    for t in topics:
        if t not in seen_topics:
            seen_topics.add(t)
            held.append(t)

    trends_result = search_trends(held)
    rec = summarize_and_recommend(main_llm, trends_result, state.get("risk_profile"))
    summary = rec.get("summary", {})
    rr = rec.get("risk_recommendation", {})

    # Always include sources for any internet-derived content.
    # We track per-asset URLs so we can reference them inline as superscript-style links.
    asset_sources: Dict[str, List[str]] = {}
    seen_urls: set[str] = set()
    for ac in held:
        items = trends_result.get(ac, [])
        urls: List[str] = []
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                r = item.get("result", {})
                if not isinstance(r, dict):
                    continue
                results_list = r.get("results", [])
                if not isinstance(results_list, list):
                    continue
                for res in results_list:
                    if not isinstance(res, dict):
                        continue
                    url = (res.get("url") or "").strip()
                    if not url or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    urls.append(url)
        if urls:
            asset_sources[ac] = urls

    # Helper to render text as Unicode superscript, including digits and parentheses
    def _superscript(n: int | str) -> str:
        mapping = {
            "0": "\u2070",
            "1": "\u00b9",
            "2": "\u00b2",
            "3": "\u00b3",
            "4": "\u2074",
            "5": "\u2075",
            "6": "\u2076",
            "7": "\u2077",
            "8": "\u2078",
            "9": "\u2079",
            "(": "\u207d",
            ")": "\u207e",
        }
        return "".join(mapping.get(ch, ch) for ch in str(n))

    blocks = []
    for ac in held:
        pts = summary.get(ac, [])
        if pts:
            urls = asset_sources.get(ac, [])
            lines: List[str] = []
            for i, pt in enumerate(pts[:4]):
                # Attach a fully superscript-style hyperlink label, e.g. [superscript(1)], without exposing the raw URL
                if i < len(urls):
                    label = _superscript(f"({i + 1})")
                    # Markdown link where only the superscript label is visible/clickable
                    suffix = f" [{label}]({urls[i]})"
                else:
                    # If we somehow have fewer URLs than points, fall back to no link for that point
                    suffix = ""
                lines.append(f"- {pt}{suffix}")
            blocks.append(f"**{ac}**\n" + "\n".join(lines))

    suggested_change = rr.get("suggested_change", "none")
    suggested_profile = rr.get("suggested_profile")
    reason = rr.get("reason", "")

    # Do not mention the underlying provider name (e.g. Tavily) in user-facing text.
    # Sources are now shown only as inline numbered hyperlinks [1], [2] next to each bullet.
    msg = "Latest trend summary:\n\n" + "\n\n".join(blocks) + "\n\n"
    if suggested_change == "none" or not suggested_profile:
        msg += f"Risk profile: **no change suggested**. {reason}"
        return msg, {"pending_kind": "none", "pending_payload": {}, "did_use_tavily": True}
    msg += (
        f"Risk profile suggestion: **consider {suggested_change} risk** \u2192 **{suggested_profile}**.\n"
        f"Reason: {reason}"
    )
    # Do NOT auto-prompt to update profile here; keep it informational only.
    return msg, {"pending_kind": "none", "pending_payload": {}, "did_use_tavily": True}


def _handler_apply_risk_profile_change(
    _args: Dict[str, Any], state: Dict[str, Any], _config: Dict[str, Any]
) -> Tuple[str, Dict[str, Any]]:
    if state.get("pending_kind") != "apply_risk_change":
        return "No pending risk profile change. Use fetch_market_trends first to see trend-based observations.", {}

    prof = (state.get("pending_payload") or {}).get("suggested_profile")
    if prof not in ("Conservative", "Moderate", "Aggressive"):
        return "I couldn't note that risk profile (invalid value).", {"pending_kind": "none", "pending_payload": {}}
    # Session-only: update the in-memory risk profile but do NOT persist to SQLite.
    # The assistant is read-only and cannot make permanent data changes.
    return (
        f"Noted \u2014 your risk profile for this session is **{prof}**. "
        "This is not saved permanently. To make lasting changes, consult your financial advisor.",
        {"pending_kind": "none", "pending_payload": {}, "risk_profile": prof},
    )


# Map tool name -> handler for the custom tools node
TOOL_HANDLERS: Dict[str, Any] = {
    "get_capabilities": _handler_get_capabilities,
    "get_edit_help": _handler_get_edit_help,
    "get_current_risk": _handler_get_current_risk,
    "suggest_rebalance": _handler_suggest_rebalance,
    "get_next_risk_question": _handler_get_next_risk_question,
    "submit_risk_answer": _handler_submit_risk_answer,
    "get_performance": _handler_get_performance,
    "fetch_market_trends": _handler_fetch_market_trends,
    "apply_risk_profile_change": _handler_apply_risk_profile_change,
}


def run_tool(
    name: str,
    args: Dict[str, Any],
    state: Dict[str, Any],
    config: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    """Execute a tool by name; returns (result_text, state_updates)."""
    handler = TOOL_HANDLERS.get(name)
    if not handler:
        return f"Unknown tool: {name}.", {}
    return handler(args, state, config)
