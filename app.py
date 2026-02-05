from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

import pandas as pd
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from analytics import normalize_wide_csv, allocation_on_date
from storage import SQLiteStore
from graph import ChatState, build_graph, _state_to_dict, _dict_to_state
from policy import ExecutionFlags, enforce_policy
from tools import run_tool
from risk import infer_risk_from_allocation

PROJECT_NAME = "ai-portfolio-chatbot"
GREETING = "Hi, I am your personal portfolio assistant. What can I help you with today?"
DB_PATH = str(Path(__file__).parent / "portfolio.db")


def get_llms():
    main_llm = ChatOpenAI(model="gpt-5-mini", temperature=3)
    judge_llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
    policy_llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
    return main_llm, judge_llm, policy_llm


def ensure_session():
    if "main_llm" not in st.session_state:
        st.session_state.main_llm, st.session_state.judge_llm, st.session_state.policy_llm = get_llms()
    if "store" not in st.session_state:
        store = SQLiteStore(DB_PATH)
        store.init()
        st.session_state.store = store
    if "portfolio_cleared_on_start" not in st.session_state:
        st.session_state.store.clear_portfolio()
        st.session_state.portfolio_cleared_on_start = True
    if "graph" not in st.session_state:
        st.session_state.graph = build_graph(st.session_state.main_llm)
    if "state" not in st.session_state:
        st.session_state.state = ChatState(messages=[AIMessage(content=GREETING)])
    if "risk_loaded" not in st.session_state:
        rp = st.session_state.store.get_risk_profile()
        if rp:
            st.session_state.state.risk_profile = rp
        st.session_state.risk_loaded = True
    if "sidebar_messages" not in st.session_state:
        st.session_state.sidebar_messages = []


def render_chat(messages, risk_buttons=None):
    """Render chat messages. If risk_buttons is (q_num, options), the last message must be an AI risk question: show content without the 'Choose one:' line and render option buttons inside that assistant bubble. Returns the chosen option text if a risk button was clicked, else None. ToolMessages are skipped so the final AIMessage (which echoes the tool result) is shown once."""
    chosen: Optional[str] = None
    for i, m in enumerate(messages):
        if isinstance(m, ToolMessage):
            continue  # Don't show tool results; the final AIMessage contains the same content
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None) and not (getattr(m, "content", None) or "").strip():
            continue  # Skip AIMessage that only requested tool calls (no content); final AIMessage has the reply
        if isinstance(m, HumanMessage):
            with st.chat_message("user"):
                st.markdown(m.content)
        else:
            with st.chat_message("assistant"):
                is_last = i == len(messages) - 1
                if is_last and risk_buttons:
                    q_num, options = risk_buttons
                    display_content = _strip_choose_one_line(m.content or "")
                    st.markdown(display_content)
                    cols = st.columns(len(options))
                    for j, opt in enumerate(options):
                        with cols[j]:
                            if st.button(opt, key=f"risk_q{q_num}_{j}_{opt[:20]}"):
                                chosen = opt
                else:
                    st.markdown(m.content)
    return chosen


def _is_no(text: str) -> bool:
    t = (text or "").strip().lower()
    return t in {"n", "no", "nope", "cancel", "stop", "don't", "do not"}


def _is_likely_tool_output(content: str) -> bool:
    """True if content looks like a direct tool result; skip judge LLM to save latency."""
    if not (content or "").strip():
        return False
    c = content.strip()
    return (
        c.startswith("**Question ") and " of 4**" in c
        or c.startswith("Performance for ")
        or c.startswith("Your current risk category")
        or c.startswith("Based on your current portfolio allocation")
        or c.startswith("**As of ")
        or c.startswith("Latest trend summary")
        or c.startswith("Portfolio changes require")
        or c.startswith("Your inferred risk profile is:")
        or "I cannot change your portfolio" in c
        or "I cannot edit your portfolio" in c
        or "I don't have permission to edit your portfolio" in c
    )


def _dedupe_risk_question_content(content: str) -> str:
    """If the same 'Question N of 4' block appears twice, return content with one copy only."""
    if not content or "**Question " not in content or " of 4**" not in content:
        return content
    if content.count("**Question ") < 2:
        return content
    # Extract first full block (Question N of 4 ... through Choose one: ...)
    m = re.search(r"\*\*Question \d of 4\*\*.*?Choose one:.*?(?=\*\*Question \d of 4\*\*|\Z)", content, re.DOTALL)
    if m:
        return m.group(0).strip()
    return content


def _strip_choose_one_line(content: str) -> str:
    """Remove the 'Choose one: ...' line from risk question content so only the question text is shown (buttons replace it)."""
    if not content or "Choose one:" not in content:
        return content
    idx = content.find("Choose one:")
    # Remove from the newline before "Choose one:" through end of that line
    line_start = content.rfind("\n", 0, idx)
    if line_start == -1:
        line_start = 0
    else:
        line_start += 1  # keep the newline we found
    line_end = content.find("\n", idx)
    if line_end == -1:
        line_end = len(content)
    return (content[:line_start].rstrip() + "\n" + content[line_end:].lstrip()).strip()


def _parse_risk_question(last_content: str) -> Optional[Tuple[int, List[str]]]:
    """If last message is a risk question, return (question_number_1_to_4, list_of_option_labels)."""
    if not last_content or "**Question " not in last_content or " of 4**" not in last_content or "Choose one:" not in last_content:
        return None
    m = re.search(r"\*\*Question (\d) of 4\*\*", last_content)
    if not m:
        return None
    n = int(m.group(1))
    idx = last_content.find("Choose one:")
    if idx == -1:
        return None
    rest = last_content[idx + len("Choose one:"):].split("\n")[0].strip()
    options = [s.strip().strip("*").strip() for s in rest.split("|")]
    options = [o for o in options if o]
    if not options:
        return None
    return (n, options)


def main():
    st.set_page_config(page_title=PROJECT_NAME, layout="wide")
    st.title("AI Portfolio Chatbot")

    ensure_session()
    state: ChatState = st.session_state.state
    graph = st.session_state.graph
    store: SQLiteStore = st.session_state.store
    main_llm = st.session_state.main_llm
    judge_llm = st.session_state.judge_llm
    policy_llm = st.session_state.policy_llm
    # Load portfolio once per run to avoid duplicate DB reads in sidebar, graph context, and flags.
    df_all = store.load_all()

    # --- Sidebar: Portfolio data ---
    st.sidebar.header("Portfolio data")
    upload_info = st.sidebar.expander("Upload portfolio CSV", expanded=True)
    with upload_info:
        st.caption(
            "CSV columns: Date, StockValue, BondValue, ETFValue, CashValue, "
            "CashAddition, CashWithdrawal. TotalValue is computed on load."
        )
        file = st.file_uploader("Select CSV file", type=["csv"], key="portfolio_csv")

        if file is not None:
            # Detect when the user selects a new file and reset the completion flag
            file_id = (getattr(file, "name", None), getattr(file, "size", None), getattr(file, "type", None))
            prev_id = st.session_state.get("current_upload_file_id")
            if prev_id != file_id:
                st.session_state["current_upload_file_id"] = file_id
                st.session_state["upload_completed_for_current_file"] = False

            try:
                df = pd.read_csv(file)
                df2 = normalize_wide_csv(df)

                # Only show load controls while this file has not been successfully loaded
                if not st.session_state.get("upload_completed_for_current_file", False):
                    mode = st.radio(
                        "Load mode",
                        ["Merge with existing (by date)", "Replace all existing data"],
                        index=0,
                    )
                    if st.button("Confirm & Load"):
                        if mode.startswith("Replace"):
                            with store.connect() as con:
                                con.execute("DELETE FROM portfolio_daily")
                                con.commit()

                        for _, r in df2.iterrows():
                            store.upsert_day(
                                r["Date"],
                                r["StockValue"],
                                r["BondValue"],
                                r["CashValue"],
                                etf=r["ETFValue"],
                                cash_addition=r.get("CashAddition", 0),
                                cash_withdrawal=r.get("CashWithdrawal", 0),
                            )

                        state.already_suggested_upload_for_edit = False
                        df_all_after = store.load_all()
                        last_date_after = df_all_after["Date"].max()
                        last_row_after = df_all_after[df_all_after["Date"] == last_date_after].iloc[0]
                        last_total = float(last_row_after.get(
                            "TotalValue",
                            last_row_after["StockValue"] + last_row_after["BondValue"] + last_row_after["ETFValue"] + last_row_after["CashValue"],
                        ))
                        full_min = df_all_after["Date"].min()
                        full_max = df_all_after["Date"].max()
                        total_rows = len(df_all_after)
                        # Per-file stats (this upload only)
                        file_min = df2["Date"].min()
                        file_max = df2["Date"].max()
                        file_rows = len(df2)
                        total_line = f"- Final portfolio total value (per last date): **{last_total:,.2f}**\n" if last_total is not None else ""
                        msg_text = (
                            "✅ Loaded daily portfolio values.\n\n"
                            "**This upload (file-level):**\n"
                            f"- Rows loaded from this file: **{file_rows}**\n"
                            f"- Date range in this file: **{file_min} → {file_max}**\n\n"
                            "**Overall portfolio (after upload):**\n"
                            f"- Total rows: **{total_rows}**\n"
                            f"- Date range: **{full_min} → {full_max}**\n"
                            f"{total_line}\n"
                            "You can ask for performance over a date range, run the risk questionnaire, or get latest trends in the US market."
                        )
                        state.messages.append(AIMessage(content=msg_text))
                        st.session_state.sidebar_messages.append({"kind": "info", "text": "Portfolio CSV uploaded and merged successfully."})
                        st.session_state.state = state
                        # Toast-style notification instead of sidebar messages list
                        st.toast("Portfolio CSV uploaded and merged successfully.", icon="✅")
                        # Refresh df_all for snapshot and app_config
                        df_all = df_all_after
                        # Mark this file as processed so the controls disappear
                        st.session_state["upload_completed_for_current_file"] = True
            except Exception as e:
                st.error(f"Could not parse CSV: {e}")
                st.session_state.sidebar_messages.append({"kind": "error", "text": f"Could not parse CSV: {e}"})
                st.toast(f"Could not parse CSV: {e}", icon="❌")

    # --- Sidebar: Snapshot ---
    st.sidebar.header("Snapshot")
    if not df_all.empty:
        last_date = df_all["Date"].max()
        alloc = allocation_on_date(df_all, last_date)
        last_row = df_all[df_all["Date"] == last_date].iloc[0]
        total_val = float(
            last_row.get(
                "TotalValue",
                last_row["StockValue"]
                + last_row["BondValue"]
                + last_row["CashValue"]
                + last_row["ETFValue"],
            )
        )
        st.sidebar.text(f"Date: {last_date.isoformat()}")
        st.sidebar.text(f"Total value: {total_val:,.2f}")
        st.sidebar.text(f"Stock %: {round(alloc['Stock'] * 100, 2)}")
        st.sidebar.text(f"Bond %: {round(alloc['Bond'] * 100, 2)}")
        st.sidebar.text(f"ETF %: {round(alloc['ETF'] * 100, 2)}")
        st.sidebar.text(f"Cash %: {round(alloc['Cash'] * 100, 2)}")
        # Always show current risk category based on portfolio allocation (last date)
        current_risk = infer_risk_from_allocation(
            alloc["Stock"], alloc["Bond"], alloc["Cash"], alloc["ETF"]
        )
        st.sidebar.text(f"Risk profile: {current_risk}")
    else:
        st.sidebar.caption("No portfolio loaded yet. Upload a CSV to see a snapshot.")

    # --- Sidebar: Maintenance ---
    st.sidebar.header("Maintenance")
    with st.sidebar.expander("Clear portfolio data"):
        st.caption("Remove all daily portfolio values. Cannot be undone.")
        if st.button("Confirm & Clear", type="secondary"):
            store.clear_portfolio()
            state.messages.append(AIMessage(content="Portfolio data has been cleared. You can upload a new CSV to start over."))
            st.session_state.sidebar_messages.append({"kind": "warning", "text": "All portfolio data has been cleared."})
            st.session_state.state = state
            # Toast-style notification instead of sidebar messages list
            st.toast("All portfolio data has been cleared.", icon="⚠️")
            st.rerun()

    # Chat (risk question: only show question text; buttons replace "Choose one: ..." and are inside the assistant bubble)
    last_content = (state.messages[-1].content or "") if state.messages and isinstance(state.messages[-1], AIMessage) else ""
    parsed = _parse_risk_question(last_content) if last_content else None
    chosen = render_chat(state.messages, risk_buttons=parsed)

    def _app_config() -> Dict[str, Any]:
        """Config for graph: store, main_llm, judge_llm, and optional portfolio_date_range to avoid duplicate load_all()."""
        out: Dict[str, Any] = {"store": store, "main_llm": main_llm, "judge_llm": judge_llm}
        if not df_all.empty:
            out["portfolio_date_range"] = (df_all["Date"].min(), df_all["Date"].max())
        return out

    if parsed and chosen:
        q_num, _ = parsed
        state.messages.append(HumanMessage(content=chosen))
        state_dict = _state_to_dict(state)
        config = {"configurable": {"store": store, "main_llm": main_llm}}
        result_text, state_updates = run_tool(
            "submit_risk_answer",
            {"question_number": str(q_num), "answer_text": chosen},
            state_dict,
            config,
        )
        if "risk_answers" in state_updates:
            state.risk_answers = state_updates["risk_answers"]
        if "risk_profile" in state_updates:
            state.risk_profile = state_updates["risk_profile"]
        state.messages.append(AIMessage(content=result_text))
        st.session_state.state = state
        st.rerun()
        return

    # If we have a pending user message, run the AI pipeline (so user message is already visible)
    if st.session_state.get("pending_ai_response"):
        # Look at the last human message to infer what the LLM is likely doing
        last_user_text = ""
        for m in reversed(state.messages):
            if isinstance(m, HumanMessage):
                last_user_text = (m.content or "").lower()
                break

        thinking_text = "Analyzing your request and preparing a response..."
        if any(k in last_user_text for k in ["performance", "return", "drawdown"]):
            thinking_text = "Calculating portfolio performance from your stored daily values..."
        elif any(k in last_user_text for k in ["trend", "outlook", "market"]):
            thinking_text = "Fetching and summarizing the latest market trends..."
        elif any(k in last_user_text for k in ["rebalance", "allocation", "aggressive", "moderate", "conservative"]):
            thinking_text = "Computing a rebalance suggestion based on your portfolio and target risk..."
        elif "questionnaire" in last_user_text or "risk profile" in last_user_text:
            thinking_text = "Evaluating your risk profile and next questionnaire step..."

        with st.chat_message("assistant"):
            st.markdown(f"_{thinking_text}_")

        with st.spinner(""):
            flags = ExecutionFlags(
                did_persist_change=False,
                did_use_tavily=False,
                did_compute_metrics=False,
                has_portfolio_data=not df_all.empty,
                user_consented_tavily=False,
                has_data_in_requested_range=True,
                user_requested_metrics=any(k in last_user_text for k in ["performance", "return", "drawdown"]),
                user_requested_trends=any(k in last_user_text for k in ["trend", "outlook", "market"]),
            )
            state_dict = _state_to_dict(state)
            state_dict["app_config"] = _app_config()
            result = graph.invoke(state_dict)
            result.pop("app_config", None)
            state = _dict_to_state(result)

            if state.messages and isinstance(state.messages[-1], AIMessage) and state.messages[-1].content:
                state.messages[-1] = AIMessage(content=_dedupe_risk_question_content(state.messages[-1].content))
            if state.messages:
                last_content = (state.messages[-1].content or "").lower() if hasattr(state.messages[-1], "content") else ""
                if "confirmed — updated your portfolio" in last_content or "risk profile is now" in last_content:
                    flags.did_persist_change = True
                if "latest trend summary" in last_content or "via tavily" in last_content:
                    flags.did_use_tavily = True
                if "return:" in last_content or "start value" in last_content or "max drawdown" in last_content:
                    flags.did_compute_metrics = True
                if flags.user_requested_metrics and any(p in last_content for p in ["don't have", "no data", "no stored", "i don't have"]):
                    flags.has_data_in_requested_range = False
            if state.messages and isinstance(state.messages[-1], AIMessage):
                last_content = state.messages[-1].content or ""

                if _is_likely_tool_output(last_content):
                    state.last_policy = {"ok": True, "issues": []}
                else:
                    safe_text, policy_report = enforce_policy(policy_llm, last_content, flags)
                    state.last_policy = policy_report
                    if safe_text != last_content:
                        state.messages[-1] = AIMessage(content=safe_text)
            # Judge now runs as a LangGraph node, so we don't call it here
            # Dedupe risk question content if needed
            if state.messages and isinstance(state.messages[-1], AIMessage) and state.messages[-1].content:
                state.messages[-1] = AIMessage(content=_dedupe_risk_question_content(state.messages[-1].content))
        st.session_state.pending_ai_response = False
        st.session_state.state = state
        st.rerun()

    user_input = st.chat_input("Ask about latest trend in US markets or calculate performance of your portfolio.")
    if not user_input:
        return

    state.messages.append(HumanMessage(content=user_input))
    user_lower = (user_input or "").lower()
    flags = ExecutionFlags(
        did_persist_change=False,
        did_use_tavily=False,
        did_compute_metrics=False,
        has_portfolio_data=not df_all.empty,
        user_consented_tavily=False,
        has_data_in_requested_range=True,
        user_requested_metrics=any(k in user_lower for k in ["performance", "return", "drawdown"]),
        user_requested_trends=any(k in user_lower for k in ["trend", "outlook", "market"]),
    )

    if state.pending_kind != "none" and _is_no(user_input):
        state.pending_kind = "none"
        state.pending_payload = {}
        state.messages.append(AIMessage(content="Canceled. What would you like to do next?"))
        st.session_state.state = state
        st.rerun()
        return

    st.session_state.pending_ai_response = True
    st.session_state.state = state
    st.rerun()


if __name__ == "__main__":
    main()
