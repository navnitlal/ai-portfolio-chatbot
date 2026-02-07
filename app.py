from __future__ import annotations

import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

import pandas as pd
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

from core.analytics import normalize_wide_csv, allocation_on_date
from core.storage import SQLiteStore
from core.risk import infer_risk_from_allocation
from agent import ChatState, build_graph, state_to_dict, dict_to_state
from agent.state import MemorySaver, Command
from agent.tools import run_tool
from observability import setup_logging, setup_langsmith, logger

PROJECT_NAME = "ai-portfolio-chatbot"
GREETING = "Hi, I am your personal portfolio assistant. What can I help you with today?"
DB_PATH = str(Path(__file__).parent / "portfolio.db")

# Node display names for streaming progress
_NODE_LABELS = {
    "planner": "Planning approach...",
    "agent": "Generating response...",
    "tools": "Executing tools...",
    "risk_questionnaire": "Risk questionnaire...",
    "market_analysis": "Analyzing market trends...",
    "judge": "Quality review...",
    "reflect": "Refining response...",
    "policy": "Policy check...",
}


# ---------------------------------------------------------------------------
# Session & LLM setup
# ---------------------------------------------------------------------------

def get_llms():
    main_llm = ChatOpenAI(
        model=os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini"),
        temperature=0.3,
    )
    judge_llm = ChatOpenAI(
        model=os.getenv("OPENAI_JUDGE_MODEL", "gpt-4o-mini"),
        temperature=0,
    )
    policy_llm = ChatOpenAI(
        model=os.getenv("OPENAI_POLICY_MODEL", "gpt-4o-mini"),
        temperature=0,
    )
    return main_llm, judge_llm, policy_llm


def ensure_session():
    # Observability — once per process
    setup_logging(os.getenv("LOG_LEVEL", "INFO"))
    setup_langsmith()

    if "main_llm" not in st.session_state:
        st.session_state.main_llm, st.session_state.judge_llm, st.session_state.policy_llm = get_llms()
    if "store" not in st.session_state:
        store = SQLiteStore(DB_PATH)
        store.init()
        st.session_state.store = store
    if "portfolio_cleared_on_start" not in st.session_state:
        st.session_state.store.clear_portfolio()
        st.session_state.portfolio_cleared_on_start = True

    # Checkpointer for conversation persistence + HITL
    if "checkpointer" not in st.session_state:
        st.session_state.checkpointer = MemorySaver() if MemorySaver is not None else None

    # Unique thread ID per session (required for checkpointing)
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())

    if "graph" not in st.session_state:
        st.session_state.graph = build_graph(
            st.session_state.main_llm,
            checkpointer=st.session_state.checkpointer,
        )
    if "state" not in st.session_state:
        st.session_state.state = ChatState(messages=[AIMessage(content=GREETING)])
    if "risk_loaded" not in st.session_state:
        rp = st.session_state.store.get_risk_profile()
        if rp:
            st.session_state.state.risk_profile = rp
        st.session_state.risk_loaded = True
    if "sidebar_messages" not in st.session_state:
        st.session_state.sidebar_messages = []


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _runnable_config(store, main_llm, judge_llm, policy_llm, df_all) -> Dict[str, Any]:
    """RunnableConfig with thread_id and dependencies in configurable."""
    cfg: Dict[str, Any] = {
        "thread_id": st.session_state.thread_id,
        "store": store,
        "main_llm": main_llm,
        "judge_llm": judge_llm,
        "policy_llm": policy_llm,
    }
    if not df_all.empty:
        cfg["portfolio_date_range"] = (df_all["Date"].min(), df_all["Date"].max())
    return {"configurable": cfg}


# ---------------------------------------------------------------------------
# Chat rendering
# ---------------------------------------------------------------------------

def render_chat(messages, risk_buttons=None):
    """Render chat messages. ToolMessages and internal SystemMessages are skipped.
    If risk_buttons is (q_num, options), the last AI message shows option buttons."""
    chosen: Optional[str] = None
    for i, m in enumerate(messages):
        # Skip internal messages
        if isinstance(m, ToolMessage):
            continue
        if isinstance(m, SystemMessage):
            continue
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None) and not (getattr(m, "content", None) or "").strip():
            continue
        # Skip AIMessage that will be superseded by a reflection revision
        if isinstance(m, AIMessage) and i + 1 < len(messages):
            next_msg = messages[i + 1]
            if isinstance(next_msg, SystemMessage) and "[Reflection" in (next_msg.content or ""):
                continue

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


# ---------------------------------------------------------------------------
# Streaming graph invocation
# ---------------------------------------------------------------------------

def _stream_graph(graph, input_data, config, state: ChatState):
    """Stream the graph response, show progress, and return the updated state.
    Uses stream_mode='updates' for node-by-node progress tracking."""

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("_Thinking..._")

        with st.status("Processing...", expanded=False) as status:
            final_content = ""
            node_count = 0

            try:
                for chunk in graph.stream(input_data, config=config, stream_mode="updates"):
                    for node_name, updates in chunk.items():
                        node_count += 1
                        label = _NODE_LABELS.get(node_name, f"Step: {node_name}")
                        status.update(label=label)

                        # Track the latest AI message content for live display
                        for msg in updates.get("messages", []):
                            if isinstance(msg, AIMessage) and msg.content:
                                final_content = msg.content
                                message_placeholder.markdown(final_content)

                status.update(label=f"Done ({node_count} steps)", state="complete", expanded=False)

            except Exception as e:
                logger.error("Graph streaming failed: %s", e)
                final_content = f"Sorry, something went wrong. Please try again. (Error: {e})"
                status.update(label="Error", state="error", expanded=False)

        if final_content:
            message_placeholder.markdown(final_content)
        else:
            message_placeholder.empty()

    # Reconstruct state from graph output
    # With checkpointer: read from graph state; without: reconstruct from last stream
    has_checkpointer = st.session_state.checkpointer is not None
    if has_checkpointer:
        try:
            snapshot = graph.get_state(config)
            if snapshot and snapshot.values:
                return dict_to_state(snapshot.values), snapshot
        except Exception:
            pass

    # Fallback: use the current state with the final message appended
    if final_content:
        state.messages.append(AIMessage(content=final_content))
    return state, None


# ---------------------------------------------------------------------------
# HITL interrupt handling
# ---------------------------------------------------------------------------

def _check_and_handle_interrupt(graph, config, state: ChatState):
    """Check if the graph is paused at an interrupt (e.g., risk question).
    If so, display the question and set up resume flow. Returns True if interrupt is active."""
    if st.session_state.checkpointer is None or Command is None:
        return False

    try:
        snapshot = graph.get_state(config)
    except Exception:
        return False

    if not snapshot or not snapshot.next:
        return False

    # Extract interrupt data from pending tasks
    for task in getattr(snapshot, "tasks", []):
        for interrupt_info in getattr(task, "interrupts", []):
            data = interrupt_info.value
            if isinstance(data, dict) and data.get("type") == "risk_question":
                q_num = data.get("question_number", 1)
                q_text = data.get("question_text", "")
                options = data.get("options", [])

                # Display the question with buttons
                with st.chat_message("assistant"):
                    display_text = _strip_choose_one_line(q_text)
                    st.markdown(display_text)
                    if options:
                        cols = st.columns(len(options))
                        for j, opt in enumerate(options):
                            with cols[j]:
                                if st.button(opt, key=f"hitl_q{q_num}_{j}_{opt[:20]}"):
                                    st.session_state.resume_answer = opt
                                    st.rerun()
                return True

    return False


def _resume_from_interrupt(graph, config, state: ChatState):
    """Resume the graph from a HITL interrupt with the user's answer."""
    answer = st.session_state.pop("resume_answer", None)
    if not answer or Command is None:
        return state

    logger.info("Resuming HITL interrupt with answer: %s", answer)
    try:
        # Resume the interrupted graph
        result_events = list(graph.stream(
            Command(resume=answer), config=config, stream_mode="updates"
        ))

        # Read updated state from checkpointer
        snapshot = graph.get_state(config)
        if snapshot and snapshot.values:
            return dict_to_state(snapshot.values)
    except Exception as e:
        logger.error("HITL resume failed: %s", e)
        state.messages.append(AIMessage(content=f"Error resuming: {e}"))

    return state


# ---------------------------------------------------------------------------
# UI helpers (risk question parsing, etc.)
# ---------------------------------------------------------------------------

def _is_no(text: str) -> bool:
    t = (text or "").strip().lower()
    return t in {"n", "no", "nope", "cancel", "stop", "don't", "do not"}


def _dedupe_risk_question_content(content: str) -> str:
    if not content or "**Question " not in content or " of 4**" not in content:
        return content
    if content.count("**Question ") < 2:
        return content
    m = re.search(r"\*\*Question \d of 4\*\*.*?Choose one:.*?(?=\*\*Question \d of 4\*\*|\Z)", content, re.DOTALL)
    if m:
        return m.group(0).strip()
    return content


def _strip_choose_one_line(content: str) -> str:
    if not content or "Choose one:" not in content:
        return content
    idx = content.find("Choose one:")
    line_start = content.rfind("\n", 0, idx)
    if line_start == -1:
        line_start = 0
    else:
        line_start += 1
    line_end = content.find("\n", idx)
    if line_end == -1:
        line_end = len(content)
    return (content[:line_start].rstrip() + "\n" + content[line_end:].lstrip()).strip()


def _parse_risk_question(last_content: str) -> Optional[Tuple[int, List[str]]]:
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
    df_all = store.load_all()

    config = _runnable_config(store, main_llm, judge_llm, policy_llm, df_all)

    # --- Handle HITL resume (risk questionnaire answer from previous rerun) ---
    if st.session_state.get("resume_answer"):
        state = _resume_from_interrupt(graph, config, state)
        st.session_state.state = state
        # Don't return — continue rendering the updated state

    # --- Sidebar: Upload portfolio CSV ---
    upload_info = st.sidebar.expander("Upload portfolio CSV", expanded=False)
    with upload_info:
        st.caption(
            "CSV columns: Date, StockValue, BondValue, ETFValue, CashValue, "
            "CashAddition, CashWithdrawal. TotalValue is computed on load."
        )
        file = st.file_uploader("Select CSV file", type=["csv"], key="portfolio_csv")

        if file is not None:
            file_id = (getattr(file, "name", None), getattr(file, "size", None), getattr(file, "type", None))
            prev_id = st.session_state.get("current_upload_file_id")
            if prev_id != file_id:
                st.session_state["current_upload_file_id"] = file_id
                st.session_state["upload_completed_for_current_file"] = False

            try:
                df = pd.read_csv(file)
                df2 = normalize_wide_csv(df)

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
                                r["Date"], r["StockValue"], r["BondValue"], r["CashValue"],
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
                        full_min, full_max = df_all_after["Date"].min(), df_all_after["Date"].max()
                        total_rows = len(df_all_after)
                        file_min, file_max = df2["Date"].min(), df2["Date"].max()
                        file_rows = len(df2)
                        total_line = f"- Final portfolio total value (per last date): **{last_total:,.2f}**\n" if last_total is not None else ""
                        msg_text = (
                            "Loaded daily portfolio values.\n\n"
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
                        st.session_state.state = state
                        st.toast("Portfolio CSV uploaded and merged successfully.", icon="✅")
                        df_all = df_all_after
                        st.session_state["upload_completed_for_current_file"] = True
            except Exception as e:
                st.error(f"Could not parse CSV: {e}")
                st.toast(f"Could not parse CSV: {e}", icon="❌")

    # --- Sidebar: Snapshot ---
    with st.sidebar.expander("Snapshot", expanded=False):
        if not df_all.empty:
            last_date = df_all["Date"].max()
            alloc = allocation_on_date(df_all, last_date)
            last_row = df_all[df_all["Date"] == last_date].iloc[0]
            total_val = float(last_row.get(
                "TotalValue",
                last_row["StockValue"] + last_row["BondValue"] + last_row["CashValue"] + last_row["ETFValue"],
            ))
            st.text(f"Date: {last_date.isoformat()}")
            st.text(f"Total value: {total_val:,.2f}")
            st.text(f"Stock %: {round(alloc['Stock'] * 100, 2)}")
            st.text(f"Bond %: {round(alloc['Bond'] * 100, 2)}")
            st.text(f"ETF %: {round(alloc['ETF'] * 100, 2)}")
            st.text(f"Cash %: {round(alloc['Cash'] * 100, 2)}")
            current_risk = infer_risk_from_allocation(alloc["Stock"], alloc["Bond"], alloc["Cash"], alloc["ETF"])
            st.text(f"Risk profile: {current_risk}")
        else:
            st.caption("No portfolio loaded yet. Upload a CSV to see a snapshot.")

    # --- Sidebar: Clear portfolio data ---
    if st.sidebar.button("Clear portfolio data", type="secondary"):
        store.clear_portfolio()
        state.messages.append(AIMessage(content="Portfolio data has been cleared. You can upload a new CSV to start over."))
        st.session_state.state = state
        st.toast("All portfolio data has been cleared.", icon="⚠️")
        st.rerun()

    # --- Check for active HITL interrupt (risk question waiting for answer) ---
    if _check_and_handle_interrupt(graph, config, state):
        return  # Interrupt is active — buttons are shown, waiting for user

    # --- Render chat ---
    last_content = (state.messages[-1].content or "") if state.messages and isinstance(state.messages[-1], AIMessage) else ""
    parsed = _parse_risk_question(last_content) if last_content else None
    chosen = render_chat(state.messages, risk_buttons=parsed)

    # --- Risk button click (fallback for when HITL interrupt is not available) ---
    if parsed and chosen:
        q_num, _ = parsed
        state.messages.append(HumanMessage(content=chosen))
        state_dict = state_to_dict(state)
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

    # --- Pending AI response: stream the graph ---
    if st.session_state.get("pending_ai_response"):
        state_dict = state_to_dict(state)
        state, snapshot = _stream_graph(graph, state_dict, config, state)

        # Deduplicate risk question content
        if state.messages and isinstance(state.messages[-1], AIMessage) and state.messages[-1].content:
            state.messages[-1] = AIMessage(content=_dedupe_risk_question_content(state.messages[-1].content))

        st.session_state.pending_ai_response = False
        st.session_state.state = state
        st.rerun()

    # --- Chat input ---
    user_input = st.chat_input("Ask about latest trend in US markets or calculate performance of your portfolio.")
    if not user_input:
        return

    state.messages.append(HumanMessage(content=user_input))

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
