"""LangGraph: agent with declared tools. The LLM chooses which tool to call (no keyword branching)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any, Dict, List, Optional, Literal, TypedDict

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import tools_condition
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage, SystemMessage

from prompts import SYSTEM_PROMPT
from tools import get_binding_tools, run_tool
from judge import judge_and_rewrite

# Cap messages sent to the LLM so long chats don't blow token usage and latency.
MAX_MESSAGES_IN_CONTEXT = 24

PendingKind = Literal["none", "apply_edit", "apply_risk_change"]


class GraphState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    risk_answers: Dict[str, str]
    risk_profile: Optional[str]
    pending_kind: str
    pending_payload: Dict[str, Any]
    last_judge: Dict[str, Any]
    last_policy: Dict[str, Any]
    already_suggested_upload_for_edit: bool
    app_config: Dict[str, Any]  # store, main_llm — configurable is reserved by LangGraph


@dataclass
class ChatState:
    messages: List[BaseMessage] = field(default_factory=list)
    risk_answers: Dict[str, str] = field(default_factory=dict)
    risk_profile: Optional[str] = None
    pending_kind: PendingKind = "none"
    pending_payload: Dict[str, Any] = field(default_factory=dict)
    last_judge: Dict[str, Any] = field(default_factory=dict)
    last_policy: Dict[str, Any] = field(default_factory=dict)
    already_suggested_upload_for_edit: bool = False


def _state_to_dict(state: ChatState) -> Dict[str, Any]:
    return {
        "messages": state.messages,
        "risk_answers": state.risk_answers,
        "risk_profile": state.risk_profile,
        "pending_kind": state.pending_kind,
        "pending_payload": state.pending_payload,
        "last_judge": state.last_judge,
        "last_policy": state.last_policy,
        "already_suggested_upload_for_edit": state.already_suggested_upload_for_edit,
    }


def _dict_to_state(d: Dict[str, Any]) -> ChatState:
    return ChatState(
        messages=d.get("messages", []),
        risk_answers=d.get("risk_answers", {}),
        risk_profile=d.get("risk_profile"),
        pending_kind=d.get("pending_kind", "none"),
        pending_payload=d.get("pending_payload", {}),
        last_judge=d.get("last_judge", {}),
        last_policy=d.get("last_policy", {}),
        already_suggested_upload_for_edit=d.get("already_suggested_upload_for_edit", False),
    )


def _data_availability_context(state: Dict[str, Any]) -> str:
    """If portfolio data is in SQLite, return a context snippet to append to the system prompt. Uses app_config.portfolio_date_range when provided to avoid duplicate load_all()."""
    config = state.get("app_config") or {}
    date_range = config.get("portfolio_date_range")
    if date_range is not None:
        min_d, max_d = date_range
        return (
            f"\n\n[Context: The user has portfolio data loaded in SQLite from {min_d} to {max_d}. "
            "When they ask for return or performance over a period, call get_performance with start_date and end_date. "
            "Do NOT say you don't have their data or ask them to upload a CSV or provide start/end values—use the stored data.]"
        )
    store = config.get("store")
    if not store:
        return ""
    try:
        df = store.load_all()
        if df.empty:
            # Explicitly tell the LLM that no portfolio data is available yet, so for
            # rebalance / performance / risk questions it must ask the user to upload a CSV
            # instead of asking follow-up questions like "which risk category".
            return (
                "\n\n[Context: There is currently NO portfolio data loaded in SQLite. "
                "If the user asks for a rebalance, performance, or current risk profile, "
                "respond with: 'No portfolio data loaded. Use the sidebar to upload a CSV file with your daily portfolio values.' "
                "Do NOT ask which risk category they want, do NOT start or continue the risk questionnaire, "
                "and do NOT call tools that require portfolio data until a CSV has been uploaded.]"
            )
        min_d = df["Date"].min()
        max_d = df["Date"].max()
        return (
            f"\n\n[Context: The user has portfolio data loaded in SQLite from {min_d} to {max_d}. "
            "When they ask for return or performance over a period, call get_performance with start_date and end_date. "
            "Do NOT say you don't have their data or ask them to upload a CSV or provide start/end values—use the stored data.]"
        )
    except Exception:
        return ""


def create_agent_node(model_with_tools):
    """Create the agent node that calls the LLM with tools."""

    def agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
        messages = state.get("messages", [])
        
        # Keep only the last N messages, but ensure we don't leave orphaned ToolMessages
        # (ToolMessages must follow their corresponding AIMessage with tool_calls)
        if len(messages) > MAX_MESSAGES_IN_CONTEXT:
            truncated = list(messages)[-MAX_MESSAGES_IN_CONTEXT:]
            # If first message is a ToolMessage, we need to find its corresponding AIMessage
            # and include it, or skip the ToolMessage
            if truncated and isinstance(truncated[0], ToolMessage):
                # Look backwards in original messages to find the AIMessage with matching tool_calls
                tool_call_id = getattr(truncated[0], "tool_call_id", None) or getattr(truncated[0], "id", None)
                found_ai = False
                for i in range(len(messages) - MAX_MESSAGES_IN_CONTEXT - 1, -1, -1):
                    msg = messages[i]
                    if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
                        # Check if this AIMessage has the matching tool_call_id
                        for tc in (msg.tool_calls or []):
                            tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                            if tc_id == tool_call_id:
                                # Include this AIMessage before the ToolMessage
                                truncated = [msg] + truncated
                                found_ai = True
                                break
                        if found_ai:
                            break
                # If we couldn't find the matching AIMessage, skip the orphaned ToolMessage
                if not found_ai:
                    truncated = truncated[1:]
            messages = truncated

        system_content = SYSTEM_PROMPT + _data_availability_context(state)
        # Prepend system prompt once if not at start
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_content)] + list(messages)
        else:
            # Refresh system message with current data-availability context
            messages = [SystemMessage(content=system_content)] + list(messages)[1:]
        
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}

    return agent_node


def create_tools_node():
    """Create the tools node that executes tool calls and merges state updates."""

    def tools_node(state: Dict[str, Any]) -> Dict[str, Any]:
        messages = list(state.get("messages", []))
        # Store/main_llm passed via app_config (configurable is reserved by LangGraph)
        app_config = state.get("app_config") or {}
        config = {"configurable": app_config}
        state_dict = {k: v for k, v in state.items() if k != "app_config"}

        last_msg = messages[-1] if messages else None
        if not isinstance(last_msg, AIMessage) or not getattr(last_msg, "tool_calls", None):
            return state

        tool_calls = last_msg.tool_calls or []
        tool_messages = []
        updates = {}

        risk_answers = state_dict.get("risk_answers") or {}

        for tc in tool_calls:
            if not isinstance(tc, dict):
                tc = {"id": getattr(tc, "id", ""), "name": getattr(tc, "name", ""), "args": getattr(tc, "args", {}) or {}}
            name = tc.get("name") or (tc.get("args") or {}).get("__tool_name__")
            args = tc.get("args") or {}
            if isinstance(args, dict) and "__tool_name__" in args:
                args = {k: v for k, v in args.items() if k != "__tool_name__"}
            tool_call_id = tc.get("id", "")

            # Never run submit_risk_answer when no answers yet—avoids skipping question 1 when LLM calls both get_next_risk_question and submit_risk_answer on "run questionnaire".
            if name == "submit_risk_answer" and not risk_answers:
                tool_messages.append(
                    ToolMessage(content="[No answer recorded yet; show the first question from get_next_risk_question above.]", tool_call_id=tool_call_id, name=name)
                )
                continue

            result_text, state_updates = run_tool(name, args, state_dict, config)
            for k, v in state_updates.items():
                updates[k] = v
            state_dict.update(updates)
            if "risk_answers" in state_updates:
                risk_answers = state_dict.get("risk_answers") or {}

            tool_messages.append(
                ToolMessage(content=result_text, tool_call_id=tool_call_id, name=name)
            )

        # Append tool messages after the assistant message that requested them,
        # so that the next LLM call sees a valid assistant→tool sequence.
        if not tool_messages:
            return {**state_dict, **updates}

        out = {
            "messages": messages + tool_messages,
            **updates,
        }
        return out

    return tools_node


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


def create_judge_node():
    """Create the judge node that scores and optionally rewrites the final assistant message."""

    def judge_node(state: Dict[str, Any]) -> Dict[str, Any]:
        messages = list(state.get("messages", []))
        app_config = state.get("app_config") or {}
        judge_llm = app_config.get("judge_llm")
        main_llm = app_config.get("main_llm")

        if not judge_llm or not main_llm:
            # If judge_llm not provided, skip judging (shouldn't happen in normal flow)
            return {"last_judge": {"score": 10, "issues": [], "rewrite_needed": False}}

        last_msg = messages[-1] if messages else None
        if not isinstance(last_msg, AIMessage):
            return {"last_judge": {"score": 10, "issues": [], "rewrite_needed": False}}

        last_content = last_msg.content or ""
        
        # Skip judge for tool-like outputs to save latency
        if _is_likely_tool_output(last_content):
            return {"last_judge": {"score": 10, "issues": [], "rewrite_needed": False}}

        improved, verdict = judge_and_rewrite(judge_llm, main_llm, last_content)
        
        updates = {"last_judge": verdict}
        
        # If rewritten, update the last message
        if improved != last_content:
            updated_messages = messages[:-1] + [AIMessage(content=improved)]
            updates["messages"] = updated_messages

        return updates

    return judge_node


def build_graph(main_llm):
    """Build the agent graph: agent (LLM with tools) <-> tools node until no more tool_calls, then judge node."""
    binding_tools = get_binding_tools()
    model_with_tools = main_llm.bind_tools(binding_tools)

    graph = StateGraph(GraphState)

    graph.add_node("agent", create_agent_node(model_with_tools))
    graph.add_node("tools", create_tools_node())
    graph.add_node("judge", create_judge_node())

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", "__end__": "judge"})
    graph.add_edge("tools", "agent")
    graph.add_edge("judge", END)

    return graph.compile()
