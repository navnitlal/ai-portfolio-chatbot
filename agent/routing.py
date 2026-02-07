"""Conditional edge routing functions for the LangGraph agent."""
from __future__ import annotations

from typing import Any, Dict

from langchain_core.messages import AIMessage

from agent.state import extract_tool_call_info, REFLECTION_THRESHOLD, MAX_REFLECTIONS
from observability import logger


def route_from_agent(state: Dict[str, Any]) -> str:
    """Route from agent to the appropriate execution node.
    - Risk tools -> risk_questionnaire (HITL subgraph)
    - Market trends -> market_analysis (error recovery subgraph)
    - Other tools -> tools (parallel execution)
    - No tools -> judge
    """
    messages = state.get("messages", [])
    last = messages[-1] if messages else None
    if not isinstance(last, AIMessage) or not getattr(last, "tool_calls", None):
        return "judge"

    tool_names = set()
    for tc in (last.tool_calls or []):
        name, _, _ = extract_tool_call_info(tc)
        tool_names.add(name)

    if tool_names & {"get_next_risk_question", "submit_risk_answer"}:
        return "risk_questionnaire"
    if "fetch_market_trends" in tool_names:
        return "market_analysis"
    return "tools"


def reflect_or_pass(state: Dict[str, Any]) -> str:
    """After judge: reflect (send feedback to agent) or pass to policy."""
    judge = state.get("last_judge", {})
    score = judge.get("score", 10)
    reflection_count = state.get("reflection_count", 0)
    if score < REFLECTION_THRESHOLD and reflection_count < MAX_REFLECTIONS:
        logger.info("Judge score %d < %d, reflecting (cycle %d)", score, REFLECTION_THRESHOLD, reflection_count + 1)
        return "reflect"
    return "policy"
