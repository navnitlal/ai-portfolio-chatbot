"""Graph builder: assembles all nodes and edges into the compiled LangGraph."""
from __future__ import annotations

from langgraph.graph import StateGraph, END

from agent.state import GraphState, MemorySaver
from agent.nodes import (
    create_planner_node,
    create_agent_node,
    create_tools_node,
    create_risk_questionnaire_node,
    create_market_analysis_node,
    create_judge_node,
    create_reflect_node,
    create_policy_node,
)
from agent.routing import route_from_agent, reflect_or_pass
from agent.tools import get_binding_tools


def build_graph(main_llm, checkpointer=None):
    """Build the full agentic graph with all 9 features.

    Graph topology:
        planner -> agent -> [route_from_agent]
                              -> "tools"              (parallel + error recovery) -> agent
                              -> "risk_questionnaire" (HITL with interrupt)       -> agent
                              -> "market_analysis"    (error recovery + retry)    -> agent
                              -> "judge"              (no tools needed)
        judge -> [reflect_or_pass]
                   -> "reflect" -> agent  (self-correction, max 1 cycle)
                   -> "policy"  -> END

    Args:
        main_llm: ChatOpenAI instance for the agent
        checkpointer: LangGraph checkpointer (e.g. MemorySaver) for persistence + HITL.
                       If None and MemorySaver is available, one is created automatically.
    """
    binding_tools = get_binding_tools()
    model_with_tools = main_llm.bind_tools(binding_tools).with_retry(stop_after_attempt=3)

    graph = StateGraph(GraphState)

    # --- Nodes ---
    graph.add_node("planner", create_planner_node())
    graph.add_node("agent", create_agent_node(model_with_tools))
    graph.add_node("tools", create_tools_node())
    graph.add_node("risk_questionnaire", create_risk_questionnaire_node())
    graph.add_node("market_analysis", create_market_analysis_node())
    graph.add_node("judge", create_judge_node())
    graph.add_node("reflect", create_reflect_node())
    graph.add_node("policy", create_policy_node())

    # --- Edges ---
    graph.set_entry_point("planner")
    graph.add_edge("planner", "agent")

    graph.add_conditional_edges("agent", route_from_agent, {
        "tools": "tools",
        "risk_questionnaire": "risk_questionnaire",
        "market_analysis": "market_analysis",
        "judge": "judge",
    })

    # All tool execution nodes loop back to agent
    graph.add_edge("tools", "agent")
    graph.add_edge("risk_questionnaire", "agent")
    graph.add_edge("market_analysis", "agent")

    # Reflection loop: judge -> reflect -> agent (max 1 cycle)
    graph.add_conditional_edges("judge", reflect_or_pass, {
        "reflect": "reflect",
        "policy": "policy",
    })
    graph.add_edge("reflect", "agent")

    graph.add_edge("policy", END)

    return graph.compile(checkpointer=checkpointer)
