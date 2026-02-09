"""Tests for agent/nodes.py - graph node functions."""
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent.nodes import (
    create_agent_node,
    create_judge_node,
    create_planner_node,
    create_policy_node,
    create_reflect_node,
    create_tools_node,
)


class TestPlannerNode:
    """Tests for planner node."""

    def test_no_messages_returns_empty_plan(self):
        """Test that no messages returns empty plan."""
        planner = create_planner_node()
        state = {"messages": []}
        config = {"configurable": {}}

        result = planner(state, config)

        assert result["plan"] == ""
        assert result["reflection_count"] == 0

    def test_non_human_last_message_returns_empty_plan(self):
        """Test that non-human last message returns empty plan."""
        planner = create_planner_node()
        state = {"messages": [AIMessage(content="Hello")]}
        config = {"configurable": {}}

        result = planner(state, config)

        assert result["plan"] == ""

    def test_rule_based_plan_performance(self):
        """Test rule-based plan for performance keyword."""
        planner = create_planner_node()
        state = {"messages": [HumanMessage(content="Show me my performance")]}
        config = {"configurable": {}}

        result = planner(state, config)

        assert "get_performance" in result["plan"]

    def test_rule_based_plan_risk(self):
        """Test rule-based plan for risk keyword."""
        planner = create_planner_node()
        state = {"messages": [HumanMessage(content="What is my risk profile?")]}
        config = {"configurable": {}}

        result = planner(state, config)

        assert "get_current_risk" in result["plan"]

    def test_rule_based_plan_trends(self):
        """Test rule-based plan for trends keyword."""
        planner = create_planner_node()
        state = {"messages": [HumanMessage(content="Show me market trends")]}
        config = {"configurable": {}}

        result = planner(state, config)

        assert "fetch_market_trends" in result["plan"]

    def test_rule_based_plan_rebalance(self):
        """Test rule-based plan for rebalance keyword."""
        planner = create_planner_node()
        state = {"messages": [HumanMessage(content="Suggest a rebalance")]}
        config = {"configurable": {}}

        result = planner(state, config)

        assert "suggest_rebalance" in result["plan"]

    def test_llm_plan_for_complex_query(self, mock_llm_with_responses):
        """Test LLM-based plan for complex query."""
        llm = mock_llm_with_responses(["Call get_performance then suggest_rebalance."])
        planner = create_planner_node()
        state = {"messages": [HumanMessage(content="Analyze my portfolio and optimize it")]}
        config = {"configurable": {"main_llm": llm}}

        result = planner(state, config)

        assert result["plan"] != ""
        assert llm.call_count == 1


class TestAgentNode:
    """Tests for agent node."""

    def test_agent_invokes_model(self, mock_llm):
        """Test that agent invokes the model."""
        agent = create_agent_node(mock_llm)
        state = {
            "messages": [HumanMessage(content="Hello")],
            "plan": "",
        }
        config = {"configurable": {}}

        result = agent(state, config)

        assert "messages" in result
        assert len(result["messages"]) == 1
        assert mock_llm.call_count == 1

    def test_agent_includes_system_prompt(self, mock_llm):
        """Test that agent includes system prompt."""
        agent = create_agent_node(mock_llm)
        state = {
            "messages": [HumanMessage(content="Hello")],
            "plan": "",
        }
        config = {"configurable": {}}

        agent(state, config)

        # Check that system message was added
        messages = mock_llm.last_messages
        assert any(isinstance(m, SystemMessage) for m in messages)

    def test_agent_includes_plan_in_system(self, mock_llm):
        """Test that agent includes plan in system prompt."""
        agent = create_agent_node(mock_llm)
        state = {
            "messages": [HumanMessage(content="Hello")],
            "plan": "Call get_performance",
        }
        config = {"configurable": {}}

        agent(state, config)

        messages = mock_llm.last_messages
        system_msg = next(m for m in messages if isinstance(m, SystemMessage))
        assert "get_performance" in system_msg.content


class TestToolsNode:
    """Tests for tools node."""

    def test_no_tool_calls_returns_empty(self):
        """Test that no tool calls returns empty dict."""
        tools_node = create_tools_node()
        state = {"messages": [AIMessage(content="Hello")]}
        config = {"configurable": {}}

        result = tools_node(state, config)

        assert result == {}

    def test_executes_tool_and_returns_message(self, agent_config):
        """Test that tool is executed and result returned."""
        tools_node = create_tools_node()
        ai_msg = MagicMock(spec=AIMessage)
        ai_msg.tool_calls = [
            {"name": "get_capabilities", "args": {}, "id": "test-id"}
        ]
        state = {"messages": [ai_msg]}

        result = tools_node(state, agent_config)

        assert "messages" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], ToolMessage)

    def test_handles_tool_error(self, agent_config):
        """Test that tool errors are handled gracefully."""
        tools_node = create_tools_node()
        ai_msg = MagicMock(spec=AIMessage)
        ai_msg.tool_calls = [
            {"name": "unknown_tool", "args": {}, "id": "test-id"}
        ]
        state = {"messages": [ai_msg]}

        result = tools_node(state, agent_config)

        assert "messages" in result
        # Should contain error message
        assert "Unknown tool" in result["messages"][0].content


class TestJudgeNode:
    """Tests for judge node."""

    def test_no_judge_llm_returns_high_score(self):
        """Test that missing judge LLM returns high score."""
        judge_node = create_judge_node()
        state = {"messages": [AIMessage(content="Hello")]}
        config = {"configurable": {}}

        result = judge_node(state, config)

        assert result["last_judge"]["score"] == 10
        assert result["last_judge"]["rewrite_needed"] is False

    def test_non_ai_message_returns_high_score(self, agent_config):
        """Test that non-AI last message returns high score."""
        judge_node = create_judge_node()
        state = {"messages": [HumanMessage(content="Hello")]}

        result = judge_node(state, agent_config)

        assert result["last_judge"]["score"] == 10

    def test_tool_output_skipped(self, agent_config):
        """Test that tool output is skipped (no judging)."""
        judge_node = create_judge_node()
        state = {"messages": [AIMessage(content="**Question 1 of 4**\nWhat is your time horizon?")]}

        result = judge_node(state, agent_config)

        assert result["last_judge"]["score"] == 10


class TestReflectNode:
    """Tests for reflect node."""

    def test_creates_feedback_message(self):
        """Test that reflect node creates feedback message."""
        reflect_node = create_reflect_node()
        state = {
            "messages": [AIMessage(content="Original response")],
            "last_judge": {"issues": ["Issue 1", "Issue 2"]},
            "reflection_count": 0,
        }
        config = {"configurable": {}}

        result = reflect_node(state, config)

        assert "messages" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], SystemMessage)
        assert "Issue 1" in result["messages"][0].content
        assert "Issue 2" in result["messages"][0].content

    def test_increments_reflection_count(self):
        """Test that reflection count is incremented."""
        reflect_node = create_reflect_node()
        state = {
            "messages": [AIMessage(content="Original")],
            "last_judge": {"issues": ["Issue"]},
            "reflection_count": 0,
        }
        config = {"configurable": {}}

        result = reflect_node(state, config)

        assert result["reflection_count"] == 1


class TestPolicyNode:
    """Tests for policy node."""

    def test_no_policy_llm_returns_ok(self):
        """Test that missing policy LLM returns ok."""
        policy_node = create_policy_node()
        state = {"messages": [AIMessage(content="Hello")]}
        config = {"configurable": {}}

        result = policy_node(state, config)

        assert result["last_policy"]["ok"] is True

    def test_non_ai_message_returns_ok(self, agent_config):
        """Test that non-AI last message returns ok."""
        policy_node = create_policy_node()
        state = {"messages": [HumanMessage(content="Hello")]}

        result = policy_node(state, agent_config)

        assert result["last_policy"]["ok"] is True

    def test_tool_output_skipped(self, agent_config):
        """Test that tool output is skipped."""
        policy_node = create_policy_node()
        state = {"messages": [AIMessage(content="Performance for **Total portfolio**")]}

        result = policy_node(state, agent_config)

        assert result["last_policy"]["ok"] is True

    def test_resets_execution_flags(self, agent_config):
        """Test that execution flags are reset."""
        policy_node = create_policy_node()
        state = {
            "messages": [AIMessage(content="Some response")],
            "did_persist_change": True,
            "did_use_tavily": True,
            "did_compute_metrics": True,
        }

        result = policy_node(state, agent_config)

        assert result["did_persist_change"] is False
        assert result["did_use_tavily"] is False
        assert result["did_compute_metrics"] is False
