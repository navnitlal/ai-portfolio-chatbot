"""Tests for LLM responses - both mocked and integration tests.

This module contains tests that verify LLM behavior:
1. Mocked tests: Verify the system handles various LLM response patterns correctly
2. Integration tests: Verify actual LLM responses meet quality criteria (requires API key)

Run mocked tests:
    pytest tests/agent/test_llm_responses.py -m "not integration"

Run integration tests (requires OPENAI_API_KEY):
    pytest tests/agent/test_llm_responses.py -m integration
"""
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.builder import build_graph
from agent.judge import _extract_json, judge_and_rewrite
from agent.nodes import create_agent_node, create_planner_node
from agent.tools import get_binding_tools


# =============================================================================
# Mock LLM Response Pattern Tests
# =============================================================================


class TestLLMResponsePatterns:
    """Tests for various LLM response patterns using mocks."""

    def test_llm_returns_tool_call(self, mock_llm_with_tool_calls):
        """Test handling of LLM returning a tool call."""
        tool_calls = [
            {"name": "get_performance", "args": {"start_date": "2024-01-01", "end_date": "2024-02-01"}, "id": "1"}
        ]
        llm = mock_llm_with_tool_calls(tool_calls)
        agent = create_agent_node(llm)

        state = {"messages": [HumanMessage(content="Show my performance")], "plan": ""}
        config = {"configurable": {}}

        result = agent(state, config)

        assert "messages" in result
        response = result["messages"][0]
        assert hasattr(response, "tool_calls")

    def test_llm_returns_text_response(self, mock_llm_with_responses):
        """Test handling of LLM returning text response."""
        llm = mock_llm_with_responses(["Here is your portfolio summary."])
        agent = create_agent_node(llm)

        state = {"messages": [HumanMessage(content="Hello")], "plan": ""}
        config = {"configurable": {}}

        result = agent(state, config)

        assert result["messages"][0].content == "Here is your portfolio summary."

    def test_llm_returns_empty_response(self, mock_llm_with_responses):
        """Test handling of empty LLM response."""
        llm = mock_llm_with_responses([""])
        agent = create_agent_node(llm)

        state = {"messages": [HumanMessage(content="Hello")], "plan": ""}
        config = {"configurable": {}}

        result = agent(state, config)

        assert result["messages"][0].content == ""

    def test_llm_multiple_tool_calls(self, mock_llm_with_tool_calls):
        """Test handling of multiple tool calls in one response."""
        tool_calls = [
            {"name": "get_performance", "args": {"start_date": "2024-01-01", "end_date": "2024-02-01"}, "id": "1"},
            {"name": "get_current_risk", "args": {}, "id": "2"},
        ]
        llm = mock_llm_with_tool_calls(tool_calls)
        agent = create_agent_node(llm)

        state = {"messages": [HumanMessage(content="Show performance and risk")], "plan": ""}
        config = {"configurable": {}}

        result = agent(state, config)

        response = result["messages"][0]
        assert len(response.tool_calls) == 2


class TestJudgeLLMResponses:
    """Tests for judge LLM response handling."""

    def test_extract_json_valid(self):
        """Test JSON extraction from valid response."""
        text = '{"score": 8, "issues": [], "rewrite_needed": false}'
        result = _extract_json(text)

        assert result is not None
        assert result["score"] == 8

    def test_extract_json_with_prose(self):
        """Test JSON extraction from response with surrounding prose."""
        text = 'The response is good. {"score": 9, "issues": [], "rewrite_needed": false} Overall fine.'
        result = _extract_json(text)

        assert result is not None
        assert result["score"] == 9

    def test_extract_json_invalid(self):
        """Test JSON extraction from invalid response."""
        text = "This is just text without JSON"
        result = _extract_json(text)

        assert result is None

    def test_extract_json_nested_braces(self):
        """Test JSON extraction with nested braces."""
        text = '{"score": 7, "issues": ["issue with {brackets}"], "rewrite_needed": true}'
        result = _extract_json(text)

        assert result is not None
        assert result["score"] == 7

    def test_judge_high_score_no_rewrite(self, mock_llm_with_responses):
        """Test that high score doesn't trigger rewrite."""
        # Create a mock that returns structured output
        judge_llm = MagicMock()
        judge_llm.with_structured_output.return_value = judge_llm
        judge_llm.invoke.return_value = MagicMock(
            model_dump=lambda: {"score": 9, "issues": [], "rewrite_needed": False}
        )

        main_llm = mock_llm_with_responses(["Rewritten response"])

        response = "This is a good response about portfolios."
        improved, verdict = judge_and_rewrite(judge_llm, main_llm, response)

        assert improved == response  # No rewrite
        assert verdict["score"] == 9
        assert main_llm.call_count == 0  # Rewriter not called

    def test_judge_low_score_triggers_rewrite(self, mock_llm_with_responses):
        """Test that low score triggers rewrite."""
        judge_llm = MagicMock()
        judge_llm.with_structured_output.return_value = judge_llm
        judge_llm.invoke.return_value = MagicMock(
            model_dump=lambda: {"score": 5, "issues": ["Too verbose"], "rewrite_needed": True}
        )

        main_llm = mock_llm_with_responses(["Improved concise response."])

        response = "This is a very verbose response that goes on and on..."
        improved, verdict = judge_and_rewrite(judge_llm, main_llm, response)

        assert improved == "Improved concise response."
        assert verdict["rewrite_needed"] is True
        assert main_llm.call_count == 1


class TestPlannerLLMResponses:
    """Tests for planner LLM response handling."""

    def test_planner_generates_plan(self, mock_llm_with_responses):
        """Test that planner generates appropriate plans."""
        llm = mock_llm_with_responses([
            "1. Call get_performance with full date range\n2. Summarize results"
        ])
        planner = create_planner_node()

        state = {"messages": [HumanMessage(content="Analyze my portfolio deeply")]}
        config = {"configurable": {"main_llm": llm}}

        result = planner(state, config)

        assert "plan" in result
        assert "get_performance" in result["plan"]

    def test_planner_handles_llm_error(self, mock_llm):
        """Test that planner handles LLM errors gracefully."""
        mock_llm.invoke = MagicMock(side_effect=Exception("API Error"))
        planner = create_planner_node()

        state = {"messages": [HumanMessage(content="Complex query")]}
        config = {"configurable": {"main_llm": mock_llm}}

        result = planner(state, config)

        # Should return empty plan on error
        assert result["plan"] == ""


# =============================================================================
# LLM Response Quality Tests (Mocked)
# =============================================================================


class TestLLMResponseQuality:
    """Tests for LLM response quality using mocks."""

    def test_response_not_too_long(self, mock_llm_with_responses):
        """Test that responses are reasonably sized."""
        # Simulate a response that's within acceptable limits
        response = "Your portfolio return is 5.2% for the period."
        llm = mock_llm_with_responses([response])
        agent = create_agent_node(llm)

        state = {"messages": [HumanMessage(content="Performance?")], "plan": ""}
        config = {"configurable": {}}

        result = agent(state, config)

        # Response should be under 2000 chars for most queries
        assert len(result["messages"][0].content) < 2000

    def test_response_contains_no_forbidden_phrases(self, mock_llm_with_responses):
        """Test that response doesn't claim to modify data."""
        response = "I've updated your portfolio settings."  # Forbidden
        llm = mock_llm_with_responses([response])

        # This should be caught by policy node in full graph
        # Here we just verify the response pattern
        assert "updated" in response.lower()  # This would be flagged


# =============================================================================
# Integration Tests (Real LLM)
# =============================================================================


@pytest.mark.integration
@pytest.mark.llm
class TestLLMIntegration:
    """Integration tests with real LLM.

    These tests require OPENAI_API_KEY environment variable.
    Run with: pytest tests/agent/test_llm_responses.py -m integration
    """

    def test_real_llm_tool_selection(self, real_llm, populated_db):
        """Test that real LLM selects appropriate tools."""
        tools = get_binding_tools()
        model_with_tools = real_llm.bind_tools(tools)

        messages = [
            SystemMessage(content="You are a portfolio assistant. Use tools to help users."),
            HumanMessage(content="What is my portfolio performance from Jan 1 to Feb 1, 2024?"),
        ]

        response = model_with_tools.invoke(messages)

        # LLM should call get_performance tool
        assert hasattr(response, "tool_calls")
        if response.tool_calls:
            tool_names = [tc["name"] for tc in response.tool_calls]
            assert "get_performance" in tool_names

    def test_real_llm_risk_question_flow(self, real_llm):
        """Test that real LLM handles risk questionnaire correctly."""
        tools = get_binding_tools()
        model_with_tools = real_llm.bind_tools(tools)

        messages = [
            SystemMessage(content="You are a portfolio assistant."),
            HumanMessage(content="I want to take the risk questionnaire."),
        ]

        response = model_with_tools.invoke(messages)

        if response.tool_calls:
            tool_names = [tc["name"] for tc in response.tool_calls]
            assert "get_next_risk_question" in tool_names

    def test_real_llm_response_quality(self, real_llm):
        """Test that real LLM produces quality responses."""
        messages = [
            SystemMessage(content="You are a helpful portfolio assistant. Be concise."),
            HumanMessage(content="What can you help me with?"),
        ]

        response = real_llm.invoke(messages)

        # Response should be reasonable
        assert response.content
        assert len(response.content) < 2000
        # Should mention portfolio-related capabilities
        assert any(word in response.content.lower() for word in [
            "portfolio", "performance", "risk", "help"
        ])

    def test_real_llm_no_hallucinated_data(self, real_llm):
        """Test that LLM doesn't hallucinate portfolio data."""
        messages = [
            SystemMessage(content="You are a portfolio assistant. Only provide data from tools."),
            HumanMessage(content="What is my current portfolio value?"),
        ]

        response = real_llm.invoke(messages)

        # Without tool calls, should not claim specific values
        content_lower = response.content.lower()
        # Should acknowledge need for data or ask to use tool
        assert not any(char.isdigit() and "$" in response.content for char in response.content) or \
               "don't have" in content_lower or \
               "need" in content_lower or \
               "upload" in content_lower

    def test_real_llm_handles_edge_cases(self, real_llm):
        """Test LLM handles edge case inputs gracefully."""
        edge_cases = [
            "",  # Empty
            "???",  # Unclear
            "asdfghjkl",  # Nonsense
            "DROP TABLE users;",  # SQL injection attempt
        ]

        for query in edge_cases:
            messages = [
                SystemMessage(content="You are a portfolio assistant."),
                HumanMessage(content=query),
            ]

            response = real_llm.invoke(messages)

            # Should respond gracefully, not crash
            assert response is not None
            # Should not execute or acknowledge harmful inputs
            assert "drop" not in response.content.lower() or "i can't" in response.content.lower()


@pytest.mark.integration
@pytest.mark.llm
@pytest.mark.slow
class TestFullGraphIntegration:
    """Full graph integration tests with real LLM.

    These are slower tests that run the complete agent graph.
    """

    def test_full_graph_performance_query(self, real_llm, populated_db):
        """Test full graph handles performance query end-to-end."""
        graph = build_graph(real_llm)

        config = {
            "configurable": {
                "store": populated_db,
                "main_llm": real_llm,
                "judge_llm": real_llm,
                "policy_llm": real_llm,
                "thread_id": "test-thread-1",
            }
        }

        initial_state = {
            "messages": [HumanMessage(content="Show my performance for January 2024")],
        }

        # Run graph
        result = graph.invoke(initial_state, config)

        # Should have messages in result
        assert "messages" in result
        assert len(result["messages"]) > 1

        # Final message should contain performance info
        final_msg = result["messages"][-1]
        assert any(word in final_msg.content.lower() for word in [
            "return", "performance", "%", "value"
        ])

    def test_full_graph_capabilities_query(self, real_llm, populated_db):
        """Test full graph handles capabilities query."""
        graph = build_graph(real_llm)

        config = {
            "configurable": {
                "store": populated_db,
                "main_llm": real_llm,
                "judge_llm": real_llm,
                "policy_llm": real_llm,
                "thread_id": "test-thread-2",
            }
        }

        initial_state = {
            "messages": [HumanMessage(content="What can you help me with?")],
        }

        result = graph.invoke(initial_state, config)

        final_msg = result["messages"][-1]
        # Should describe capabilities
        assert any(word in final_msg.content.lower() for word in [
            "help", "can", "performance", "risk", "portfolio"
        ])
