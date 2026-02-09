"""Tests for agent/routing.py - conditional edge routing."""
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent.routing import reflect_or_pass, route_from_agent
from agent.state import MAX_REFLECTIONS, REFLECTION_THRESHOLD


class TestRouteFromAgent:
    """Tests for route_from_agent function."""

    def test_no_messages_routes_to_judge(self):
        """Test that empty messages routes to judge."""
        state = {"messages": []}
        result = route_from_agent(state)
        assert result == "judge"

    def test_human_message_routes_to_judge(self):
        """Test that human message routes to judge."""
        state = {"messages": [HumanMessage(content="Hello")]}
        result = route_from_agent(state)
        assert result == "judge"

    def test_ai_message_no_tools_routes_to_judge(self):
        """Test that AI message without tool calls routes to judge."""
        ai_msg = AIMessage(content="Here's your answer")
        state = {"messages": [ai_msg]}
        result = route_from_agent(state)
        assert result == "judge"

    def test_risk_tool_routes_to_risk_questionnaire(self):
        """Test that risk tools route to risk_questionnaire."""
        ai_msg = MagicMock(spec=AIMessage)
        ai_msg.tool_calls = [{"name": "get_next_risk_question", "args": {}, "id": "1"}]
        state = {"messages": [ai_msg]}

        result = route_from_agent(state)
        assert result == "risk_questionnaire"

    def test_submit_risk_answer_routes_to_risk_questionnaire(self):
        """Test that submit_risk_answer routes to risk_questionnaire."""
        ai_msg = MagicMock(spec=AIMessage)
        ai_msg.tool_calls = [
            {"name": "submit_risk_answer", "args": {"question_number": "1", "answer_text": "test"}, "id": "1"}
        ]
        state = {"messages": [ai_msg]}

        result = route_from_agent(state)
        assert result == "risk_questionnaire"

    def test_market_trends_routes_to_market_analysis(self):
        """Test that fetch_market_trends routes to market_analysis."""
        ai_msg = MagicMock(spec=AIMessage)
        ai_msg.tool_calls = [{"name": "fetch_market_trends", "args": {}, "id": "1"}]
        state = {"messages": [ai_msg]}

        result = route_from_agent(state)
        assert result == "market_analysis"

    def test_other_tools_route_to_tools(self):
        """Test that other tools route to tools node."""
        ai_msg = MagicMock(spec=AIMessage)
        ai_msg.tool_calls = [{"name": "get_performance", "args": {}, "id": "1"}]
        state = {"messages": [ai_msg]}

        result = route_from_agent(state)
        assert result == "tools"

    def test_multiple_tools_risk_takes_priority(self):
        """Test that risk tools take priority over other tools."""
        ai_msg = MagicMock(spec=AIMessage)
        ai_msg.tool_calls = [
            {"name": "get_performance", "args": {}, "id": "1"},
            {"name": "get_next_risk_question", "args": {}, "id": "2"},
        ]
        state = {"messages": [ai_msg]}

        result = route_from_agent(state)
        assert result == "risk_questionnaire"

    def test_multiple_tools_market_second_priority(self):
        """Test that market analysis takes second priority."""
        ai_msg = MagicMock(spec=AIMessage)
        ai_msg.tool_calls = [
            {"name": "get_performance", "args": {}, "id": "1"},
            {"name": "fetch_market_trends", "args": {}, "id": "2"},
        ]
        state = {"messages": [ai_msg]}

        result = route_from_agent(state)
        assert result == "market_analysis"


class TestReflectOrPass:
    """Tests for reflect_or_pass function."""

    def test_high_score_passes_to_policy(self):
        """Test that high score passes to policy."""
        state = {
            "last_judge": {"score": 9, "issues": []},
            "reflection_count": 0,
        }
        result = reflect_or_pass(state)
        assert result == "policy"

    def test_low_score_triggers_reflect(self):
        """Test that low score triggers reflection."""
        state = {
            "last_judge": {"score": REFLECTION_THRESHOLD - 1, "issues": ["issue1"]},
            "reflection_count": 0,
        }
        result = reflect_or_pass(state)
        assert result == "reflect"

    def test_max_reflections_reached_passes_to_policy(self):
        """Test that max reflections passes to policy."""
        state = {
            "last_judge": {"score": REFLECTION_THRESHOLD - 1, "issues": ["issue1"]},
            "reflection_count": MAX_REFLECTIONS,
        }
        result = reflect_or_pass(state)
        assert result == "policy"

    def test_threshold_score_passes_to_policy(self):
        """Test that threshold score exactly passes to policy."""
        state = {
            "last_judge": {"score": REFLECTION_THRESHOLD, "issues": []},
            "reflection_count": 0,
        }
        result = reflect_or_pass(state)
        assert result == "policy"

    def test_missing_judge_passes_to_policy(self):
        """Test that missing judge info passes to policy."""
        state = {"reflection_count": 0}
        result = reflect_or_pass(state)
        assert result == "policy"

    def test_missing_score_defaults_to_10(self):
        """Test that missing score defaults to 10 (passes)."""
        state = {
            "last_judge": {"issues": []},
            "reflection_count": 0,
        }
        result = reflect_or_pass(state)
        assert result == "policy"
