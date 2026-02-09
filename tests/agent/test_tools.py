"""Tests for agent/tools.py - tool handlers."""
from datetime import date
from typing import Any, Dict

import pytest

from agent.tools import (
    TOOL_HANDLERS,
    get_binding_tools,
    run_tool,
)


class TestGetBindingTools:
    """Tests for get_binding_tools function."""

    def test_returns_list(self):
        """Test that get_binding_tools returns a list."""
        tools = get_binding_tools()
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_all_tools_have_names(self):
        """Test that all tools have names."""
        tools = get_binding_tools()
        for tool in tools:
            assert hasattr(tool, "name")
            assert tool.name

    def test_expected_tools_present(self):
        """Test that expected tools are present."""
        tools = get_binding_tools()
        tool_names = {tool.name for tool in tools}

        expected = {
            "get_capabilities",
            "get_edit_help",
            "get_current_risk",
            "suggest_rebalance",
            "get_next_risk_question",
            "submit_risk_answer",
            "get_performance",
            "fetch_market_trends",
            "apply_risk_profile_change",
        }

        assert expected.issubset(tool_names)


class TestRunTool:
    """Tests for run_tool function."""

    def test_unknown_tool_returns_error(self):
        """Test that unknown tool returns error message."""
        result, updates = run_tool("unknown_tool", {}, {}, {})

        assert "Unknown tool" in result
        assert updates == {}


class TestGetCapabilitiesHandler:
    """Tests for get_capabilities tool handler."""

    def test_returns_capabilities_description(self):
        """Test that handler returns capabilities description."""
        result, updates = run_tool("get_capabilities", {}, {}, {})

        assert "read-only" in result.lower()
        assert "performance" in result.lower()
        assert "risk" in result.lower()
        assert updates == {}


class TestGetEditHelpHandler:
    """Tests for get_edit_help tool handler."""

    def test_first_call_suggests_upload(self):
        """Test first call suggests CSV upload."""
        state = {}
        result, updates = run_tool("get_edit_help", {}, state, {})

        assert "upload" in result.lower() or "csv" in result.lower()
        assert updates.get("already_suggested_upload_for_edit") is True

    def test_second_call_suggests_advisor(self):
        """Test second call suggests human advisor."""
        state = {"already_suggested_upload_for_edit": True}
        result, updates = run_tool("get_edit_help", {}, state, {})

        assert "advisor" in result.lower() or "permission" in result.lower()


class TestGetCurrentRiskHandler:
    """Tests for get_current_risk tool handler."""

    def test_no_store_returns_message(self, config_without_store):
        """Test that missing store returns appropriate message."""
        result, updates = run_tool("get_current_risk", {}, {}, config_without_store)

        assert "no portfolio" in result.lower() or "upload" in result.lower()

    def test_empty_portfolio_returns_message(self, temp_db, mock_llm):
        """Test empty portfolio returns appropriate message."""
        config = {"configurable": {"store": temp_db, "main_llm": mock_llm}}
        result, updates = run_tool("get_current_risk", {}, {}, config)

        assert "no portfolio" in result.lower() or "upload" in result.lower()

    def test_returns_inferred_risk(self, agent_config):
        """Test that risk is inferred from allocation."""
        result, updates = run_tool("get_current_risk", {}, {}, agent_config)

        # Should contain risk category and allocation percentages
        assert any(risk in result for risk in ["Conservative", "Moderate", "Aggressive"])
        assert "%" in result


class TestSuggestRebalanceHandler:
    """Tests for suggest_rebalance tool handler."""

    def test_no_store_returns_message(self, config_without_store):
        """Test that missing store returns appropriate message."""
        result, updates = run_tool("suggest_rebalance", {}, {}, config_without_store)

        assert "not available" in result.lower() or "upload" in result.lower()

    def test_returns_rebalance_suggestion(self, agent_config):
        """Test that rebalance suggestion is returned."""
        result, updates = run_tool("suggest_rebalance", {}, {}, agent_config)

        assert "target" in result.lower() or "allocation" in result.lower()

    def test_respects_target_risk_parameter(self, agent_config):
        """Test that target_risk parameter is respected."""
        args = {"target_risk": "Aggressive"}
        result, updates = run_tool("suggest_rebalance", args, {}, agent_config)

        assert "Aggressive" in result


class TestGetNextRiskQuestionHandler:
    """Tests for get_next_risk_question tool handler."""

    def test_starts_with_question_1(self):
        """Test that handler starts with question 1."""
        state = {}
        result, updates = run_tool("get_next_risk_question", {}, state, {})

        assert "Question 1" in result
        assert "of 4" in result

    def test_returns_next_unanswered_question(self):
        """Test that handler returns next unanswered question."""
        state = {"risk_answers": {"1": "8+ years"}}
        result, updates = run_tool("get_next_risk_question", {}, state, {})

        assert "Question 2" in result

    def test_all_answered_restarts_questionnaire(self):
        """Test that completed questionnaire restarts from question 1."""
        state = {
            "risk_answers": {
                "1": "8+ years",
                "2": "high stability",
                "3": "maximum growth",
                "4": "buy more",
            }
        }
        result, updates = run_tool("get_next_risk_question", {}, state, {})

        # When all questions are answered, handler restarts from Q1
        assert "Question 1" in result
        assert updates.get("risk_answers") == {}

    def test_restarts_if_completed(self):
        """Test questionnaire restarts if already completed."""
        state = {
            "risk_answers": {"1": "a", "2": "b", "3": "c", "4": "d"},
            "risk_profile": "Moderate",
        }
        result, updates = run_tool("get_next_risk_question", {}, state, {})

        assert "Question 1" in result
        assert updates.get("risk_answers") == {}


class TestSubmitRiskAnswerHandler:
    """Tests for submit_risk_answer tool handler."""

    def test_records_answer_and_returns_next(self):
        """Test that answer is recorded and next question returned."""
        state = {"risk_answers": {}}
        args = {"question_number": "1", "answer_text": "8+ years"}

        result, updates = run_tool("submit_risk_answer", args, state, {})

        assert "Question 2" in result
        assert updates["risk_answers"]["1"] == "8+ years"

    def test_final_answer_returns_profile(self):
        """Test that final answer returns profile."""
        state = {"risk_answers": {"1": "a", "2": "b", "3": "c"}}
        args = {"question_number": "4", "answer_text": "buy more"}

        result, updates = run_tool("submit_risk_answer", args, state, {})

        assert "profile" in result.lower()
        assert "risk_profile" in updates

    def test_empty_answer_shows_question(self):
        """Test that empty answer with no prior answers shows question 1."""
        state = {"risk_answers": {}}
        args = {"question_number": "1", "answer_text": ""}

        result, updates = run_tool("submit_risk_answer", args, state, {})

        assert "Question 1" in result


class TestGetPerformanceHandler:
    """Tests for get_performance tool handler."""

    def test_no_store_returns_message(self, config_without_store):
        """Test that missing store returns appropriate message."""
        args = {"start_date": "2024-01-01", "end_date": "2024-02-01"}
        result, updates = run_tool("get_performance", args, {}, config_without_store)

        assert "no portfolio" in result.lower() or "upload" in result.lower()

    def test_returns_performance_metrics(self, agent_config):
        """Test that performance metrics are returned."""
        args = {"start_date": "2024-01-01", "end_date": "2024-02-01"}
        result, updates = run_tool("get_performance", args, {}, agent_config)

        assert "return" in result.lower() or "%" in result
        assert updates.get("did_compute_metrics") is True

    def test_handles_date_clamping(self, agent_config):
        """Test that dates outside range are clamped."""
        args = {"start_date": "2020-01-01", "end_date": "2030-12-31"}
        result, updates = run_tool("get_performance", args, {}, agent_config)

        # Should still return a result (clamped to available data)
        assert "value" in result.lower() or "return" in result.lower()

    def test_asset_class_filter(self, agent_config):
        """Test filtering by asset class."""
        args = {
            "start_date": "2024-01-01",
            "end_date": "2024-02-01",
            "asset_class": "Stock",
        }
        result, updates = run_tool("get_performance", args, {}, agent_config)

        assert "Stock" in result


class TestApplyRiskProfileChangeHandler:
    """Tests for apply_risk_profile_change tool handler."""

    def test_no_pending_change_returns_message(self):
        """Test that no pending change returns appropriate message."""
        state = {"pending_kind": "none"}
        result, updates = run_tool("apply_risk_profile_change", {}, state, {})

        assert "no pending" in result.lower()

    def test_applies_pending_change(self):
        """Test that pending change is applied."""
        state = {
            "pending_kind": "apply_risk_change",
            "pending_payload": {"suggested_profile": "Aggressive"},
        }
        result, updates = run_tool("apply_risk_profile_change", {}, state, {})

        assert "Aggressive" in result
        assert updates.get("risk_profile") == "Aggressive"

    def test_invalid_profile_returns_error(self):
        """Test that invalid profile returns error."""
        state = {
            "pending_kind": "apply_risk_change",
            "pending_payload": {"suggested_profile": "InvalidProfile"},
        }
        result, updates = run_tool("apply_risk_profile_change", {}, state, {})

        assert "couldn't" in result.lower() or "invalid" in result.lower()


class TestToolHandlersRegistry:
    """Tests for TOOL_HANDLERS registry."""

    def test_all_binding_tools_have_handlers(self):
        """Test that all binding tools have corresponding handlers."""
        tools = get_binding_tools()
        tool_names = {tool.name for tool in tools}

        for name in tool_names:
            assert name in TOOL_HANDLERS, f"Handler missing for tool: {name}"
