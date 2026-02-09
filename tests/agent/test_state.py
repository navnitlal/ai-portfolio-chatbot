"""Tests for agent/state.py - state definitions and helpers."""
import pytest
from langchain_core.messages import HumanMessage, AIMessage

from agent.state import (
    ChatState,
    GraphState,
    JudgeVerdict,
    PolicyVerdict,
    MAX_MESSAGES_IN_CONTEXT,
    MAX_REFLECTIONS,
    REFLECTION_THRESHOLD,
    dict_to_state,
    extract_tool_call_info,
    state_to_dict,
)


class TestConstants:
    """Tests for state constants."""

    def test_max_messages_reasonable(self):
        """Test that MAX_MESSAGES_IN_CONTEXT is reasonable."""
        assert MAX_MESSAGES_IN_CONTEXT > 0
        assert MAX_MESSAGES_IN_CONTEXT <= 100

    def test_max_reflections_limited(self):
        """Test that MAX_REFLECTIONS is limited."""
        assert MAX_REFLECTIONS >= 0
        assert MAX_REFLECTIONS <= 5

    def test_reflection_threshold_in_range(self):
        """Test that REFLECTION_THRESHOLD is in valid range."""
        assert REFLECTION_THRESHOLD >= 1
        assert REFLECTION_THRESHOLD <= 10


class TestJudgeVerdict:
    """Tests for JudgeVerdict schema."""

    def test_valid_verdict(self):
        """Test creating a valid JudgeVerdict."""
        verdict = JudgeVerdict(score=8, issues=[], rewrite_needed=False)

        assert verdict.score == 8
        assert verdict.issues == []
        assert verdict.rewrite_needed is False

    def test_verdict_with_issues(self):
        """Test verdict with issues."""
        verdict = JudgeVerdict(
            score=5,
            issues=["Too verbose", "Missing context"],
            rewrite_needed=True,
        )

        assert len(verdict.issues) == 2
        assert verdict.rewrite_needed is True

    def test_verdict_defaults(self):
        """Test verdict default values."""
        verdict = JudgeVerdict(score=7)

        assert verdict.issues == []
        assert verdict.rewrite_needed is False


class TestPolicyVerdict:
    """Tests for PolicyVerdict schema."""

    def test_valid_policy_verdict(self):
        """Test creating a valid PolicyVerdict."""
        verdict = PolicyVerdict(ok=True, issues=[], safe_text="")

        assert verdict.ok is True
        assert verdict.issues == []

    def test_policy_verdict_with_issues(self):
        """Test policy verdict with issues."""
        verdict = PolicyVerdict(
            ok=False,
            issues=["Claims to modify data"],
            safe_text="I cannot modify your data.",
        )

        assert verdict.ok is False
        assert len(verdict.issues) == 1
        assert verdict.safe_text != ""


class TestChatState:
    """Tests for ChatState dataclass."""

    def test_default_state(self):
        """Test creating default ChatState."""
        state = ChatState()

        assert state.messages == []
        assert state.risk_answers == {}
        assert state.risk_profile is None
        assert state.pending_kind == "none"
        assert state.plan == ""
        assert state.reflection_count == 0

    def test_state_with_values(self):
        """Test creating ChatState with values."""
        state = ChatState(
            messages=[HumanMessage(content="Hello")],
            risk_profile="Moderate",
            plan="Call get_performance",
        )

        assert len(state.messages) == 1
        assert state.risk_profile == "Moderate"
        assert state.plan == "Call get_performance"


class TestStateConversion:
    """Tests for state conversion functions."""

    def test_state_to_dict(self):
        """Test converting ChatState to dict."""
        state = ChatState(
            messages=[HumanMessage(content="Test")],
            risk_profile="Aggressive",
            reflection_count=1,
        )

        result = state_to_dict(state)

        assert isinstance(result, dict)
        assert result["risk_profile"] == "Aggressive"
        assert result["reflection_count"] == 1
        assert len(result["messages"]) == 1

    def test_dict_to_state(self):
        """Test converting dict to ChatState."""
        d = {
            "messages": [HumanMessage(content="Test")],
            "risk_profile": "Conservative",
            "plan": "Test plan",
        }

        result = dict_to_state(d)

        assert isinstance(result, ChatState)
        assert result.risk_profile == "Conservative"
        assert result.plan == "Test plan"

    def test_dict_to_state_missing_fields(self):
        """Test dict_to_state with missing fields uses defaults."""
        d = {"risk_profile": "Moderate"}

        result = dict_to_state(d)

        assert result.risk_profile == "Moderate"
        assert result.messages == []
        assert result.plan == ""

    def test_roundtrip_conversion(self):
        """Test state -> dict -> state roundtrip."""
        original = ChatState(
            messages=[HumanMessage(content="Hello")],
            risk_profile="Aggressive",
            risk_answers={"1": "answer1"},
            plan="Test plan",
            reflection_count=1,
        )

        d = state_to_dict(original)
        restored = dict_to_state(d)

        assert restored.risk_profile == original.risk_profile
        assert restored.plan == original.plan
        assert restored.reflection_count == original.reflection_count
        assert len(restored.messages) == len(original.messages)


class TestExtractToolCallInfo:
    """Tests for extract_tool_call_info function."""

    def test_extract_from_dict(self):
        """Test extracting info from dict tool call."""
        tc = {"name": "get_performance", "args": {"start_date": "2024-01-01"}, "id": "123"}

        name, args, tid = extract_tool_call_info(tc)

        assert name == "get_performance"
        assert args == {"start_date": "2024-01-01"}
        assert tid == "123"

    def test_extract_from_object(self):
        """Test extracting info from object tool call."""
        class MockToolCall:
            name = "suggest_rebalance"
            args = {"target_risk": "Aggressive"}
            id = "456"

        tc = MockToolCall()
        name, args, tid = extract_tool_call_info(tc)

        assert name == "suggest_rebalance"
        assert args["target_risk"] == "Aggressive"
        assert tid == "456"

    def test_missing_fields_default_to_empty(self):
        """Test that missing fields default to empty values."""
        tc = {}

        name, args, tid = extract_tool_call_info(tc)

        assert name == ""
        assert args == {}
        assert tid == ""

    def test_none_args_becomes_empty_dict(self):
        """Test that None args becomes empty dict."""
        tc = {"name": "test", "args": None, "id": "1"}

        name, args, tid = extract_tool_call_info(tc)

        assert args == {}
