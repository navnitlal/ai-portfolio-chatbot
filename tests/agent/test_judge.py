"""Tests for agent/judge.py - response quality scoring and rewriting."""
from unittest.mock import MagicMock

import pytest

from agent.judge import _extract_json, judge_and_rewrite


class TestExtractJson:
    """Tests for _extract_json function."""

    def test_simple_json(self):
        """Test extraction of simple JSON."""
        text = '{"score": 8, "issues": [], "rewrite_needed": false}'
        result = _extract_json(text)

        assert result == {"score": 8, "issues": [], "rewrite_needed": False}

    def test_json_with_prefix(self):
        """Test extraction with text before JSON."""
        text = 'Here is my assessment: {"score": 7, "issues": ["verbose"], "rewrite_needed": true}'
        result = _extract_json(text)

        assert result["score"] == 7
        assert "verbose" in result["issues"]

    def test_json_with_suffix(self):
        """Test extraction with text after JSON."""
        text = '{"score": 9, "issues": [], "rewrite_needed": false} That is all.'
        result = _extract_json(text)

        assert result["score"] == 9

    def test_json_with_nested_objects(self):
        """Test extraction with nested JSON objects."""
        text = '{"outer": {"inner": "value"}, "score": 5}'
        result = _extract_json(text)

        assert result["outer"]["inner"] == "value"
        assert result["score"] == 5

    def test_no_json_returns_none(self):
        """Test that missing JSON returns None."""
        text = "This response has no JSON at all."
        result = _extract_json(text)

        assert result is None

    def test_invalid_json_returns_none(self):
        """Test that invalid JSON returns None."""
        text = '{"score": 8, "issues": ['  # Incomplete
        result = _extract_json(text)

        assert result is None

    def test_empty_string(self):
        """Test empty string input."""
        result = _extract_json("")
        assert result is None

    def test_none_input(self):
        """Test None input."""
        result = _extract_json(None)
        assert result is None

    def test_json_with_special_characters(self):
        """Test JSON with special characters in strings."""
        text = '{"score": 6, "issues": ["Contains \\"quotes\\" and {braces}"], "rewrite_needed": true}'
        result = _extract_json(text)

        assert result is not None
        assert result["score"] == 6


class TestJudgeAndRewrite:
    """Tests for judge_and_rewrite function."""

    @pytest.fixture
    def mock_judge_llm(self):
        """Create a mock judge LLM."""
        llm = MagicMock()
        llm.with_structured_output.return_value = llm
        return llm

    @pytest.fixture
    def mock_main_llm(self):
        """Create a mock main LLM."""
        llm = MagicMock()
        return llm

    def test_high_score_no_rewrite(self, mock_judge_llm, mock_main_llm):
        """Test that high score doesn't trigger rewrite."""
        mock_judge_llm.invoke.return_value = MagicMock(
            model_dump=lambda: {"score": 9, "issues": [], "rewrite_needed": False}
        )

        response = "This is a good response."
        improved, verdict = judge_and_rewrite(mock_judge_llm, mock_main_llm, response)

        assert improved == response
        assert verdict["score"] == 9
        mock_main_llm.invoke.assert_not_called()

    def test_low_score_triggers_rewrite(self, mock_judge_llm, mock_main_llm):
        """Test that low score triggers rewrite."""
        mock_judge_llm.invoke.return_value = MagicMock(
            model_dump=lambda: {"score": 4, "issues": ["Too verbose"], "rewrite_needed": True}
        )
        mock_main_llm.invoke.return_value = MagicMock(content="Improved response.")

        response = "This is a very long and verbose response that could be shorter."
        improved, verdict = judge_and_rewrite(mock_judge_llm, mock_main_llm, response)

        assert improved == "Improved response."
        assert verdict["rewrite_needed"] is True
        mock_main_llm.invoke.assert_called_once()

    def test_threshold_score_no_rewrite(self, mock_judge_llm, mock_main_llm):
        """Test that threshold score (7) doesn't trigger rewrite."""
        mock_judge_llm.invoke.return_value = MagicMock(
            model_dump=lambda: {"score": 7, "issues": [], "rewrite_needed": False}
        )

        response = "Acceptable response."
        improved, verdict = judge_and_rewrite(mock_judge_llm, mock_main_llm, response, threshold=7)

        assert improved == response
        mock_main_llm.invoke.assert_not_called()

    def test_custom_threshold(self, mock_judge_llm, mock_main_llm):
        """Test custom threshold parameter."""
        mock_judge_llm.invoke.return_value = MagicMock(
            model_dump=lambda: {"score": 8, "issues": ["Minor issue"], "rewrite_needed": True}
        )
        mock_main_llm.invoke.return_value = MagicMock(content="Rewritten.")

        response = "Original response."
        improved, verdict = judge_and_rewrite(
            mock_judge_llm, mock_main_llm, response, threshold=9
        )

        # Score 8 is below threshold 9, so should rewrite
        assert improved == "Rewritten."

    def test_structured_output_fallback(self, mock_judge_llm, mock_main_llm):
        """Test fallback when structured output fails."""
        # First call (structured) raises, second call returns JSON text
        mock_judge_llm.with_structured_output.side_effect = Exception("Not supported")
        mock_judge_llm.invoke.return_value = MagicMock(
            content='{"score": 8, "issues": [], "rewrite_needed": false}'
        )

        response = "Test response."
        improved, verdict = judge_and_rewrite(mock_judge_llm, mock_main_llm, response)

        assert improved == response
        assert verdict["score"] == 8

    def test_rewrite_filters_meta_commentary(self, mock_judge_llm, mock_main_llm):
        """Test that rewrite with meta-commentary is rejected."""
        mock_judge_llm.invoke.return_value = MagicMock(
            model_dump=lambda: {"score": 5, "issues": ["Issue"], "rewrite_needed": True}
        )
        # Rewriter returns meta-commentary instead of clean response
        mock_main_llm.invoke.return_value = MagicMock(
            content="Score: 8/10. I can rewrite this better. Here's the improved version..."
        )

        response = "Original response."
        improved, verdict = judge_and_rewrite(mock_judge_llm, mock_main_llm, response)

        # Should keep original when rewrite contains meta-commentary
        assert improved == response

    def test_empty_rewrite_returns_original(self, mock_judge_llm, mock_main_llm):
        """Test that empty rewrite returns original."""
        mock_judge_llm.invoke.return_value = MagicMock(
            model_dump=lambda: {"score": 5, "issues": ["Issue"], "rewrite_needed": True}
        )
        mock_main_llm.invoke.return_value = MagicMock(content="")

        response = "Original response."
        improved, verdict = judge_and_rewrite(mock_judge_llm, mock_main_llm, response)

        assert improved == response

    def test_issues_passed_to_rewriter(self, mock_judge_llm, mock_main_llm):
        """Test that issues are passed to the rewriter prompt."""
        mock_judge_llm.invoke.return_value = MagicMock(
            model_dump=lambda: {"score": 4, "issues": ["Too long", "Missing disclaimer"], "rewrite_needed": True}
        )
        mock_main_llm.invoke.return_value = MagicMock(content="Fixed response.")

        response = "Original."
        judge_and_rewrite(mock_judge_llm, mock_main_llm, response)

        # Check that issues were in the prompt
        call_args = mock_main_llm.invoke.call_args[0][0]
        prompt_content = call_args[0].content
        assert "Too long" in prompt_content
        assert "Missing disclaimer" in prompt_content

    def test_rewrite_includes_rules(self, mock_judge_llm, mock_main_llm):
        """Test that rewrite prompt includes important rules."""
        mock_judge_llm.invoke.return_value = MagicMock(
            model_dump=lambda: {"score": 5, "issues": ["Issue"], "rewrite_needed": True}
        )
        mock_main_llm.invoke.return_value = MagicMock(content="Fixed.")

        response = "Original."
        judge_and_rewrite(mock_judge_llm, mock_main_llm, response)

        call_args = mock_main_llm.invoke.call_args[0][0]
        prompt_content = call_args[0].content

        # Should include key rules
        assert "READ-ONLY" in prompt_content
        assert "disclaimer" in prompt_content.lower()
