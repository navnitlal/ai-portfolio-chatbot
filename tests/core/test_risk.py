"""Tests for core/risk.py - risk profiling functions."""
import pytest

from core.risk import (
    MIN_CASH_TARGET,
    TARGET_ALLOCATION_BY_RISK,
    infer_risk_from_allocation,
    score_risk_answers,
)


class TestInferRiskFromAllocation:
    """Tests for infer_risk_from_allocation function."""

    def test_aggressive_high_equity(self):
        """Test aggressive profile with high equity allocation."""
        result = infer_risk_from_allocation(
            stock_pct=0.60,
            bond_pct=0.10,
            cash_pct=0.05,
            etf_pct=0.25,  # Total equity = 0.85
        )
        assert result == "Aggressive"

    def test_moderate_balanced(self):
        """Test moderate profile with balanced allocation."""
        result = infer_risk_from_allocation(
            stock_pct=0.35,
            bond_pct=0.35,
            cash_pct=0.10,
            etf_pct=0.20,  # Total equity = 0.55
        )
        assert result == "Moderate"

    def test_conservative_low_equity(self):
        """Test conservative profile with low equity allocation."""
        result = infer_risk_from_allocation(
            stock_pct=0.15,
            bond_pct=0.50,
            cash_pct=0.25,
            etf_pct=0.10,  # Total equity = 0.25
        )
        assert result == "Conservative"

    def test_boundary_aggressive_moderate(self):
        """Test boundary between aggressive and moderate (equity = 0.65)."""
        result = infer_risk_from_allocation(
            stock_pct=0.50,
            bond_pct=0.20,
            cash_pct=0.15,
            etf_pct=0.15,  # Total equity = 0.65
        )
        assert result == "Aggressive"

    def test_boundary_moderate_conservative(self):
        """Test boundary between moderate and conservative (equity = 0.35)."""
        result = infer_risk_from_allocation(
            stock_pct=0.25,
            bond_pct=0.40,
            cash_pct=0.25,
            etf_pct=0.10,  # Total equity = 0.35
        )
        assert result == "Moderate"

    def test_edge_case_all_stock(self):
        """Test edge case with 100% stock."""
        result = infer_risk_from_allocation(
            stock_pct=1.0,
            bond_pct=0.0,
            cash_pct=0.0,
            etf_pct=0.0,
        )
        assert result == "Aggressive"

    def test_edge_case_all_bonds(self):
        """Test edge case with 100% bonds."""
        result = infer_risk_from_allocation(
            stock_pct=0.0,
            bond_pct=1.0,
            cash_pct=0.0,
            etf_pct=0.0,
        )
        assert result == "Conservative"

    def test_default_etf_zero(self):
        """Test that ETF defaults to 0 if not provided."""
        result = infer_risk_from_allocation(
            stock_pct=0.50,
            bond_pct=0.30,
            cash_pct=0.20,
        )
        assert result == "Moderate"


class TestScoreRiskAnswers:
    """Tests for score_risk_answers function."""

    def test_aggressive_answers(self):
        """Test scoring for aggressive risk profile."""
        answers = {
            "1": "8+ years",         # Long horizon (+2)
            "2": "high stability",   # High income (+2)
            "3": "maximum growth",   # Max growth (+2)
            "4": "buy more",         # Buy on dip (+2)
        }
        result = score_risk_answers(answers)
        assert result == "Aggressive"

    def test_conservative_answers(self):
        """Test scoring for conservative risk profile."""
        answers = {
            "1": "1-2 years",        # Short horizon (+0)
            "2": "low stability",    # Low income (+0)
            "3": "preserve capital", # Capital preservation (+0)
            "4": "sell immediately", # Panic sell (+0)
        }
        result = score_risk_answers(answers)
        assert result == "Conservative"

    def test_moderate_answers(self):
        """Test scoring for moderate risk profile."""
        answers = {
            "1": "3-7 years",        # Medium horizon (+1)
            "2": "medium stability", # Medium income (+1)
            "3": "balanced growth",  # Balanced (+1)
            "4": "hold steady",      # Hold (+1)
        }
        result = score_risk_answers(answers)
        assert result == "Moderate"

    def test_mixed_answers(self):
        """Test scoring with mixed answers."""
        answers = {
            "1": "8+ years",         # Long (+2)
            "2": "low stability",    # Low (+0)
            "3": "balanced growth",  # Balanced (+1)
            "4": "hold steady",      # Hold (+1)
        }
        result = score_risk_answers(answers)
        # Score = 4, which is Moderate (3-5)
        assert result == "Moderate"

    def test_empty_answers(self):
        """Test scoring with empty answers."""
        result = score_risk_answers({})
        assert result == "Conservative"

    def test_partial_answers(self):
        """Test scoring with partial answers."""
        answers = {
            "1": "8+ years",
            "3": "maximum growth",
        }
        result = score_risk_answers(answers)
        # Score = 4 (2 + 2)
        assert result == "Moderate"

    def test_case_insensitive(self):
        """Test that answers are case insensitive."""
        answers = {
            "1": "8+ YEARS",
            "2": "HIGH stability",
            "3": "MAXIMUM Growth",
            "4": "BUY more",
        }
        result = score_risk_answers(answers)
        assert result == "Aggressive"


class TestTargetAllocationByRisk:
    """Tests for TARGET_ALLOCATION_BY_RISK constants."""

    def test_all_profiles_defined(self):
        """Test all three risk profiles are defined."""
        assert "Conservative" in TARGET_ALLOCATION_BY_RISK
        assert "Moderate" in TARGET_ALLOCATION_BY_RISK
        assert "Aggressive" in TARGET_ALLOCATION_BY_RISK

    def test_allocations_sum_to_one(self):
        """Test that allocations for each profile sum to 1."""
        for profile, alloc in TARGET_ALLOCATION_BY_RISK.items():
            total = sum(alloc)
            assert total == pytest.approx(1.0, rel=0.001), f"{profile} allocation doesn't sum to 1"

    def test_cash_at_minimum(self):
        """Test that cash is at minimum target for all profiles."""
        for profile, alloc in TARGET_ALLOCATION_BY_RISK.items():
            stock, bond, cash, etf = alloc
            assert cash == pytest.approx(MIN_CASH_TARGET, rel=0.001), f"{profile} cash not at minimum"

    def test_aggressive_has_most_stock(self):
        """Test that aggressive profile has highest stock allocation."""
        conservative_stock = TARGET_ALLOCATION_BY_RISK["Conservative"][0]
        moderate_stock = TARGET_ALLOCATION_BY_RISK["Moderate"][0]
        aggressive_stock = TARGET_ALLOCATION_BY_RISK["Aggressive"][0]

        assert aggressive_stock > moderate_stock > conservative_stock

    def test_conservative_has_most_bonds(self):
        """Test that conservative profile has highest bond allocation."""
        conservative_bond = TARGET_ALLOCATION_BY_RISK["Conservative"][1]
        moderate_bond = TARGET_ALLOCATION_BY_RISK["Moderate"][1]
        aggressive_bond = TARGET_ALLOCATION_BY_RISK["Aggressive"][1]

        assert conservative_bond > moderate_bond > aggressive_bond
