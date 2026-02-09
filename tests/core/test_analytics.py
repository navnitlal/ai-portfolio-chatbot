"""Tests for core/analytics.py - portfolio analytics functions."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from core.analytics import (
    add_total,
    allocation_on_date,
    held_asset_classes_recent,
    max_drawdown,
    normalize_wide_csv,
    time_weighted_return,
)


class TestNormalizeWideCsv:
    """Tests for normalize_wide_csv function."""

    def test_valid_csv_normalized(self, minimal_portfolio_df):
        """Test that valid CSV data is normalized correctly."""
        result = normalize_wide_csv(minimal_portfolio_df)

        assert "TotalValue" in result.columns
        assert result["Date"].dtype == object  # date objects
        assert len(result) == 3

    def test_missing_required_columns_raises(self):
        """Test that missing required columns raise ValueError."""
        df = pd.DataFrame({"Date": ["2024-01-01"], "StockValue": [100]})

        with pytest.raises(ValueError, match="Missing required columns"):
            normalize_wide_csv(df)

    def test_non_numeric_values_raise(self):
        """Test that non-numeric values raise ValueError."""
        df = pd.DataFrame({
            "Date": ["2024-01-01"],
            "StockValue": ["not_a_number"],
            "BondValue": [1000],
            "ETFValue": [500],
            "CashValue": [200],
        })

        with pytest.raises(ValueError, match="not numeric"):
            normalize_wide_csv(df)

    def test_negative_values_raise(self):
        """Test that negative values raise ValueError."""
        df = pd.DataFrame({
            "Date": ["2024-01-01"],
            "StockValue": [-100],
            "BondValue": [1000],
            "ETFValue": [500],
            "CashValue": [200],
        })

        with pytest.raises(ValueError, match="must be >= 0"):
            normalize_wide_csv(df)

    def test_cash_flow_columns_optional(self):
        """Test that CashAddition/CashWithdrawal are optional."""
        df = pd.DataFrame({
            "Date": ["2024-01-01"],
            "StockValue": [100],
            "BondValue": [100],
            "ETFValue": [50],
            "CashValue": [50],
        })

        result = normalize_wide_csv(df)

        assert "CashAddition" in result.columns
        assert "CashWithdrawal" in result.columns
        assert result["CashAddition"].iloc[0] == 0.0

    def test_duplicates_removed_keep_last(self):
        """Test that duplicate dates keep the last entry."""
        df = pd.DataFrame({
            "Date": ["2024-01-01", "2024-01-01"],
            "StockValue": [100, 200],
            "BondValue": [50, 60],
            "ETFValue": [25, 30],
            "CashValue": [10, 15],
        })

        result = normalize_wide_csv(df)

        assert len(result) == 1
        assert result["StockValue"].iloc[0] == 200


class TestAddTotal:
    """Tests for add_total function."""

    def test_total_value_calculated(self, minimal_portfolio_df):
        """Test that TotalValue is correctly calculated."""
        result = add_total(minimal_portfolio_df)

        expected_first = 100000 + 50000 + 25000 + 10000
        assert result["TotalValue"].iloc[0] == expected_first

    def test_original_df_not_modified(self, minimal_portfolio_df):
        """Test that original DataFrame is not modified."""
        original_cols = set(minimal_portfolio_df.columns)
        add_total(minimal_portfolio_df)

        assert set(minimal_portfolio_df.columns) == original_cols


class TestTimeWeightedReturn:
    """Tests for time_weighted_return function."""

    def test_simple_return_no_cash_flows(self, minimal_portfolio_df):
        """Test TWR with no cash flows (simple return)."""
        df = add_total(minimal_portfolio_df)
        result = time_weighted_return(
            df,
            start=date(2024, 1, 1),
            end=date(2024, 2, 1),
        )

        assert "return_pct" in result
        assert "start_value" in result
        assert "end_value" in result
        assert result["start_value"] == 185000.0  # 100k + 50k + 25k + 10k
        assert result["return_pct"] > 0  # Portfolio grew

    def test_twr_with_cash_flows(self, portfolio_with_cash_flows):
        """Test TWR with cash additions and withdrawals."""
        df = add_total(portfolio_with_cash_flows)
        result = time_weighted_return(
            df,
            start=date(2024, 1, 1),
            end=date(2024, 4, 1),
        )

        assert result["sub_periods"] > 0  # Multiple sub-periods due to cash flows
        assert "return_pct" in result

    def test_twr_single_asset_class(self, minimal_portfolio_df):
        """Test TWR for a single asset class."""
        df = add_total(minimal_portfolio_df)
        result = time_weighted_return(
            df,
            start=date(2024, 1, 1),
            end=date(2024, 2, 1),
            asset_class="Stock",
        )

        assert result["start_value"] == 100000.0
        assert result["end_value"] == 105000.0
        assert result["return_pct"] == pytest.approx(5.0, rel=0.01)

    def test_empty_df_returns_zeros(self, empty_portfolio_df):
        """Test that empty DataFrame returns zero values."""
        result = time_weighted_return(
            empty_portfolio_df,
            start=date(2024, 1, 1),
            end=date(2024, 2, 1),
        )

        assert result["start_value"] == 0.0
        assert result["end_value"] == 0.0
        assert result["return_pct"] == 0.0

    def test_date_range_outside_data(self, minimal_portfolio_df):
        """Test TWR with date range outside available data."""
        df = add_total(minimal_portfolio_df)
        result = time_weighted_return(
            df,
            start=date(2020, 1, 1),
            end=date(2020, 12, 31),
        )

        assert result["return_pct"] == 0.0


class TestMaxDrawdown:
    """Tests for max_drawdown function."""

    def test_drawdown_calculation(self):
        """Test max drawdown is calculated correctly."""
        df = pd.DataFrame({
            "Date": [date(2024, 1, i) for i in range(1, 6)],
            "StockValue": [100, 110, 90, 95, 100],  # Peak at 110, trough at 90
            "BondValue": [0, 0, 0, 0, 0],
            "ETFValue": [0, 0, 0, 0, 0],
            "CashValue": [0, 0, 0, 0, 0],
        })

        result = max_drawdown(df, date(2024, 1, 1), date(2024, 1, 5))

        # Max drawdown: (90 - 110) / 110 = -18.18%
        assert result < 0
        assert result == pytest.approx(-18.18, rel=0.01)

    def test_no_drawdown_uptrend(self, minimal_portfolio_df):
        """Test no significant drawdown in uptrend."""
        df = add_total(minimal_portfolio_df)
        result = max_drawdown(df, date(2024, 1, 1), date(2024, 2, 1))

        # Minimal or no drawdown in consistently growing portfolio
        assert result >= -5.0  # Small tolerance for fluctuations

    def test_empty_df_returns_zero(self, empty_portfolio_df):
        """Test empty DataFrame returns zero drawdown."""
        result = max_drawdown(empty_portfolio_df, date(2024, 1, 1), date(2024, 2, 1))
        assert result == 0.0


class TestAllocationOnDate:
    """Tests for allocation_on_date function."""

    def test_allocation_percentages(self, minimal_portfolio_df):
        """Test that allocation percentages sum to 1."""
        alloc = allocation_on_date(minimal_portfolio_df, date(2024, 1, 1))

        total = sum(alloc.values())
        assert total == pytest.approx(1.0, rel=0.001)

    def test_allocation_values(self, minimal_portfolio_df):
        """Test specific allocation values."""
        alloc = allocation_on_date(minimal_portfolio_df, date(2024, 1, 1))

        # Total = 100k + 50k + 25k + 10k = 185k
        assert alloc["Stock"] == pytest.approx(100000 / 185000, rel=0.001)
        assert alloc["Bond"] == pytest.approx(50000 / 185000, rel=0.001)
        assert alloc["ETF"] == pytest.approx(25000 / 185000, rel=0.001)
        assert alloc["Cash"] == pytest.approx(10000 / 185000, rel=0.001)

    def test_date_not_found_uses_previous(self, minimal_portfolio_df):
        """Test that missing date uses previous available date."""
        # Date between 2024-01-01 and 2024-01-15
        alloc = allocation_on_date(minimal_portfolio_df, date(2024, 1, 10))

        # Should use 2024-01-01 data
        assert alloc["Stock"] == pytest.approx(100000 / 185000, rel=0.001)

    def test_empty_df_returns_zeros(self, empty_portfolio_df):
        """Test empty DataFrame returns zero allocations."""
        alloc = allocation_on_date(empty_portfolio_df, date(2024, 1, 1))

        assert all(v == 0.0 for v in alloc.values())


class TestHeldAssetClassesRecent:
    """Tests for held_asset_classes_recent function."""

    def test_all_assets_held(self, minimal_portfolio_df):
        """Test all asset classes with non-zero values are returned."""
        result = held_asset_classes_recent(minimal_portfolio_df)

        assert "Stock" in result
        assert "Bond" in result
        assert "ETF" in result
        assert "Cash" in result

    def test_zero_asset_excluded(self):
        """Test that asset class with zero value is excluded."""
        df = pd.DataFrame({
            "Date": [date(2024, 1, 1)],
            "StockValue": [100000],
            "BondValue": [0],  # Zero
            "ETFValue": [25000],
            "CashValue": [10000],
        })

        result = held_asset_classes_recent(df)

        assert "Stock" in result
        assert "Bond" not in result
        assert "ETF" in result
        assert "Cash" in result

    def test_empty_df_returns_empty_list(self, empty_portfolio_df):
        """Test empty DataFrame returns empty list."""
        result = held_asset_classes_recent(empty_portfolio_df)
        assert result == []

    def test_lookback_days_parameter(self, sample_portfolio_df):
        """Test lookback_days parameter limits data range."""
        result_short = held_asset_classes_recent(sample_portfolio_df, lookback_days=7)
        result_long = held_asset_classes_recent(sample_portfolio_df, lookback_days=30)

        # Both should return same asset classes for this dataset
        assert set(result_short) == set(result_long)
