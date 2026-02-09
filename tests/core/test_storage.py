"""Tests for core/storage.py - SQLite storage operations."""
from datetime import date

import pandas as pd
import pytest

from core.storage import SQLiteStore


class TestSQLiteStoreInit:
    """Tests for SQLiteStore initialization."""

    def test_init_creates_tables(self, temp_db: SQLiteStore):
        """Test that init creates required tables."""
        with temp_db.connect() as con:
            # Check portfolio_daily table exists
            result = con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='portfolio_daily'"
            ).fetchone()
            assert result is not None

            # Check settings table exists
            result = con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='settings'"
            ).fetchone()
            assert result is not None

    def test_init_idempotent(self, temp_db: SQLiteStore):
        """Test that calling init multiple times is safe."""
        temp_db.init()
        temp_db.init()

        # Should still work
        df = temp_db.load_all()
        assert isinstance(df, pd.DataFrame)


class TestUpsertDay:
    """Tests for upsert_day method."""

    def test_insert_new_day(self, temp_db: SQLiteStore):
        """Test inserting a new day's data."""
        temp_db.upsert_day(
            d=date(2024, 1, 1),
            stock=100000,
            bond=50000,
            cash=10000,
            etf=25000,
        )

        result = temp_db.get_day(date(2024, 1, 1))
        assert result is not None
        assert result["stock_value"] == 100000
        assert result["bond_value"] == 50000
        assert result["cash_value"] == 10000
        assert result["etf_value"] == 25000

    def test_update_existing_day(self, temp_db: SQLiteStore):
        """Test updating an existing day's data."""
        temp_db.upsert_day(d=date(2024, 1, 1), stock=100000, bond=50000, cash=10000, etf=25000)
        temp_db.upsert_day(d=date(2024, 1, 1), stock=110000, bond=55000, cash=12000, etf=27000)

        result = temp_db.get_day(date(2024, 1, 1))
        assert result["stock_value"] == 110000
        assert result["bond_value"] == 55000

    def test_total_value_calculated(self, temp_db: SQLiteStore):
        """Test that total_value is automatically calculated."""
        temp_db.upsert_day(d=date(2024, 1, 1), stock=100, bond=50, cash=25, etf=25)

        result = temp_db.get_day(date(2024, 1, 1))
        assert result["total_value"] == 200  # 100 + 50 + 25 + 25

    def test_cash_flows_stored(self, temp_db: SQLiteStore):
        """Test that cash additions and withdrawals are stored."""
        temp_db.upsert_day(
            d=date(2024, 1, 1),
            stock=100000,
            bond=50000,
            cash=15000,
            etf=25000,
            cash_addition=5000,
            cash_withdrawal=0,
        )

        result = temp_db.get_day(date(2024, 1, 1))
        assert result["cash_addition"] == 5000
        assert result["cash_withdrawal"] == 0


class TestDeleteDay:
    """Tests for delete_day method."""

    def test_delete_existing_day(self, temp_db: SQLiteStore):
        """Test deleting an existing day."""
        temp_db.upsert_day(d=date(2024, 1, 1), stock=100, bond=50, cash=25, etf=25)
        temp_db.delete_day(date(2024, 1, 1))

        result = temp_db.get_day(date(2024, 1, 1))
        assert result is None

    def test_delete_nonexistent_day_no_error(self, temp_db: SQLiteStore):
        """Test that deleting a non-existent day doesn't raise."""
        temp_db.delete_day(date(2024, 1, 1))  # Should not raise


class TestClearPortfolio:
    """Tests for clear_portfolio method."""

    def test_clear_removes_all_data(self, populated_db: SQLiteStore):
        """Test that clear_portfolio removes all portfolio data."""
        populated_db.clear_portfolio()

        df = populated_db.load_all()
        assert df.empty

    def test_clear_preserves_settings(self, populated_db: SQLiteStore):
        """Test that clear_portfolio preserves settings."""
        populated_db.set_risk_profile("Aggressive")
        populated_db.clear_portfolio()

        profile = populated_db.get_risk_profile()
        assert profile == "Aggressive"


class TestGetDay:
    """Tests for get_day method."""

    def test_get_existing_day(self, populated_db: SQLiteStore):
        """Test getting an existing day's data."""
        result = populated_db.get_day(date(2024, 1, 1))

        assert result is not None
        assert "stock_value" in result
        assert "bond_value" in result

    def test_get_nonexistent_day(self, temp_db: SQLiteStore):
        """Test getting a non-existent day returns None."""
        result = temp_db.get_day(date(2024, 1, 1))
        assert result is None


class TestLoadAll:
    """Tests for load_all method."""

    def test_load_all_returns_dataframe(self, populated_db: SQLiteStore):
        """Test that load_all returns a DataFrame."""
        df = populated_db.load_all()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3  # minimal_portfolio_df has 3 rows

    def test_load_all_has_required_columns(self, populated_db: SQLiteStore):
        """Test that loaded DataFrame has required columns."""
        df = populated_db.load_all()

        required_cols = ["Date", "StockValue", "BondValue", "ETFValue", "CashValue", "TotalValue"]
        for col in required_cols:
            assert col in df.columns

    def test_load_all_sorted_by_date(self, populated_db: SQLiteStore):
        """Test that results are sorted by date."""
        df = populated_db.load_all()

        dates = list(df["Date"])
        assert dates == sorted(dates)

    def test_load_all_empty_db(self, temp_db: SQLiteStore):
        """Test load_all on empty database."""
        df = temp_db.load_all()

        assert isinstance(df, pd.DataFrame)
        assert df.empty


class TestLoadRange:
    """Tests for load_range method."""

    def test_load_range_filters_correctly(self, populated_db: SQLiteStore):
        """Test that load_range filters by date range."""
        df = populated_db.load_range(date(2024, 1, 1), date(2024, 1, 15))

        assert len(df) == 2  # Jan 1 and Jan 15
        assert all(d >= date(2024, 1, 1) for d in df["Date"])
        assert all(d <= date(2024, 1, 15) for d in df["Date"])

    def test_load_range_no_data_in_range(self, populated_db: SQLiteStore):
        """Test load_range with no data in range."""
        df = populated_db.load_range(date(2020, 1, 1), date(2020, 12, 31))

        assert df.empty


class TestSettings:
    """Tests for settings methods."""

    def test_set_and_get_setting(self, temp_db: SQLiteStore):
        """Test setting and getting a setting."""
        temp_db.set_setting("test_key", "test_value")

        result = temp_db.get_setting("test_key")
        assert result == "test_value"

    def test_get_nonexistent_setting(self, temp_db: SQLiteStore):
        """Test getting a non-existent setting returns None."""
        result = temp_db.get_setting("nonexistent")
        assert result is None

    def test_update_setting(self, temp_db: SQLiteStore):
        """Test updating an existing setting."""
        temp_db.set_setting("key", "value1")
        temp_db.set_setting("key", "value2")

        result = temp_db.get_setting("key")
        assert result == "value2"


class TestRiskProfile:
    """Tests for risk profile methods."""

    def test_set_and_get_risk_profile(self, temp_db: SQLiteStore):
        """Test setting and getting risk profile."""
        temp_db.set_risk_profile("Aggressive")

        result = temp_db.get_risk_profile()
        assert result == "Aggressive"

    def test_get_risk_profile_not_set(self, temp_db: SQLiteStore):
        """Test getting risk profile when not set."""
        result = temp_db.get_risk_profile()
        assert result is None

    def test_update_risk_profile(self, temp_db: SQLiteStore):
        """Test updating risk profile."""
        temp_db.set_risk_profile("Conservative")
        temp_db.set_risk_profile("Moderate")

        result = temp_db.get_risk_profile()
        assert result == "Moderate"
