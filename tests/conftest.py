"""Shared pytest fixtures for ai-portfolio-chatbot tests."""
from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional
from unittest.mock import MagicMock

import pandas as pd
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.storage import SQLiteStore


# =============================================================================
# Data Fixtures
# =============================================================================


@pytest.fixture
def sample_portfolio_df() -> pd.DataFrame:
    """Load sample portfolio data from CSV."""
    csv_path = PROJECT_ROOT / "data" / "samples" / "sample_portfolio_1.csv"
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    return df


@pytest.fixture
def minimal_portfolio_df() -> pd.DataFrame:
    """Minimal portfolio data for quick tests."""
    return pd.DataFrame({
        "Date": [date(2024, 1, 1), date(2024, 1, 15), date(2024, 2, 1)],
        "StockValue": [100000.0, 102000.0, 105000.0],
        "BondValue": [50000.0, 50500.0, 51000.0],
        "ETFValue": [25000.0, 25250.0, 25500.0],
        "CashValue": [10000.0, 10000.0, 10000.0],
        "CashAddition": [0.0, 0.0, 0.0],
        "CashWithdrawal": [0.0, 0.0, 0.0],
    })


@pytest.fixture
def portfolio_with_cash_flows() -> pd.DataFrame:
    """Portfolio data with cash additions and withdrawals."""
    return pd.DataFrame({
        "Date": [
            date(2024, 1, 1),
            date(2024, 2, 1),
            date(2024, 3, 1),
            date(2024, 4, 1),
        ],
        "StockValue": [100000.0, 105000.0, 108000.0, 112000.0],
        "BondValue": [50000.0, 51000.0, 52000.0, 53000.0],
        "ETFValue": [20000.0, 20500.0, 21000.0, 21500.0],
        "CashValue": [10000.0, 15000.0, 13000.0, 14000.0],
        "CashAddition": [0.0, 5000.0, 0.0, 2000.0],
        "CashWithdrawal": [0.0, 0.0, 2000.0, 0.0],
    })


@pytest.fixture
def empty_portfolio_df() -> pd.DataFrame:
    """Empty portfolio DataFrame."""
    return pd.DataFrame(columns=[
        "Date", "StockValue", "BondValue", "ETFValue", "CashValue",
        "CashAddition", "CashWithdrawal",
    ])


# =============================================================================
# Storage Fixtures
# =============================================================================


@pytest.fixture
def temp_db(tmp_path) -> Generator[SQLiteStore, None, None]:
    """Create a temporary SQLite database for testing."""
    db_path = tmp_path / "test_portfolio.db"
    store = SQLiteStore(str(db_path))
    store.init()
    yield store


@pytest.fixture
def populated_db(temp_db: SQLiteStore, minimal_portfolio_df: pd.DataFrame) -> SQLiteStore:
    """Temporary database populated with minimal portfolio data."""
    for _, row in minimal_portfolio_df.iterrows():
        temp_db.upsert_day(
            d=row["Date"],
            stock=row["StockValue"],
            bond=row["BondValue"],
            cash=row["CashValue"],
            etf=row["ETFValue"],
            cash_addition=row.get("CashAddition", 0.0),
            cash_withdrawal=row.get("CashWithdrawal", 0.0),
        )
    return temp_db


# =============================================================================
# LLM Mock Fixtures
# =============================================================================


class MockLLMResponse:
    """Mock LLM response object."""

    def __init__(self, content: str, tool_calls: Optional[List[Dict]] = None):
        self.content = content
        self.tool_calls = tool_calls or []

    def model_dump(self) -> Dict[str, Any]:
        return {"content": self.content, "tool_calls": self.tool_calls}


class MockLLM:
    """Mock LLM for testing without API calls."""

    def __init__(self, responses: Optional[List[str]] = None):
        self._responses = responses or ["This is a mock response."]
        self._call_count = 0
        self._last_messages: List = []

    def invoke(self, messages: List) -> MockLLMResponse:
        """Mock invoke method."""
        self._last_messages = messages
        response = self._responses[min(self._call_count, len(self._responses) - 1)]
        self._call_count += 1
        return MockLLMResponse(content=response)

    def bind_tools(self, tools: List) -> "MockLLM":
        """Mock bind_tools - returns self."""
        return self

    def with_retry(self, **kwargs) -> "MockLLM":
        """Mock with_retry - returns self."""
        return self

    def with_structured_output(self, schema) -> "MockLLM":
        """Mock structured output - returns self."""
        return self

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def last_messages(self) -> List:
        return self._last_messages


class MockLLMWithToolCalls(MockLLM):
    """Mock LLM that returns tool calls."""

    def __init__(self, tool_calls: List[Dict[str, Any]]):
        super().__init__()
        self._tool_calls = tool_calls

    def invoke(self, messages: List) -> MockLLMResponse:
        self._last_messages = messages
        self._call_count += 1
        return MockLLMResponse(content="", tool_calls=self._tool_calls)


@pytest.fixture
def mock_llm() -> MockLLM:
    """Basic mock LLM fixture."""
    return MockLLM()


@pytest.fixture
def mock_llm_with_responses() -> callable:
    """Factory fixture for creating mock LLM with specific responses."""
    def _create(responses: List[str]) -> MockLLM:
        return MockLLM(responses=responses)
    return _create


@pytest.fixture
def mock_llm_with_tool_calls() -> callable:
    """Factory fixture for creating mock LLM that returns tool calls."""
    def _create(tool_calls: List[Dict[str, Any]]) -> MockLLMWithToolCalls:
        return MockLLMWithToolCalls(tool_calls=tool_calls)
    return _create


# =============================================================================
# Agent State Fixtures
# =============================================================================


@pytest.fixture
def empty_state() -> Dict[str, Any]:
    """Empty agent state."""
    return {
        "messages": [],
        "plan": "",
        "reflection_count": 0,
        "risk_answers": {},
        "risk_profile": None,
        "pending_kind": "none",
        "pending_payload": {},
    }


@pytest.fixture
def state_with_risk_answers() -> Dict[str, Any]:
    """Agent state with completed risk questionnaire."""
    return {
        "messages": [],
        "plan": "",
        "reflection_count": 0,
        "risk_answers": {
            "1": "8+ years",
            "2": "high stability",
            "3": "maximum growth",
            "4": "buy more",
        },
        "risk_profile": "Aggressive",
        "pending_kind": "none",
        "pending_payload": {},
    }


# =============================================================================
# Config Fixtures
# =============================================================================


@pytest.fixture
def agent_config(populated_db: SQLiteStore, mock_llm: MockLLM) -> Dict[str, Any]:
    """Configuration dictionary for agent nodes."""
    return {
        "configurable": {
            "store": populated_db,
            "main_llm": mock_llm,
            "judge_llm": mock_llm,
            "policy_llm": mock_llm,
        }
    }


@pytest.fixture
def config_without_store(mock_llm: MockLLM) -> Dict[str, Any]:
    """Configuration without a store (no portfolio data)."""
    return {
        "configurable": {
            "store": None,
            "main_llm": mock_llm,
            "judge_llm": mock_llm,
            "policy_llm": mock_llm,
        }
    }


# =============================================================================
# Integration Test Fixtures
# =============================================================================


@pytest.fixture
def real_llm():
    """
    Real LLM for integration tests.
    Requires OPENAI_API_KEY environment variable.
    Skip if not available.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set - skipping integration test")

    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


# =============================================================================
# Utility Fixtures
# =============================================================================


@pytest.fixture
def assert_no_api_calls(mocker):
    """Fixture that fails if any real API calls are made."""
    import httpx

    original_send = httpx.Client.send

    def guarded_send(self, request, **kwargs):
        if "api.openai.com" in str(request.url):
            pytest.fail(f"Unexpected API call to: {request.url}")
        return original_send(self, request, **kwargs)

    mocker.patch.object(httpx.Client, "send", guarded_send)
