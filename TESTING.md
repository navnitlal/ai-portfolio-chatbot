# Testing Guide

This document describes the automated test suite for the AI Portfolio Chatbot project.

## Overview

The project uses **pytest** as the testing framework with the following capabilities:
- Unit tests for core business logic
- Integration tests for agent components
- Mocked LLM response testing
- Real LLM integration tests (optional, requires API key)
- Code coverage reporting

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and test utilities
├── core/                    # Core module tests
│   ├── test_analytics.py    # Portfolio analytics (TWR, drawdown, allocation)
│   ├── test_risk.py         # Risk profiling and scoring
│   └── test_storage.py      # SQLite database operations
└── agent/                   # Agent module tests
    ├── test_tools.py        # Tool handler functions
    ├── test_routing.py      # Graph routing logic
    ├── test_nodes.py        # Graph node functions
    ├── test_state.py        # State management
    ├── test_judge.py        # Response quality scoring
    └── test_llm_responses.py # LLM response patterns
```

## Installation

Install test dependencies:

```bash
uv sync --extra dev
```

This installs:
- pytest
- pytest-asyncio
- pytest-cov (coverage)
- pytest-mock (mocking)

## Running Tests

### Run All Tests

```bash
uv run pytest
```

### Run with Verbose Output

```bash
uv run pytest -v
```

### Run Specific Test File

```bash
uv run pytest tests/core/test_analytics.py
```

### Run Specific Test Class

```bash
uv run pytest tests/agent/test_tools.py::TestGetPerformanceHandler
```

### Run Specific Test Function

```bash
uv run pytest tests/core/test_risk.py::TestScoreRiskAnswers::test_aggressive_answers
```

## Test Categories

Tests are marked with categories for selective execution:

| Marker | Description |
|--------|-------------|
| `unit` | Unit tests with no external dependencies |
| `integration` | Integration tests (may use external services) |
| `llm` | Tests that interact with LLM (mocked or real) |
| `slow` | Slow-running tests |

### Run Only Unit Tests

```bash
uv run pytest -m "not integration"
```

### Run Integration Tests

Requires `OPENAI_API_KEY` environment variable:

```bash
export OPENAI_API_KEY=your-api-key
uv run pytest -m integration
```

## Code Coverage

### Generate Coverage Report

```bash
uv run pytest --cov=core --cov=agent --cov-report=term-missing
```

### Generate HTML Coverage Report

```bash
uv run pytest --cov=core --cov=agent --cov-report=html
```

Then open `htmlcov/index.html` in a browser.

## Test Descriptions

### Core Module Tests

#### `test_analytics.py`
- **TestNormalizeWideCsv**: CSV validation and normalization
- **TestAddTotal**: Total value calculation
- **TestTimeWeightedReturn**: TWR calculation with/without cash flows
- **TestMaxDrawdown**: Maximum drawdown calculation
- **TestAllocationOnDate**: Portfolio allocation percentages
- **TestHeldAssetClassesRecent**: Asset class detection

#### `test_risk.py`
- **TestInferRiskFromAllocation**: Risk inference from allocation ratios
- **TestScoreRiskAnswers**: Risk questionnaire scoring
- **TestTargetAllocationByRisk**: Target allocation constants

#### `test_storage.py`
- **TestSQLiteStoreInit**: Database initialization
- **TestUpsertDay**: Insert/update daily data
- **TestDeleteDay**: Delete operations
- **TestClearPortfolio**: Clear all portfolio data
- **TestLoadAll/TestLoadRange**: Data retrieval
- **TestSettings/TestRiskProfile**: Settings management

### Agent Module Tests

#### `test_tools.py`
- **TestGetBindingTools**: Tool registration
- **TestRunTool**: Tool execution
- **Test*Handler**: Individual tool handler tests

#### `test_routing.py`
- **TestRouteFromAgent**: Agent routing decisions
- **TestReflectOrPass**: Reflection loop control

#### `test_nodes.py`
- **TestPlannerNode**: Chain-of-thought planning
- **TestAgentNode**: LLM invocation
- **TestToolsNode**: Tool execution
- **TestJudgeNode**: Response quality scoring
- **TestReflectNode**: Self-correction
- **TestPolicyNode**: Policy enforcement

#### `test_state.py`
- **TestConstants**: Configuration constants
- **TestJudgeVerdict/TestPolicyVerdict**: Schema validation
- **TestChatState**: State dataclass
- **TestStateConversion**: State serialization
- **TestExtractToolCallInfo**: Tool call parsing

#### `test_judge.py`
- **TestExtractJson**: JSON extraction from LLM output
- **TestJudgeAndRewrite**: Response quality and rewriting

#### `test_llm_responses.py`
- **TestLLMResponsePatterns**: Mock LLM response handling
- **TestJudgeLLMResponses**: Judge LLM integration
- **TestPlannerLLMResponses**: Planner LLM integration
- **TestLLMResponseQuality**: Response quality validation
- **TestLLMIntegration**: Real LLM tests (requires API key)
- **TestFullGraphIntegration**: End-to-end graph tests

## Writing New Tests

### Using Fixtures

Common fixtures are defined in `conftest.py`:

```python
def test_with_portfolio_data(sample_portfolio_df):
    """Use sample portfolio DataFrame."""
    assert len(sample_portfolio_df) > 0

def test_with_database(temp_db):
    """Use temporary SQLite database."""
    temp_db.upsert_day(date(2024, 1, 1), 100, 50, 25, 25)

def test_with_mock_llm(mock_llm):
    """Use mock LLM that returns predefined responses."""
    response = mock_llm.invoke([])
    assert response.content
```

### Mocking LLM Responses

```python
def test_custom_llm_response(mock_llm_with_responses):
    llm = mock_llm_with_responses(["Custom response 1", "Custom response 2"])

    result1 = llm.invoke([])
    result2 = llm.invoke([])

    assert result1.content == "Custom response 1"
    assert result2.content == "Custom response 2"
```

### Testing Tool Calls

```python
def test_tool_with_config(agent_config):
    result, updates = run_tool("get_performance", {
        "start_date": "2024-01-01",
        "end_date": "2024-02-01"
    }, {}, agent_config)

    assert "return" in result.lower()
```

## Continuous Integration

Add to your CI pipeline:

```yaml
# GitHub Actions example
- name: Run tests
  run: |
    uv sync --extra dev
    uv run pytest --cov=core --cov=agent --cov-report=xml

- name: Upload coverage
  uses: codecov/codecov-action@v3
  with:
    file: coverage.xml
```

## Troubleshooting

### Tests Not Found

Ensure test files follow naming convention: `test_*.py`

### Import Errors

Run from project root directory:

```bash
cd /path/to/ai-portfolio-chatbot
uv run pytest
```

### Integration Tests Skipped

Set the OpenAI API key:

```bash
export OPENAI_API_KEY=sk-...
uv run pytest -m integration
```

### Database Warnings

ResourceWarning messages about unclosed databases are benign in tests and can be ignored.
