# AI Portfolio Chatbot

A read-only portfolio management assistant built with **Streamlit**, **LangGraph**, and **LangChain**. Upload daily portfolio CSVs, ask natural-language questions, and get performance analytics, risk profiling, rebalance suggestions, and live market trend summaries — all without the agent ever modifying your data.

## Features

- **Performance Analytics** — Time-weighted return and max drawdown over any date range, per asset class or total portfolio
- **Risk Profiling** — 4-question interactive questionnaire or instant inference from current allocation (Stock/Bond/ETF/Cash ratios)
- **Rebalance Suggestions** — Target allocation for Conservative, Moderate, or Aggressive profiles with step-by-step guidance
- **Live Market Trends** — Tavily-powered web search with LLM-summarized bullet points and inline source links
- **Strictly Read-Only** — The agent cannot create, update, or delete any data; portfolio changes happen via CSV upload only

## Agentic Architecture (LangGraph)

The assistant is powered by a multi-node LangGraph state machine with 9 agentic AI features:

| # | Feature | Description |
|---|---------|-------------|
| 1 | **Checkpointing** | `MemorySaver` for conversation persistence across graph invocations |
| 2 | **Human-in-the-Loop** | `interrupt()` / `Command(resume=...)` pauses the graph for user input during the risk questionnaire |
| 3 | **Streaming** | `graph.stream(stream_mode="updates")` for real-time node-by-node progress in the UI |
| 4 | **Reflection** | Judge scores responses; low scores trigger a feedback loop back to the agent (max 1 cycle) |
| 5 | **Subgraph Nodes** | Risk questionnaire and market analysis run as dedicated nodes with encapsulated logic |
| 6 | **Parallel Tool Execution** | `ThreadPoolExecutor` runs independent tool calls concurrently |
| 7 | **Error Recovery** | Per-tool `try/except` with retry logic in the market analysis node |
| 8 | **Planning (CoT)** | Planner node classifies intent via rule-based matching or LLM fallback before the agent acts |
| 9 | **Observability** | Structured logging, `node_timer` context manager, and optional LangSmith tracing |

### Graph Topology

```
planner -> agent -> [route_from_agent]
                      -> tools              (parallel + error recovery) -> agent
                      -> risk_questionnaire (HITL with interrupt)       -> agent
                      -> market_analysis    (retry error recovery)      -> agent
                      -> judge              (no tools needed)
judge -> [reflect_or_pass]
           -> reflect -> agent   (self-correction, max 1 cycle)
           -> policy  -> END
```

## Project Structure

```
ai-portfolio-chatbot/
├── app.py                       # Streamlit entry point
├── observability.py             # Structured logging & LangSmith tracing
├── core/                        # Pure domain logic (no LLM dependencies)
│   ├── analytics.py             # TWR, max drawdown, allocation
│   ├── risk.py                  # Risk profiling & scoring
│   ├── storage.py               # SQLite persistence layer
│   └── prompts.py               # System prompts & risk questions
├── agent/                       # LangGraph agent architecture
│   ├── state.py                 # GraphState, ChatState, schemas, constants
│   ├── nodes.py                 # 8 node creation functions
│   ├── routing.py               # Conditional edge routing
│   ├── builder.py               # build_graph()
│   ├── tools.py                 # Tool definitions & handlers
│   ├── judge.py                 # Response quality scoring & rewrite
│   ├── policy.py                # Policy guardrails
│   └── trends.py                # Tavily market trend search & summarization
├── data/
│   └── samples/                 # Sample portfolio CSVs for testing
├── pyproject.toml               # Project metadata & dependencies
└── uv.lock                      # Locked dependency versions
```

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- API keys:
  - **OpenAI** — for the LLM agent, judge, and policy models
  - **Tavily** — for live market trend search
  - **LangSmith** *(optional)* — for full trace observability

## Setup

1. **Clone the repository**

```bash
git clone <repo-url>
cd ai-portfolio-chatbot
```

2. **Install dependencies**

```bash
uv sync
```

3. **Configure environment variables**

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...

# Optional: model overrides (defaults to gpt-4o-mini)
OPENAI_LLM_MODEL=gpt-4o-mini
OPENAI_JUDGE_MODEL=gpt-4o-mini
OPENAI_POLICY_MODEL=gpt-4o-mini

# Optional: LangSmith tracing
LANGCHAIN_API_KEY=lsv2-...
LANGCHAIN_PROJECT=ai-portfolio-chatbot

# Optional: log level
LOG_LEVEL=INFO
```

4. **Run the app**

```bash
uv run streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Usage

1. **Upload portfolio data** — Use the sidebar to upload a CSV with columns: `Date`, `StockValue`, `BondValue`, `ETFValue`, `CashValue` (optional: `CashAddition`, `CashWithdrawal`). Sample CSVs are in `data/samples/`.

2. **Ask questions in natural language**, for example:
   - *"What's my portfolio performance?"*
   - *"What is my current risk profile?"*
   - *"Run the risk questionnaire"*
   - *"Show me the latest market trends"*
   - *"Suggest a rebalance to aggressive"*
   - *"What can you do?"*

3. **View the sidebar** for a portfolio snapshot, CSV upload, and a button to clear data.

## CSV Format

| Column | Required | Description |
|--------|----------|-------------|
| `Date` | Yes | Date in any parseable format (e.g. `2024-01-15`) |
| `StockValue` | Yes | Total stock holdings value |
| `BondValue` | Yes | Total bond holdings value |
| `ETFValue` | Yes | Total ETF holdings value |
| `CashValue` | Yes | Total cash holdings value |
| `CashAddition` | No | Cash deposited on this date (for accurate TWR) |
| `CashWithdrawal` | No | Cash withdrawn on this date (for accurate TWR) |

`TotalValue` is computed automatically on load.

## Tech Stack

- **Frontend**: Streamlit
- **Agent Framework**: LangGraph + LangChain
- **LLM**: OpenAI (GPT-4o-mini default, configurable)
- **Search**: Tavily for live market trends
- **Database**: SQLite (WAL mode) for portfolio storage
- **Observability**: Python logging + LangSmith tracing

## Constraints

- The agent is **strictly read-only** — it cannot modify portfolio data, risk profiles, or any stored records
- All financial numbers must come from tool results — the agent never fabricates figures
- The agent is **not a financial advisor** — all outputs are educational and informational

## License

This project is for educational and portfolio demonstration purposes.
