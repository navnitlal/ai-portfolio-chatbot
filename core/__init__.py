"""Core domain logic: analytics, risk profiling, storage, and prompts.

No LLM or LangGraph dependencies — pure business logic and data access.
"""
from core.analytics import (
    normalize_wide_csv,
    allocation_on_date,
    max_drawdown,
    time_weighted_return,
    held_asset_classes_recent,
)
from core.risk import (
    score_risk_answers,
    infer_risk_from_allocation,
    TARGET_ALLOCATION_BY_RISK,
)
from core.storage import SQLiteStore
