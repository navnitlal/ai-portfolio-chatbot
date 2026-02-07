"""Graph state definitions, schemas, constants, and conversion helpers.

This module is the single source of truth for all state types used across
the agent package, eliminating circular imports between graph/judge/policy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any, Dict, List, Optional, Literal, TypedDict

from pydantic import BaseModel, Field as PydanticField

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

# Checkpointing — graceful fallback
try:
    from langgraph.checkpoint.memory import MemorySaver
except ImportError:
    MemorySaver = None  # type: ignore[assignment,misc]

# HITL — graceful fallback
try:
    from langgraph.types import interrupt, Command
except ImportError:
    interrupt = None  # type: ignore[assignment]
    Command = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_MESSAGES_IN_CONTEXT = 24
MAX_REFLECTIONS = 1
REFLECTION_THRESHOLD = 7  # Judge score below this triggers reflection

PendingKind = Literal["none", "apply_edit", "apply_risk_change"]


# ---------------------------------------------------------------------------
# Structured output schemas for judge & policy
# ---------------------------------------------------------------------------

class JudgeVerdict(BaseModel):
    score: int = PydanticField(description="Quality score 1-10")
    issues: List[str] = PydanticField(default_factory=list)
    rewrite_needed: bool = PydanticField(default=False)


class PolicyVerdict(BaseModel):
    ok: bool = PydanticField(description="Whether response passes policy")
    issues: List[str] = PydanticField(default_factory=list)
    safe_text: str = PydanticField(default="")


# ---------------------------------------------------------------------------
# Graph state — single canonical type
# ---------------------------------------------------------------------------

class GraphState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    risk_answers: Dict[str, str]
    risk_profile: Optional[str]
    pending_kind: str
    pending_payload: Dict[str, Any]
    last_judge: Dict[str, Any]
    last_policy: Dict[str, Any]
    already_suggested_upload_for_edit: bool
    # Execution flags — set by tool handlers, consumed by policy node
    did_persist_change: bool
    did_use_tavily: bool
    did_compute_metrics: bool
    # Agentic features
    plan: str               # Chain-of-thought plan from planner
    reflection_count: int   # Number of reflection cycles (max 1)


@dataclass
class ChatState:
    """Convenience wrapper for Streamlit session_state.
    Maps 1:1 with GraphState; use state_to_dict / dict_to_state to convert."""
    messages: List[BaseMessage] = field(default_factory=list)
    risk_answers: Dict[str, str] = field(default_factory=dict)
    risk_profile: Optional[str] = None
    pending_kind: PendingKind = "none"
    pending_payload: Dict[str, Any] = field(default_factory=dict)
    last_judge: Dict[str, Any] = field(default_factory=dict)
    last_policy: Dict[str, Any] = field(default_factory=dict)
    already_suggested_upload_for_edit: bool = False
    did_persist_change: bool = False
    did_use_tavily: bool = False
    did_compute_metrics: bool = False
    plan: str = ""
    reflection_count: int = 0


_STATE_FIELDS = [
    "messages", "risk_answers", "risk_profile", "pending_kind", "pending_payload",
    "last_judge", "last_policy", "already_suggested_upload_for_edit",
    "did_persist_change", "did_use_tavily", "did_compute_metrics",
    "plan", "reflection_count",
]

_STATE_DEFAULTS: Dict[str, Any] = {
    "messages": [], "risk_answers": {}, "risk_profile": None,
    "pending_kind": "none", "pending_payload": {}, "last_judge": {},
    "last_policy": {}, "already_suggested_upload_for_edit": False,
    "did_persist_change": False, "did_use_tavily": False, "did_compute_metrics": False,
    "plan": "", "reflection_count": 0,
}


def state_to_dict(state: ChatState) -> Dict[str, Any]:
    return {f: getattr(state, f) for f in _STATE_FIELDS}


def dict_to_state(d: Dict[str, Any]) -> ChatState:
    return ChatState(**{f: d.get(f, _STATE_DEFAULTS.get(f)) for f in _STATE_FIELDS})


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def extract_tool_call_info(tc) -> tuple:
    """Normalize a tool call (dict or object) to (name, args, id)."""
    if isinstance(tc, dict):
        return tc.get("name", ""), tc.get("args", {}) or {}, tc.get("id", "")
    return getattr(tc, "name", ""), getattr(tc, "args", {}) or {}, getattr(tc, "id", "")
