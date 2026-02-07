from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from agent.state import PolicyVerdict

POLICY_PROMPT = """You are a policy guardrail agent for a portfolio chatbot.

Goal: Ensure the assistant is strictly read-only — it must NOT claim to have performed actions, modified data, or used data sources unless the application actually did so. It must NOT fabricate financial figures. It must NOT offer or promise to make any changes.

Hard rules:
- The assistant is READ-ONLY. It cannot save, update, delete, apply, or persist anything. Flag any text that claims or offers to do so.
- Do not claim you loaded, saved, updated, deleted, or persisted anything unless did_persist_change=true.
- Do not present web/trend facts as if searched unless did_use_tavily=true.
- Do not present performance/allocation numbers unless did_compute_metrics=true.
- Do not fabricate, estimate, or hallucinate financial numbers (returns, allocations, drawdowns). Every number must come from a tool result.
- Do not offer or promise to apply changes (e.g. "I can update your profile", "Shall I apply?", "I'll save that").
- If the user asked for metrics/trends but prerequisites are missing (e.g. no data for metrics), respond by asking for what is needed — do not make up values.

Given:
- assistant_text (draft)
- execution_flags (json)

Return ONLY JSON:
{
  "ok": true|false,
  "issues": ["..."],
  "safe_text": "..."  // compliant rewrite if needed; keep it short and conversational, no command reference or manual
}
"""


@dataclass
class ExecutionFlags:
    did_persist_change: bool = False
    did_use_tavily: bool = False
    did_compute_metrics: bool = False

    has_portfolio_data: bool = False
    has_data_in_requested_range: Optional[bool] = None

    user_consented_tavily: bool = False
    user_requested_metrics: bool = False
    user_requested_trends: bool = False


def _detect_issues(text: str, flags: ExecutionFlags) -> List[str]:
    t = (text or "").lower()
    issues: List[str] = []

    # 1. Claims of data modification
    persistence_phrases = [
        "saved", "persisted", "stored", "updated", "deleted", "removed",
        "loaded into sqlite", "loaded into the database", "updated your sqlite",
        "i applied", "confirmed — updated", "i have updated", "i've updated",
        "changes have been applied", "profile has been updated",
    ]
    if any(p in t for p in persistence_phrases) and not flags.did_persist_change:
        issues.append("Text implies data was persisted/modified, but did_persist_change=false.")

    # 2. Offers or promises to change data (always forbidden — agent is read-only)
    offer_phrases = [
        "i can update", "i'll update", "i will update",
        "i can save", "i'll save", "i will save",
        "i can apply", "i'll apply", "i will apply",
        "shall i apply", "shall i update", "shall i save",
        "want me to update", "want me to apply", "want me to save",
        "i can change", "i'll change", "i will change",
        "let me update", "let me apply", "let me save",
    ]
    if any(p in t for p in offer_phrases):
        issues.append("Text offers or promises to modify data. The assistant is strictly read-only.")

    # 3. Claims of web search without actually doing it
    web_phrases = ["i searched the web", "via tavily", "from tavily", "latest trend summary", "i used tavily"]
    if any(p in t for p in web_phrases) and not flags.did_use_tavily:
        issues.append("Text implies web/Tavily was used, but did_use_tavily=false.")

    # 4. Performance metrics without computation
    metric_phrases = ["start value", "end value", "max drawdown", "return:", "return %", "drawdown"]
    if any(p in t for p in metric_phrases) and not flags.did_compute_metrics:
        issues.append("Text includes performance metrics, but did_compute_metrics=false.")

    # 5. Portfolio facts without data
    if not flags.has_portfolio_data:
        factualish = any(p in t for p in ["your portfolio is", "you currently have", "as of", "latest snapshot", "allocation is"])
        if factualish:
            issues.append("Text states portfolio facts, but has_portfolio_data=false.")

    # 6. Missing data for requested metrics
    if flags.user_requested_metrics and flags.has_data_in_requested_range is False:
        if not any(p in t for p in ["don't have", "do not have", "no data", "upload", "add day", "date range"]):
            issues.append("User requested metrics but there is no data in the requested range; assistant should ask for data instead of implying results.")

    return issues


def enforce_policy(
    policy_llm: BaseChatModel,
    assistant_text: str,
    flags: ExecutionFlags,
) -> Tuple[str, Dict[str, Any]]:
    issues = _detect_issues(assistant_text, flags)
    if not issues:
        return assistant_text, {"ok": True, "issues": []}

    payload = {
        "assistant_text": assistant_text,
        "execution_flags": asdict(flags),
        "issues": issues,
    }

    # Try structured output first for reliable JSON
    try:
        structured_policy = policy_llm.with_structured_output(PolicyVerdict)
        result = structured_policy.invoke([
            SystemMessage(content=POLICY_PROMPT),
            HumanMessage(content="INPUT:\n" + json.dumps(payload, default=str)),
        ])
        safe_text = result.safe_text or assistant_text
        return safe_text, {"ok": result.ok, "issues": result.issues}
    except Exception:
        pass

    # Fallback: free-text call + manual JSON extraction
    msg = policy_llm.invoke([
        SystemMessage(content=POLICY_PROMPT),
        HumanMessage(content="INPUT:\n" + json.dumps(payload, default=str)),
    ])

    try:
        out = json.loads(msg.content)
        safe_text = out.get("safe_text") or assistant_text
        return safe_text, {"ok": bool(out.get("ok", False)), "issues": out.get("issues", issues)}
    except Exception:
        fallback = (
            "I want to be careful not to assume anything I haven't actually computed or stored. "
            "Use the sidebar to upload a CSV file with your portfolio data for the dates you need, or confirm Tavily for trends."
        )
        return fallback, {"ok": False, "issues": issues + ["Policy agent output was not valid JSON."]}
