from __future__ import annotations

import json
from typing import Dict, Any, List, Literal, Optional

from langchain_tavily import TavilySearch
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

AssetClass = Literal["Stock","Bond","Cash","ETF"]
RiskProfile = Literal["Conservative","Moderate","Aggressive"]

tavily = TavilySearch(max_results=6)

ASSET_CLASS_QUERIES: Dict[AssetClass, List[str]] = {
    "Stock": [
        "latest stock market trend outlook volatility risk sentiment",
        "latest equities market trend inflation earnings recession risk",
    ],
    "Bond": [
        "latest bond market trend interest rate outlook duration credit spreads",
        "latest treasury yield trend bond market volatility outlook",
    ],
    "Cash": [
        "latest cash yields trend high interest savings money market outlook",
        "latest interest rate trend central bank policy outlook cash returns",
    ],
    "ETF": [
        "latest ETF market trend passive investing flows liquidity",
        "latest exchange traded fund trend index ETF flows outlook",
    ],
}

def search_trends(asset_classes: List[AssetClass]) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for ac in asset_classes:
        merged = []
        for q in ASSET_CLASS_QUERIES.get(ac, []):
            merged.append({"query": q, "result": tavily.invoke({"query": q})})
        results[ac] = merged
    return results

def _trim_trends_for_prompt(trends_result: Dict[str, Any], max_per_result: int = 1200) -> Dict[str, Any]:
    """Trim raw Tavily results so the prompt to the LLM stays within token limits."""
    out: Dict[str, Any] = {}
    for ac, items in trends_result.items():
        if not isinstance(items, list):
            out[ac] = items
            continue
        trimmed = []
        for item in items[:2]:  # at most 2 queries per asset class
            if isinstance(item, dict):
                q = item.get("query", "")
                r = item.get("result", "")
                if isinstance(r, str) and len(r) > max_per_result:
                    r = r[:max_per_result] + "..."
                trimmed.append({"query": q, "result": r})
            else:
                trimmed.append(item)
        out[ac] = trimmed
    return out


def summarize_and_recommend(llm: BaseChatModel, trends_result: Dict[str, Any], current_profile: Optional[RiskProfile]) -> Dict[str, Any]:
    trends_trimmed = _trim_trends_for_prompt(trends_result)
    prompt = f"""
You are analyzing internet trend snippets for asset classes (stocks/bonds/cash/ETF) and giving risk-aware guidance.

Current risk profile: {current_profile}

Task:
1) Summarize trend per asset class in 2-4 bullet points each (risk/volatility/interest rates focus).
2) Provide a cautious recommendation: keep risk profile / consider decreasing risk / consider increasing risk.
3) Return ONLY JSON:
{{
  "summary": {{"Stock":[...],"Bond":[...],"Cash":[...],"ETF":[...]}},
  "risk_recommendation": {{
     "suggested_change":"none|decrease|increase",
     "suggested_profile":"Conservative|Moderate|Aggressive|null",
     "reason":"short"
  }}
}}

Trends result:
{trends_trimmed}
""".strip()

    msg = llm.invoke([HumanMessage(content=prompt)])
    try:
        return json.loads(msg.content)
    except (json.JSONDecodeError, TypeError):
        return {
            "summary": {"Stock": [], "Bond": [], "Cash": [], "ETF": []},
            "risk_recommendation": {
                "suggested_change": "none",
                "suggested_profile": None,
                "reason": "Could not parse trend summary.",
            },
        }
