from __future__ import annotations

import json
from typing import Dict, Any, List, Literal, Optional

from langchain_tavily import TavilySearch
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

AssetClass = Literal["Stock","Bond","Cash","ETF"]
RiskProfile = Literal["Conservative","Moderate","Aggressive"]

# Lazy initialization: created on first use so missing TAVILY_API_KEY fails gracefully.
_tavily: Optional[TavilySearch] = None


def _get_tavily() -> TavilySearch:
    global _tavily
    if _tavily is None:
        _tavily = TavilySearch(max_results=6)
    return _tavily

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
    """Search Tavily for each asset class's queries. Errors are caught per-query so one failure doesn't break the whole batch."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    tavily = _get_tavily()
    results: Dict[str, Any] = {}

    def _run_query(ac: str, q: str) -> tuple[str, Dict[str, Any]]:
        try:
            return ac, {"query": q, "result": tavily.invoke({"query": q})}
        except Exception as e:
            return ac, {"query": q, "result": f"[Tavily error: {e}]"}

    # Collect all (asset_class, query) pairs and run them in parallel
    tasks = []
    for ac in asset_classes:
        for q in ASSET_CLASS_QUERIES.get(ac, []):
            tasks.append((ac, q))

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_run_query, ac, q): (ac, q) for ac, q in tasks}
        for future in as_completed(futures):
            ac, item = future.result()
            results.setdefault(ac, []).append(item)

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
You are analyzing internet trend snippets for asset classes (stocks/bonds/cash/ETF) and providing educational, risk-aware observations.

Current risk profile: {current_profile}

Important: You are strictly informational. Do NOT recommend the user change their portfolio or risk profile. Do NOT say "you should" or "I recommend". Instead, present neutral observations about how current trends relate to different risk postures (conservative, moderate, aggressive). The user and their financial advisor will decide what to do.

Task:
1) Summarize trend per asset class in 2-4 bullet points each (risk/volatility/interest rates focus).
2) Provide a neutral observation about whether current market conditions generally favor conservative, moderate, or aggressive positioning — framed as educational context, not a personal recommendation.
3) Return ONLY JSON:
{{
  "summary": {{"Stock":[...],"Bond":[...],"Cash":[...],"ETF":[...]}},
  "risk_recommendation": {{
     "suggested_change":"none|decrease|increase",
     "suggested_profile":"Conservative|Moderate|Aggressive|null",
     "reason":"short educational observation, not a personal recommendation"
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
