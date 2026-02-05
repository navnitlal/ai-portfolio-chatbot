from __future__ import annotations
from typing import Dict, Literal, Tuple

RiskProfile = Literal["Conservative", "Moderate", "Aggressive"]

# Minimum cash target: rebalance suggestions keep cash close to this (rest in Stock/Bond/ETF).
MIN_CASH_TARGET = 0.05  # 5% buffer; Stock/Bond/ETF fill the remainder in risk-appropriate ratios.

# Nominal Stock/Bond/ETF ratios (excluding cash) per risk category; cash is minimized in targets.
_NOMINAL_RATIOS: Dict[RiskProfile, Tuple[float, float, float]] = {
    "Conservative": (0.20, 0.45, 0.10),   # Stock, Bond, ETF (relative weights)
    "Moderate": (0.45, 0.25, 0.10),
    "Aggressive": (0.65, 0.15, 0.10),
}


def _allocation_with_min_cash(risk: RiskProfile) -> Tuple[float, float, float, float]:
    """Target (Stock, Bond, Cash, ETF) with cash at MIN_CASH_TARGET and S/B/E in required risk range."""
    s, b, e = _NOMINAL_RATIOS[risk]
    rest = 1.0 - MIN_CASH_TARGET
    total = s + b + e
    return (rest * s / total, rest * b / total, MIN_CASH_TARGET, rest * e / total)


# Target Stock/Bond/Cash/ETF allocation for rebalance suggestions (cash kept close to zero).
TARGET_ALLOCATION_BY_RISK: Dict[RiskProfile, Tuple[float, float, float, float]] = {
    profile: _allocation_with_min_cash(profile) for profile in ("Conservative", "Moderate", "Aggressive")
}


def infer_risk_from_allocation(
    stock_pct: float, bond_pct: float, cash_pct: float, etf_pct: float = 0.0
) -> RiskProfile:
    """Infer risk category from portfolio allocation (Stock/Bond/Cash/ETF ratios)."""
    # Equity-heavy (Stock + ETF) → more aggressive; bond/cash-heavy → more conservative
    equity = stock_pct + etf_pct
    if equity >= 0.65:
        return "Aggressive"
    if equity >= 0.35:
        return "Moderate"
    return "Conservative"


def score_risk_answers(answers: Dict[str, str]) -> RiskProfile:
    # Q1=horizon, Q2=stability, Q3=goal, Q4=drop reaction (tolerance question removed)
    a1 = (answers.get("1") or "").lower()
    a2 = (answers.get("2") or "").lower()
    a3 = (answers.get("3") or "").lower()
    a4 = (answers.get("4") or "").lower()

    score = 0
    score += 2 if "8" in a1 or "+" in a1 else (1 if "3" in a1 or "7" in a1 else 0)
    score += 2 if "high" in a2 else (1 if "medium" in a2 else 0)
    score += 2 if "max" in a3 or "growth" in a3 else (1 if "balanced" in a3 else 0)
    score += 2 if "buy" in a4 else (1 if "hold" in a4 else 0)

    if score >= 6:
        return "Aggressive"
    if score >= 3:
        return "Moderate"
    return "Conservative"
