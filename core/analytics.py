from __future__ import annotations

from datetime import date
from typing import Optional, Literal, Dict

import numpy as np
import pandas as pd

AssetClass = Literal["Stock", "Bond", "Cash", "ETF"]

def normalize_wide_csv(df: pd.DataFrame) -> pd.DataFrame:
    required = ["Date", "StockValue", "BondValue", "ETFValue", "CashValue"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"]).dt.date
    for c in ["StockValue", "BondValue", "ETFValue", "CashValue"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        if out[c].isna().any():
            raise ValueError(f"Some {c} values are not numeric.")
        out[c] = out[c].astype(float)
        if (out[c] < 0).any():
            raise ValueError(f"{c} must be >= 0")

    # Optional: cash flow fields for accurate performance (TWR/MWR)
    for c in ["CashAddition", "CashWithdrawal"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(float)
            if (out[c] < 0).any():
                raise ValueError(f"{c} must be >= 0")
        else:
            out[c] = 0.0

    out = out.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
    # Canonical column order: ETFValue before CashValue
    value_cols = [c for c in ["Date", "StockValue", "BondValue", "ETFValue", "CashValue", "CashAddition", "CashWithdrawal"] if c in out.columns]
    out = out[value_cols]
    out = add_total(out)
    return out

def add_total(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalValue"] = df["StockValue"] + df["BondValue"] + df["CashValue"] + df["ETFValue"]
    return df

def time_weighted_return(df: pd.DataFrame, start: date, end: date, asset_class: Optional[AssetClass] = None) -> Dict[str, float]:
    """Time-Weighted Return (TWR). With no cash flows this is one sub-period = (End-Start)/Start; with cash flows compounds sub-period returns."""
    if df.empty:
        return {"start_value": 0.0, "end_value": 0.0, "return_pct": 0.0, "sub_periods": 0}

    dff = df[(df["Date"] >= start) & (df["Date"] <= end)].copy()
    if dff.empty:
        return {"start_value": 0.0, "end_value": 0.0, "return_pct": 0.0, "sub_periods": 0}

    dff = dff.sort_values("Date")

    # Cash flow dates (after start); if none, we have one sub-period [start, end] so TWR = simple return
    cash_flow_dates = []
    if "CashAddition" in dff.columns and "CashWithdrawal" in dff.columns:
        cash_flow_dates = dff[(dff["Date"] > start) & (
            (dff["CashAddition"] > 0) | (dff["CashWithdrawal"] > 0)
        )]["Date"].tolist()

    if asset_class is None:
        dff = add_total(dff)
        value_col = "TotalValue"
    else:
        value_col = {"Stock": "StockValue", "Bond": "BondValue", "Cash": "CashValue", "ETF": "ETFValue"}[asset_class]

    split_dates = sorted([start] + cash_flow_dates + [end])
    sub_returns = []

    for i in range(len(split_dates) - 1):
        sub_start = split_dates[i]
        sub_end = split_dates[i + 1]
        sub_df = dff[(dff["Date"] >= sub_start) & (dff["Date"] <= sub_end)].copy()
        if sub_df.empty:
            continue
        sub_df = sub_df.sort_values("Date")
        start_val = float(sub_df.iloc[0][value_col])
        end_val = float(sub_df.iloc[-1][value_col])
        if start_val > 0:
            sub_returns.append((end_val - start_val) / start_val)

    if not sub_returns:
        start_val = float(dff.iloc[0][value_col])
        end_val = float(dff.iloc[-1][value_col])
        ret_pct = 0.0 if start_val == 0 else (end_val - start_val) / start_val * 100.0
        return {"start_value": start_val, "end_value": end_val, "return_pct": ret_pct, "sub_periods": 0}

    twr = 1.0
    for r in sub_returns:
        twr *= (1.0 + r)
    twr_pct = (twr - 1.0) * 100.0
    start_val = float(dff.iloc[0][value_col])
    end_val = float(dff.iloc[-1][value_col])
    return {
        "start_value": start_val,
        "end_value": end_val,
        "return_pct": twr_pct,
        "sub_periods": len(sub_returns),
    }

def max_drawdown(df: pd.DataFrame, start: date, end: date) -> float:
    if df.empty:
        return 0.0
    dff = df[(df["Date"] >= start) & (df["Date"] <= end)].copy()
    if dff.empty:
        return 0.0
    dff = add_total(dff).sort_values("Date")
    vals = dff["TotalValue"].to_numpy(dtype=float)
    if len(vals) < 2:
        return 0.0
    peaks = np.maximum.accumulate(vals)
    dd = (vals - peaks) / peaks
    return float(dd.min() * 100.0)

def allocation_on_date(df: pd.DataFrame, d: date) -> Dict[str, float]:
    if df.empty:
        return {"Stock":0.0,"Bond":0.0,"Cash":0.0,"ETF":0.0}
    row = df[df["Date"] == d]
    if row.empty:
        prev = df[df["Date"] <= d].sort_values("Date")
        if prev.empty:
            return {"Stock":0.0,"Bond":0.0,"Cash":0.0,"ETF":0.0}
        row = prev.tail(1)
    r = row.iloc[0]
    etf_val = float(r["ETFValue"])
    total = float(r["StockValue"] + r["BondValue"] + r["CashValue"] + etf_val)
    if total <= 0:
        return {"Stock":0.0,"Bond":0.0,"Cash":0.0,"ETF":0.0}
    return {
        "Stock": float(r["StockValue"]/total),
        "Bond": float(r["BondValue"]/total),
        "Cash": float(r["CashValue"]/total),
        "ETF": float(etf_val/total),
    }

def held_asset_classes_recent(df: pd.DataFrame, lookback_days: int = 14) -> list[str]:
    if df.empty:
        return []
    dff = df.sort_values("Date").tail(lookback_days)
    avgs = {
        "Stock": float(dff["StockValue"].mean()),
        "Bond": float(dff["BondValue"].mean()),
        "Cash": float(dff["CashValue"].mean()),
        "ETF": float(dff["ETFValue"].mean()),
    }
    return [k for k,v in avgs.items() if v > 0.0]
