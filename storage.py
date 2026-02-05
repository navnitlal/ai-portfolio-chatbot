from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date
from typing import Optional, Dict, Any

import pandas as pd

DB_SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS portfolio_daily (
  date TEXT PRIMARY KEY,
  stock_value REAL NOT NULL,
  bond_value REAL NOT NULL,
  cash_value REAL NOT NULL,
  etf_value REAL NOT NULL DEFAULT 0,
  cash_addition REAL NOT NULL DEFAULT 0,
  cash_withdrawal REAL NOT NULL DEFAULT 0,
  total_value REAL,
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS settings (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL,
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def _migrate_portfolio_columns(con: sqlite3.Connection) -> None:
    """Add cash_addition, cash_withdrawal, total_value, etf_value if missing (existing DBs)."""
    cur = con.execute("PRAGMA table_info(portfolio_daily)")
    cols = {row[1] for row in cur.fetchall()}
    if "cash_addition" not in cols:
        con.execute("ALTER TABLE portfolio_daily ADD COLUMN cash_addition REAL NOT NULL DEFAULT 0")
    if "cash_withdrawal" not in cols:
        con.execute("ALTER TABLE portfolio_daily ADD COLUMN cash_withdrawal REAL NOT NULL DEFAULT 0")
    if "total_value" not in cols:
        con.execute("ALTER TABLE portfolio_daily ADD COLUMN total_value REAL")
        con.execute(
            "UPDATE portfolio_daily SET total_value = stock_value + bond_value + cash_value WHERE total_value IS NULL"
        )
    if "etf_value" not in cols:
        con.execute("ALTER TABLE portfolio_daily ADD COLUMN etf_value REAL NOT NULL DEFAULT 0")
        con.execute(
            "UPDATE portfolio_daily SET total_value = stock_value + bond_value + cash_value + etf_value WHERE total_value IS NOT NULL"
        )
    con.execute(
        "UPDATE portfolio_daily SET total_value = stock_value + bond_value + cash_value + etf_value WHERE total_value IS NULL OR total_value = 0"
    )
    con.commit()

@dataclass
class SQLiteStore:
    path: str

    def connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, check_same_thread=False)
        con.row_factory = sqlite3.Row
        return con

    def init(self) -> None:
        with self.connect() as con:
            con.executescript(DB_SCHEMA)
            con.commit()
            _migrate_portfolio_columns(con)

    def upsert_day(
        self,
        d: date,
        stock: float,
        bond: float,
        cash: float,
        etf: float = 0.0,
        cash_addition: float = 0.0,
        cash_withdrawal: float = 0.0,
    ) -> None:
        total = float(stock) + float(bond) + float(cash) + float(etf)
        with self.connect() as con:
            con.execute(
                """INSERT INTO portfolio_daily(
                     date, stock_value, bond_value, cash_value, etf_value,
                     cash_addition, cash_withdrawal, total_value)
                   VALUES(?,?,?,?,?,?,?,?)
                   ON CONFLICT(date) DO UPDATE SET
                      stock_value=excluded.stock_value,
                      bond_value=excluded.bond_value,
                      cash_value=excluded.cash_value,
                      etf_value=excluded.etf_value,
                      cash_addition=excluded.cash_addition,
                      cash_withdrawal=excluded.cash_withdrawal,
                      total_value=excluded.total_value,
                      updated_at=datetime('now')
                """,
                (
                    d.isoformat(),
                    float(stock),
                    float(bond),
                    float(cash),
                    float(etf),
                    float(cash_addition),
                    float(cash_withdrawal),
                    total,
                ),
            )
            con.commit()

    def delete_day(self, d: date) -> None:
        with self.connect() as con:
            con.execute("DELETE FROM portfolio_daily WHERE date=?", (d.isoformat(),))
            con.commit()

    def clear_portfolio(self) -> None:
        """Delete all rows from portfolio_daily. Settings (e.g. risk_profile) are kept."""
        with self.connect() as con:
            con.execute("DELETE FROM portfolio_daily")
            con.commit()

    def get_day(self, d: date) -> Optional[Dict[str, Any]]:
        with self.connect() as con:
            row = con.execute("SELECT * FROM portfolio_daily WHERE date=?", (d.isoformat(),)).fetchone()
            return dict(row) if row else None

    def load_all(self) -> pd.DataFrame:
        with self.connect() as con:
            rows = con.execute(
                """SELECT date, stock_value, bond_value, etf_value, cash_value,
                          cash_addition, cash_withdrawal, total_value
                   FROM portfolio_daily ORDER BY date"""
            ).fetchall()
        df = pd.DataFrame(
            rows,
            columns=[
                "Date",
                "StockValue",
                "BondValue",
                "ETFValue",
                "CashValue",
                "CashAddition",
                "CashWithdrawal",
                "TotalValue",
            ],
        )
        if df.empty:
            return df
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        for c in ["StockValue", "BondValue", "ETFValue", "CashValue", "CashAddition", "CashWithdrawal", "TotalValue"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(float)
        # Recompute total if missing (e.g. legacy rows)
        mask = (df["TotalValue"] == 0) & (df["StockValue"] + df["BondValue"] + df["CashValue"] + df["ETFValue"] > 0)
        if mask.any():
            df.loc[mask, "TotalValue"] = (
                df.loc[mask, "StockValue"] + df.loc[mask, "BondValue"] + df.loc[mask, "CashValue"] + df.loc[mask, "ETFValue"]
            )
        return df

    def load_range(self, start: date, end: date) -> pd.DataFrame:
        with self.connect() as con:
            rows = con.execute(
                """SELECT date, stock_value, bond_value, etf_value, cash_value,
                          cash_addition, cash_withdrawal, total_value
                   FROM portfolio_daily
                   WHERE date>=? AND date<=?
                   ORDER BY date""",
                (start.isoformat(), end.isoformat()),
            ).fetchall()
        df = pd.DataFrame(
            rows,
            columns=[
                "Date",
                "StockValue",
                "BondValue",
                "ETFValue",
                "CashValue",
                "CashAddition",
                "CashWithdrawal",
                "TotalValue",
            ],
        )
        if df.empty:
            return df
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        for c in ["StockValue", "BondValue", "ETFValue", "CashValue", "CashAddition", "CashWithdrawal", "TotalValue"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(float)
        mask = (df["TotalValue"] == 0) & (df["StockValue"] + df["BondValue"] + df["CashValue"] + df["ETFValue"] > 0)
        if mask.any():
            df.loc[mask, "TotalValue"] = (
                df.loc[mask, "StockValue"] + df.loc[mask, "BondValue"] + df.loc[mask, "CashValue"] + df.loc[mask, "ETFValue"]
            )
        return df

    def set_setting(self, key: str, value: str) -> None:
        with self.connect() as con:
            con.execute(
                """INSERT INTO settings(key, value) VALUES(?,?)
                   ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=datetime('now')""",
                (key, value),
            )
            con.commit()

    def get_setting(self, key: str) -> Optional[str]:
        with self.connect() as con:
            row = con.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
            return row["value"] if row else None

    def set_risk_profile(self, profile: str) -> None:
        self.set_setting("risk_profile", profile)

    def get_risk_profile(self) -> Optional[str]:
        return self.get_setting("risk_profile")
