"""Database layer. Uses Postgres when DATABASE_URL is set, else a local SQLite file.

Both the Streamlit app and the daily GitHub Action import from here, so it is the
single source of truth for the schema and seed data.
"""
import os
import datetime as dt

import pandas as pd
from sqlalchemy import (
    create_engine, MetaData, Table, Column, Integer, String, Float, Date,
    Boolean, select, func,
)

# Seed: your current holdings, lifted from the original spreadsheet.
SEED_INSTRUMENTS = [
    # ticker, yfinance symbol, currency, is_private, manual_price
    ("ZMMK", "ZMMK.TO", "CAD", False, None),
    ("XEQT", "XEQT.TO", "CAD", False, None),
    ("XUS",  "XUS.TO",  "CAD", False, None),
    ("AVGE", "AVGE",    "USD", False, None),   # US-listed, priced in USD
    ("OPO",  None,      "CAD", True,  17.184),  # private holding, price entered manually
]

SEED_TRANSACTIONS = [
    # date, ticker, account, action, shares, price, fees
    ("ZMMK", "FHSA", "Buy", 82,                49.84),
    ("XEQT", "FHSA", "Buy", 154,               45.1243),
    ("XEQT", "TFSA", "Buy", 88,                44.4813),
    ("XUS",  "FHSA", "Buy", 75,                65.134),
    ("XUS",  "TFSA", "Buy", 60,                64.7122),
    ("AVGE", "TFSA", "Buy", 16,                98.11),
    ("OPO",  "TFSA", "Buy", 633.634029280884,  14.8766),
]

BENCHMARK_SYMBOL = "VFV.TO"   # all-equity ETF used as the portfolio benchmark
BASE_CURRENCY = "CAD"

# --- contribution-room config (edit to match your situation) ---------------
USER_BIRTH_YEAR = 2002        # TFSA room accrues from the year you turn 18
FHSA_OPEN_YEAR = 2025         # first year your FHSA existed (room starts here)
FHSA_ANNUAL_LIMIT = 8000
FHSA_LIFETIME_LIMIT = 40000
FHSA_MAX_CARRYFORWARD = 8000  # most FHSA room you can carry into a single year
TFSA_ANNUAL_LIMITS = {
    2009: 5000, 2010: 5000, 2011: 5000, 2012: 5000, 2013: 5500, 2014: 5500,
    2015: 10000, 2016: 5500, 2017: 5500, 2018: 5500, 2019: 6000, 2020: 6000,
    2021: 6000, 2022: 6000, 2023: 6500, 2024: 7000, 2025: 7000, 2026: 7000,
}

# Seed contributions (loaded once; edit in-app afterward). Confirmed with you:
#   RBC TFSA: $6,500 (2025) + $3,000 (2026) · FHSA: $16,000 (2026)
#   OPO (Optimize TFSA): $500/mo, assumed from Jan 2025 — adjust if different.
SEED_CONTRIBUTIONS = [
    (dt.date(2026, 1, 15), "TFSA", 6500.0, "RBC TFSA"),
    (dt.date(2026, 3, 1), "TFSA", 3000.0, "RBC TFSA"),
    (dt.date(2026, 2, 1), "FHSA", 16000.0, "FHSA lump"),
]


def _database_url():
    url = os.environ.get("DATABASE_URL")
    if not url:
        try:  # running inside Streamlit with a secret configured
            import streamlit as st
            url = st.secrets.get("DATABASE_URL")
        except Exception:
            url = None
    if not url:
        here = os.path.dirname(os.path.abspath(__file__))
        return f"sqlite:///{os.path.join(here, 'stonks.db')}"
    # SQLAlchemy wants postgresql+psycopg2:// ; accept the plain postgres:// form too
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg2://", 1)
    return url


engine = create_engine(_database_url(), future=True)
metadata = MetaData()

instruments = Table(
    "instruments", metadata,
    Column("ticker", String, primary_key=True),
    Column("yf_symbol", String),
    Column("currency", String, nullable=False, default="CAD"),
    Column("is_private", Boolean, nullable=False, default=False),
    Column("manual_price", Float),
)

transactions = Table(
    "transactions", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("date", Date, nullable=False),
    Column("ticker", String, nullable=False),
    Column("account", String, nullable=False),   # FHSA / TFSA / ...
    Column("action", String, nullable=False),     # Buy / Sell / Dividend
    Column("shares", Float, nullable=False, default=0.0),
    Column("price", Float, nullable=False, default=0.0),
    Column("fees", Float, nullable=False, default=0.0),
)

snapshots = Table(
    "snapshots", metadata,
    Column("date", Date, primary_key=True),
    Column("market_value", Float),
    Column("book_value", Float),
    Column("gain_loss", Float),
    Column("gain_loss_pct", Float),
    Column("daily_pnl", Float),
    Column("daily_pnl_pct", Float),
    Column("benchmark_pct", Float),
)

contributions = Table(
    "contributions", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("date", Date, nullable=False),
    Column("account", String, nullable=False),   # TFSA / FHSA / ...
    Column("amount", Float, nullable=False),
    Column("note", String),
)


def _seed_contribution_rows():
    """One-time contributions seed, incl. the OPO $500/mo series to this month."""
    rows = [dict(date=d, account=a, amount=amt, note=n)
            for (d, a, amt, n) in SEED_CONTRIBUTIONS]
    y, m = 2025, 1
    today = dt.date.today()
    while (y, m) <= (today.year, today.month):
        rows.append(dict(date=dt.date(y, m, 1), account="TFSA", amount=500.0,
                         note="OPO (Optimize TFSA) $500/mo"))
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return rows


def init_db(seed: bool = True):
    """Create tables if missing and load seed data on first run."""
    metadata.create_all(engine)
    if not seed:
        return
    with engine.begin() as conn:
        if conn.execute(select(func.count()).select_from(instruments)).scalar() == 0:
            conn.execute(instruments.insert(), [
                dict(ticker=t, yf_symbol=y, currency=c, is_private=p, manual_price=m)
                for (t, y, c, p, m) in SEED_INSTRUMENTS
            ])
        if conn.execute(select(func.count()).select_from(transactions)).scalar() == 0:
            today = dt.date.today()
            conn.execute(transactions.insert(), [
                dict(date=today, ticker=t, account=a, action=act,
                     shares=s, price=pr, fees=0.0)
                for (t, a, act, s, pr) in SEED_TRANSACTIONS
            ])
        if conn.execute(select(func.count()).select_from(contributions)).scalar() == 0:
            conn.execute(contributions.insert(), _seed_contribution_rows())


# ---- read helpers -------------------------------------------------
def get_transactions_df() -> pd.DataFrame:
    return pd.read_sql(select(transactions).order_by(transactions.c.date), engine)


def get_instruments_df() -> pd.DataFrame:
    return pd.read_sql(select(instruments), engine)


def get_snapshots_df() -> pd.DataFrame:
    return pd.read_sql(select(snapshots).order_by(snapshots.c.date), engine)


# ---- write helpers ------------------------------------------------
def add_transaction(date, ticker, account, action, shares, price, fees=0.0):
    with engine.begin() as conn:
        conn.execute(transactions.insert().values(
            date=date, ticker=ticker, account=account, action=action,
            shares=float(shares), price=float(price), fees=float(fees),
        ))


def delete_transaction(tx_id: int):
    with engine.begin() as conn:
        conn.execute(transactions.delete().where(transactions.c.id == int(tx_id)))


def get_contributions_df() -> pd.DataFrame:
    return pd.read_sql(select(contributions).order_by(contributions.c.date), engine)


def add_contribution(date, account, amount, note=""):
    with engine.begin() as conn:
        conn.execute(contributions.insert().values(
            date=date, account=account, amount=float(amount), note=note))


def delete_contribution(cid: int):
    with engine.begin() as conn:
        conn.execute(contributions.delete().where(contributions.c.id == int(cid)))


def set_manual_price(ticker, price):
    with engine.begin() as conn:
        conn.execute(instruments.update()
                     .where(instruments.c.ticker == ticker)
                     .values(manual_price=float(price)))


def upsert_snapshot(row: dict):
    """Insert or replace the snapshot for row['date']."""
    # Coerce numpy scalars (np.float64 etc.) to native Python types — psycopg2
    # cannot adapt numpy values, unlike SQLite which silently tolerates them.
    row = {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}
    d = row["date"]
    with engine.begin() as conn:
        conn.execute(snapshots.delete().where(snapshots.c.date == d))
        conn.execute(snapshots.insert().values(**row))
