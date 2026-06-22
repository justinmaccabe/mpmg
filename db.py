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
    ("ZMMK", "FHSA", "Buy", 122,               49.84),
    ("XEQT", "FHSA", "Buy", 132,               45.0617),
    ("XEQT", "TFSA", "Buy", 88,                44.4813),
    ("XUS",  "FHSA", "Buy", 60,                64.9825),
    ("XUS",  "TFSA", "Buy", 60,                64.7122),
    ("AVGE", "TFSA", "Buy", 16,                98.11),
    ("OPO",  "TFSA", "Buy", 633.634029280884,  14.8766),
]

BENCHMARK_SYMBOL = "VFV.TO"   # all-equity ETF used as the portfolio benchmark
BASE_CURRENCY = "CAD"


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
