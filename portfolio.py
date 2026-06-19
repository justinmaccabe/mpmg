"""Core portfolio maths: positions from the ledger, valuation, correlations.

Everything is reported in the base currency (CAD). USD holdings are converted at
the current USD/CAD rate. Note: cost basis is converted at *today's* FX rather than
the FX on each purchase date — a deliberate simplification for a personal tracker.
"""
import numpy as np
import pandas as pd

import prices as pricelib
from db import BASE_CURRENCY, BENCHMARK_SYMBOL


def compute_positions(tx: pd.DataFrame) -> pd.DataFrame:
    """Net shares (total + per account) and average cost (ACB) per ticker."""
    if tx.empty:
        return pd.DataFrame(columns=["ticker", "shares", "acb", "FHSA", "TFSA"])

    signed = tx["shares"] * tx["action"].map({"Buy": 1, "Sell": -1}).fillna(0)
    tx = tx.assign(net=signed)

    out = []
    for ticker, g in tx.groupby("ticker"):
        buys = g[g["action"] == "Buy"]
        buy_shares = buys["shares"].sum()
        buy_cost = (buys["shares"] * buys["price"] + buys["fees"]).sum()
        acb = buy_cost / buy_shares if buy_shares else 0.0
        rec = {"ticker": ticker, "shares": g["net"].sum(), "acb": acb}
        for acct in ("FHSA", "TFSA"):
            rec[acct] = g[g["account"] == acct]["net"].sum()
        out.append(rec)
    return pd.DataFrame(out)


def _fx(currency, usd_cad):
    return usd_cad if currency == "USD" else 1.0


def build_portfolio(tx: pd.DataFrame, instruments: pd.DataFrame):
    """Return (positions_df, totals_dict) valued in the base currency."""
    pos = compute_positions(tx)
    inst = instruments.set_index("ticker")

    symbols = [s for s in inst["yf_symbol"].dropna().tolist()]
    quotes = pricelib.get_current_and_prev(symbols)
    usd_cad = pricelib.get_usd_cad()

    rows = []
    for _, p in pos.iterrows():
        t = p["ticker"]
        if t not in inst.index:
            continue
        meta = inst.loc[t]
        cur = meta["currency"]
        fx = _fx(cur, usd_cad)

        if meta["is_private"] or not meta["yf_symbol"] or meta["yf_symbol"] not in quotes.index:
            price = meta["manual_price"] or 0.0
            prev = price
        else:
            price = quotes.loc[meta["yf_symbol"], "price"]
            prev = quotes.loc[meta["yf_symbol"], "prev_close"]

        shares = p["shares"]
        mv = shares * price * fx
        book = shares * p["acb"] * fx
        daily = shares * (price - prev) * fx
        rows.append({
            "Ticker": t, "Account": _accounts(p),
            "Shares": shares, "ACB": p["acb"], "Price": price, "Cur": cur,
            "Market Value": mv, "Book Value": book,
            "Gain/Loss $": mv - book,
            "Gain/Loss %": (mv - book) / book if book else 0.0,
            "Daily P&L $": daily,
            "MV FHSA": p.get("FHSA", 0) * price * fx,
            "MV TFSA": p.get("TFSA", 0) * price * fx,
        })
    df = pd.DataFrame(rows)

    totals = {}
    if not df.empty:
        mv = df["Market Value"].sum()
        bv = df["Book Value"].sum()
        dp = df["Daily P&L $"].sum()
        totals = {
            "market_value": mv, "book_value": bv,
            "gain_loss": mv - bv,
            "gain_loss_pct": (mv - bv) / bv if bv else 0.0,
            "daily_pnl": dp,
            "daily_pnl_pct": dp / (mv - dp) if (mv - dp) else 0.0,
            "account_mv": {"FHSA": df["MV FHSA"].sum(), "TFSA": df["MV TFSA"].sum()},
        }
    return df, totals


def _accounts(p):
    parts = [a for a in ("FHSA", "TFSA") if p.get(a, 0)]
    return "+".join(parts) if parts else "-"


def benchmark_daily_pct() -> float:
    q = pricelib.get_current_and_prev([BENCHMARK_SYMBOL])
    if BENCHMARK_SYMBOL in q.index:
        r = q.loc[BENCHMARK_SYMBOL]
        return (r["price"] - r["prev_close"]) / r["prev_close"] if r["prev_close"] else 0.0
    return 0.0


def correlation_matrix(instruments: pd.DataFrame, period="1y") -> pd.DataFrame:
    """Correlation of daily returns across non-private holdings."""
    inst = instruments[(~instruments["is_private"]) & instruments["yf_symbol"].notna()]
    sym_to_ticker = dict(zip(inst["yf_symbol"], inst["ticker"]))
    hist = pricelib.get_history(list(sym_to_ticker), period=period)
    if hist.empty:
        return pd.DataFrame()
    returns = hist.pct_change(fill_method=None).dropna(how="all")
    corr = returns.corr()
    corr = corr.rename(index=sym_to_ticker, columns=sym_to_ticker)
    return corr
