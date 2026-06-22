"""Core portfolio maths: positions from the ledger, valuation, correlations.

Everything is reported in the base currency (CAD). USD holdings are converted at
the current USD/CAD rate. Note: cost basis is converted at *today's* FX rather than
the FX on each purchase date — a deliberate simplification for a personal tracker.
"""
import numpy as np
import pandas as pd

import db
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


# Currency exposure = the fund's trading currency, with overrides for cases where
# that misrepresents the real FX risk. XUS trades in CAD but holds the unhedged
# S&P 500, so its risk is USD. (Add more overrides here if needed.)
CURRENCY_OVERRIDE = {"XUS": "USD"}


def currency_exposure(pos: pd.DataFrame) -> pd.Series:
    """CAD vs USD exposure by market value (XUS treated as USD)."""
    buckets = {}
    for _, r in pos.iterrows():
        ccy = CURRENCY_OVERRIDE.get(r["Ticker"], r["Cur"])
        buckets[ccy] = buckets.get(ccy, 0.0) + r["Market Value"]
    s = pd.Series(buckets)
    order = ["CAD", "USD"]
    return s.reindex([c for c in order if c in s.index] +
                     [c for c in s.index if c not in order])


def benchmark_daily_pct() -> float:
    q = pricelib.get_current_and_prev([BENCHMARK_SYMBOL])
    if BENCHMARK_SYMBOL in q.index:
        r = q.loc[BENCHMARK_SYMBOL]
        return (r["price"] - r["prev_close"]) / r["prev_close"] if r["prev_close"] else 0.0
    return 0.0


# Benchmarks for the relative-performance chart. Edit symbols/labels as you like.
BENCHMARKS = {"Total Market": "VT", "S&P 500": "^GSPC"}


def normalized_performance(symbols, period="1y") -> pd.DataFrame:
    """Daily closes for each symbol, rebased to 100 at the start of the period."""
    symbols = [s for s in dict.fromkeys(symbols) if s]   # de-dupe, keep order
    hist = pricelib.get_history(symbols, period=period)
    if hist.empty:
        return hist
    return hist.apply(lambda c: c / c.dropna().iloc[0] * 100
                      if c.dropna().size else c)


def tfsa_cumulative_room(year: int) -> float:
    """Total TFSA room accrued from the year you turned 18 through `year`."""
    start = db.USER_BIRTH_YEAR + 18
    return float(sum(v for y, v in db.TFSA_ANNUAL_LIMITS.items()
                     if start <= y <= year))


def fhsa_status(contribs_by_year: dict, year: int) -> dict:
    """FHSA room this year (annual + carryforward) and lifetime remaining."""
    carry, used, avail = 0.0, 0.0, 0.0
    for y in range(db.FHSA_OPEN_YEAR, year + 1):
        room = db.FHSA_ANNUAL_LIMIT + carry
        c = float(contribs_by_year.get(y, 0))
        used += c
        if y < year:
            carry = min(db.FHSA_MAX_CARRYFORWARD, max(0.0, room - c))
        else:
            avail = max(0.0, room - c)
    lifetime_remaining = max(0.0, db.FHSA_LIFETIME_LIMIT - used)
    return {"available_this_year": min(avail, lifetime_remaining),
            "used_lifetime": used, "lifetime_remaining": lifetime_remaining}


def portfolio_performance(tx, instruments, period="1y") -> pd.Series:
    """Backtest current holdings over their price history → value rebased to 100.

    Uses today's share counts and current FX (consistent with the rest of the app).
    Private holdings (no market data, e.g. OPO) are excluded.
    """
    pos = compute_positions(tx)
    inst = instruments.set_index("ticker")
    usd_cad = pricelib.get_usd_cad()
    legs = []  # (yf_symbol, shares, fx)
    for _, p in pos.iterrows():
        t = p["ticker"]
        if t not in inst.index or p["shares"] == 0:
            continue
        meta = inst.loc[t]
        if meta["is_private"] or not meta["yf_symbol"]:
            continue
        legs.append((meta["yf_symbol"], p["shares"],
                     usd_cad if meta["currency"] == "USD" else 1.0))
    if not legs:
        return pd.Series(dtype=float)
    hist = pricelib.get_history([s for s, _, _ in legs], period=period)
    if hist.empty:
        return pd.Series(dtype=float)
    value = None
    for sym, shares, fx in legs:
        if sym in hist.columns:
            leg = hist[sym] * shares * fx
            value = leg if value is None else value.add(leg)
    value = value.dropna() if value is not None else pd.Series(dtype=float)
    if value.empty:
        return value
    return value / value.iloc[0] * 100


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
