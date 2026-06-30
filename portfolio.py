"""Core portfolio maths: positions from the ledger, valuation, correlations.

Everything is reported in the base currency (CAD). USD holdings are converted at
the current USD/CAD rate. Note: cost basis is converted at *today's* FX rather than
the FX on each purchase date — a deliberate simplification for a personal tracker.
"""
import datetime as dt

import numpy as np
import pandas as pd

import db
import prices as pricelib
from db import BASE_CURRENCY, BENCHMARK_SYMBOL


def max_drawdown(series) -> float:
    """Worst peak-to-trough decline of a value series (negative number)."""
    if series is None or len(series) < 2:
        return None
    rm = series.cummax()
    return float((series / rm - 1).min())


def _ret_to_date(series, since) -> float:
    """Return from the last close strictly before `since` to the latest value."""
    if series is None or len(series) == 0:
        return None
    base = series[series.index < pd.Timestamp(since)]
    if base.empty:
        return None
    return float(series.iloc[-1] / base.iloc[-1] - 1)


def period_returns(tx, instruments) -> dict:
    """Week-to-date and month-to-date returns: portfolio (current holdings
    backtested, OPO excluded) vs the benchmark."""
    pv = portfolio_performance(tx, instruments, "6mo")
    bench = pricelib.get_history([BENCHMARK_SYMBOL], period="6mo")
    bench = bench[BENCHMARK_SYMBOL] if BENCHMARK_SYMBOL in bench.columns else pd.Series(dtype=float)
    today = dt.date.today()
    wk = today - dt.timedelta(days=today.weekday())     # this week's Monday
    mo = today.replace(day=1)                           # first of this month
    return {
        "wtd_port": _ret_to_date(pv, wk), "wtd_bench": _ret_to_date(bench, wk),
        "mtd_port": _ret_to_date(pv, mo), "mtd_bench": _ret_to_date(bench, mo),
        "benchmark_symbol": BENCHMARK_SYMBOL,
    }


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

        sym = meta["yf_symbol"]
        live = None
        if sym and sym in quotes.index and pd.notna(quotes.loc[sym, "price"]):
            live = float(quotes.loc[sym, "price"])
            prev = float(quotes.loc[sym, "prev_close"])
        if meta["is_private"]:
            price = float(meta["manual_price"] or 0.0)
            prev = price
        elif live is not None:
            price = live
        else:
            # live fetch failed → fall back to last known price (yesterday),
            # then manual, then cost basis. Never show $0 / None.
            lp = meta["last_price"] if "last_price" in inst.columns else None
            if lp is not None and pd.notna(lp) and lp:
                price = float(lp)
            elif meta["manual_price"] and pd.notna(meta["manual_price"]):
                price = float(meta["manual_price"])
            else:
                price = float(p["acb"] or 0.0)
            prev = price

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


FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]


def leverage_metrics(market_value, loc_balance, prime, spread, yld, peak):
    """Leverage ratio, financing cost, and IPS flags from a single LOC balance."""
    equity = market_value - loc_balance
    rate = (prime + spread) / 100.0          # decimal
    annual_interest = loc_balance * rate
    drawdown = max(0.0, (peak - market_value) / peak) if peak else 0.0
    return {
        "gross_exposure": market_value,
        "equity": equity,
        "leverage": market_value / equity if equity else float("nan"),
        "loc_rate": rate,
        "annual_interest": annual_interest,
        "monthly_interest": annual_interest / 12.0,
        "drawdown": drawdown,
        "yield_le_cost": (yld / 100.0) <= rate,   # IPS §7 yield-vs-cost flag
        "drawdown_trigger": drawdown >= 0.50,     # IPS §8 review trigger
    }


_FF_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"


def _ff_csv(name, cols):
    """Download a Ken French factor zip and parse its monthly block (decimal)."""
    import io
    import re
    import urllib.request
    import zipfile
    data = urllib.request.urlopen(_FF_BASE + name + "_CSV.zip", timeout=30).read()
    z = zipfile.ZipFile(io.BytesIO(data))
    raw = z.read(z.namelist()[0]).decode("latin-1")
    rows = []
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= len(cols) + 1 and re.fullmatch(r"\d{6}", parts[0]):
            try:
                rows.append([parts[0]] + [float(parts[i + 1]) for i in range(len(cols))])
            except ValueError:
                continue
    df = pd.DataFrame(rows, columns=["date"] + cols)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m").dt.to_period("M").dt.to_timestamp("M")
    return df.set_index("date") / 100.0


def _trailing_yield(symbol):
    """Trailing-12-month distribution yield = TTM distributions / last price."""
    import yfinance as yf
    try:
        tk = yf.Ticker(symbol)
        divs = tk.dividends
        if divs is None or len(divs) == 0:
            return None
        cut = divs.index.max() - pd.Timedelta(days=365)
        ttm = float(divs[divs.index > cut].sum())
        hist = tk.history(period="5d")["Close"].dropna()
        if not len(hist):
            return None
        return ttm / float(hist.iloc[-1])
    except Exception:
        return None


def portfolio_dividend_yield(positions: pd.DataFrame, instruments: pd.DataFrame) -> float:
    """MV-weighted trailing dividend yield across market holdings (decimal)."""
    inst = instruments.set_index("ticker")
    num = den = 0.0
    for _, p in positions.iterrows():
        t = p["Ticker"]
        if t in inst.index and not inst.loc[t, "is_private"] and inst.loc[t, "yf_symbol"]:
            y = _trailing_yield(inst.loc[t, "yf_symbol"])
            if y is not None:
                num += p["Market Value"] * y
                den += p["Market Value"]
    return num / den if den else 0.0


def _ann_sharpe(monthly_ret, rf_monthly):
    d = pd.concat([monthly_ret.rename("r"), rf_monthly.rename("rf")], axis=1).dropna()
    if len(d) < 24:
        return None
    ex = d["r"] - d["rf"]
    return float(ex.mean() / ex.std() * np.sqrt(12)) if ex.std() else None


def sharpe_ratios(tx, instruments, period="3y") -> dict:
    """Annualized Sharpe for the portfolio (current holdings backtested) vs the
    benchmark, using monthly total returns and the Fama-French risk-free rate."""
    try:
        rf = _get_ff_factors()["RF"]
    except Exception:
        return {"portfolio": None, "benchmark": None, "benchmark_symbol": BENCHMARK_SYMBOL}

    def monthly(series):
        m = series.resample("ME").last().pct_change().dropna()
        m.index = m.index.to_period("M").to_timestamp("M")
        return m

    pv = portfolio_performance(tx, instruments, period)
    bench = pricelib.get_history([BENCHMARK_SYMBOL], period=period)
    bench = bench[BENCHMARK_SYMBOL] if BENCHMARK_SYMBOL in bench.columns else pd.Series(dtype=float)
    return {
        "portfolio": _ann_sharpe(monthly(pv), rf) if not pv.empty else None,
        "benchmark": _ann_sharpe(monthly(bench), rf) if not bench.empty else None,
        "benchmark_symbol": BENCHMARK_SYMBOL,
    }


def _get_ff_factors():
    """Developed Fama-French 5 factors + momentum (monthly, decimal, month-end index)."""
    f5 = _ff_csv("Developed_5_Factors", ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"])
    mom = _ff_csv("Developed_Mom_Factor", ["WML"])
    return f5.join(mom).rename(columns={"WML": "Mom"})


def _regress_fund(symbol, ff, period):
    px = pricelib.get_history([symbol], period=period)
    if px.empty or symbol not in px.columns:
        return None
    monthly = px[symbol].resample("ME").last().pct_change().dropna()
    monthly.index = monthly.index.to_period("M").to_timestamp("M")
    d = ff.copy()
    d["ret"] = monthly
    d = d.dropna()
    if len(d) < 12:
        return None
    y = (d["ret"] - d["RF"]).values
    X = np.column_stack([np.ones(len(d))] + [d[f].values for f in FACTORS])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    r2 = 1 - resid.var() / y.var() if y.var() else 0.0
    return {"alpha": float(beta[0]),
            "betas": {f: float(b) for f, b in zip(FACTORS, beta[1:])},
            "r2": float(r2), "n": int(len(d))}


def factor_exposure(positions: pd.DataFrame, instruments: pd.DataFrame, period="5y"):
    """Returns-based Fama-French factor loadings per fund and for the portfolio.

    Portfolio loadings are MV-weighted across market holdings; private holdings
    (no return history) are reported as an unattributed weight.
    """
    inst = instruments.set_index("ticker")
    ff = _get_ff_factors()
    total_mv = positions["Market Value"].sum()
    legs = []
    for _, p in positions.iterrows():
        t = p["Ticker"]
        if t in inst.index and not inst.loc[t, "is_private"] and inst.loc[t, "yf_symbol"]:
            legs.append((t, inst.loc[t, "yf_symbol"], float(p["Market Value"])))
    attributed = sum(mv for _, _, mv in legs) or 1.0
    per_fund = {}
    for t, sym, mv in legs:
        reg = _regress_fund(sym, ff, period)
        if reg:
            per_fund[t] = {"weight": mv / attributed, **reg}
    portfolio = {f: sum(d["weight"] * d["betas"][f] for d in per_fund.values())
                 for f in FACTORS}
    return {"factors": FACTORS, "per_fund": per_fund, "portfolio": portfolio,
            "unattributed": (total_mv - attributed) / total_mv if total_mv else 0.0,
            "window": (ff.index.min().strftime("%b %Y"),
                       ff.index.max().strftime("%b %Y"))}


# --- Look-through (§6 Phase 1) -------------------------------------------
# AVGE's published sub-ETF weights (renormalized to 100% in look_through).
AVGE_HOLDINGS = {
    "AVUS": 42.79, "AVLV": 15.21, "AVDE": 10.61, "AVEM": 6.94, "AVIV": 5.40,
    "AVUV": 3.48, "AVES": 3.46, "AVSC": 3.37, "AVRE": 2.73, "AVMV": 1.35,
}
# XEQT regional mix (iShares published geographic, approximate — editable).
XEQT_HOLDINGS = {
    "US Market": 0.45, "Canada Market": 0.245, "Intl Dev Market": 0.215, "EM Market": 0.09,
}
# (region, style) classification for each building block.
BLOCK_CLASS = {
    "AVUS": ("US", "Market"), "AVLV": ("US", "Value"), "AVUV": ("US", "Small Value"),
    "AVSC": ("US", "Small"), "AVMV": ("US", "Value"), "AVRE": ("US", "Real Estate"),
    "AVDE": ("Intl Dev", "Market"), "AVIV": ("Intl Dev", "Value"),
    "AVEM": ("EM", "Market"), "AVES": ("EM", "Value"),
    "US Market": ("US", "Market"), "Canada Market": ("Canada", "Market"),
    "Intl Dev Market": ("Intl Dev", "Market"), "EM Market": ("EM", "Market"),
}


def look_through(positions: pd.DataFrame, instruments: pd.DataFrame) -> dict:
    """Decompose the market sleeve into underlying building blocks and roll up
    region / style exposure (MV-weighted). Private holdings (OPO) are excluded."""
    inst = instruments.set_index("ticker")
    blocks, total = {}, 0.0
    for _, p in positions.iterrows():
        t, mv = p["Ticker"], float(p["Market Value"])
        if t not in inst.index or inst.loc[t, "is_private"]:
            continue
        total += mv
        if t == "AVGE":
            denom = sum(AVGE_HOLDINGS.values())
            for b, w in AVGE_HOLDINGS.items():
                blocks[b] = blocks.get(b, 0.0) + mv * w / denom
        elif t == "XEQT":
            for b, w in XEQT_HOLDINGS.items():
                blocks[b] = blocks.get(b, 0.0) + mv * w
        elif t == "XUS":
            blocks["US Market"] = blocks.get("US Market", 0.0) + mv
        else:
            blocks[t] = blocks.get(t, 0.0) + mv
    if not total:
        return {"blocks": {}, "region": {}, "style": {}}
    blocks = {b: v / total for b, v in blocks.items()}
    region, style = {}, {}
    for b, w in blocks.items():
        r, sty = BLOCK_CLASS.get(b, ("Other", "Market"))
        region[r] = region.get(r, 0.0) + w
        style[sty] = style.get(sty, 0.0) + w
    return {"blocks": blocks, "region": region, "style": style}


def current_block_weights(positions, instruments) -> dict:
    """Current look-through weights mapped onto the Avantis-sleeve basis.
    Regional market buckets map to the matching Avantis market sleeve; Canada
    (no Avantis sleeve) is kept separate as unclassified home-market exposure."""
    lt = look_through(positions, instruments)["blocks"]
    mapping = {"US Market": "AVUS", "Intl Dev Market": "AVDE", "EM Market": "AVEM"}
    out = {}
    for b, w in lt.items():
        out[mapping.get(b, b)] = out.get(mapping.get(b, b), 0.0) + w
    return out


def optimize_blocks(period="5y") -> dict:
    """Risk-based target weights over AVGE's 10 Avantis sleeves.

    Returns equal-weight, inverse-volatility (risk-parity proxy), and long-only
    minimum-variance allocations. Robust to differing fund inception dates:
    per-asset vol uses each fund's full history; covariance uses the common window.
    """
    syms = list(AVGE_HOLDINGS.keys())
    hist = pricelib.get_history(syms, period=period)
    if hist.empty:
        return {}
    monthly = hist.resample("ME").last().pct_change(fill_method=None)
    assets = [s for s in syms if s in monthly.columns
              and int(monthly[s].notna().sum()) >= 12]
    if len(assets) < 2:
        return {}
    arr = monthly[assets].to_numpy(dtype=float)           # rows=months, cols=assets
    n = len(assets)
    ew = np.full(n, 1.0 / n)
    vol = np.nanstd(arr, axis=0, ddof=1)                  # per-asset, own history
    vol = np.where(vol > 0, vol, np.nan)
    iv = 1.0 / vol
    iv = iv / np.nansum(iv)
    common = arr[~np.isnan(arr).any(axis=1)]              # aligned window for covariance
    if len(common) >= n + 2:
        cov = np.cov(common, rowvar=False)
        cov = 0.85 * cov + 0.15 * np.diag(np.diag(cov))   # light shrinkage to diagonal
        try:
            w = np.linalg.pinv(cov) @ np.ones(n)
            w = np.clip(w, 0, None)
            mv = w / w.sum() if w.sum() > 0 else ew
        except Exception:
            mv = ew
    else:
        mv = np.nan_to_num(iv, nan=0.0)                   # not enough overlap for cov
    cov_months = int(len(common))

    def series(a):
        return pd.Series(np.nan_to_num(a, nan=0.0), index=assets)

    return {"assets": assets, "equal": series(ew), "invvol": series(iv),
            "minvar": series(mv), "vol": series(vol), "cov_months": cov_months}


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
