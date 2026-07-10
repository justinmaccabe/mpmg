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
    """Return from the last close strictly before `since` to the latest value.

    Requires at least one price on/after `since` — some feeds (e.g. TSX-listed
    tickers on yfinance) lag a day behind, and without this check a stale
    "latest" price equal to the pre-cutoff price would misreport as a 0% return
    instead of "no fresh data yet".
    """
    if series is None or len(series) == 0:
        return None
    since_ts = pd.Timestamp(since)
    if series.index[-1] < since_ts:
        return None
    base = series[series.index < since_ts]
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


# Price-history window fetched for each attribution period.
ATTRIB_FETCH = {"WTD": "1mo", "MTD": "3mo", "3M": "6mo", "1Y": "1y", "3Y": "3y"}


def return_attribution(tx, instruments, period="MTD") -> dict:
    """Decompose the portfolio's period price return into per-holding
    contributions (CAD): contribution_i = beginning weight_i x return_i, which
    sum to the total. Uses current share counts and current FX (consistent with
    the performance chart); private holdings (OPO) are excluded — no history.
    """
    pos = compute_positions(tx)
    inst = instruments.set_index("ticker")
    usd_cad = pricelib.get_usd_cad()
    legs = []  # (ticker, yf_symbol, shares, fx)
    for _, p in pos.iterrows():
        t = p["ticker"]
        if t not in inst.index or p["shares"] == 0:
            continue
        meta = inst.loc[t]
        if meta["is_private"] or not meta["yf_symbol"]:
            continue
        legs.append((t, meta["yf_symbol"], float(p["shares"]),
                     usd_cad if meta["currency"] == "USD" else 1.0))
    empty = {"rows": [], "port_return": None, "gain_total": None, "period": period}
    if not legs:
        return empty
    hist = pricelib.get_history([s for _, s, _, _ in legs],
                                period=ATTRIB_FETCH.get(period, "1y"))
    if hist.empty:
        return empty
    # Align every leg to dates where ALL of them have a price (some feeds, e.g.
    # TSX-listed tickers on yfinance, lag a day behind US-listed ones) — so
    # every holding's "as of" price is the same date, not a mix of stale and
    # fresh closes that would make one leg look flat next to another's real move.
    cols = [s for _, s, _, _ in legs if s in hist.columns]
    hist = hist[cols].dropna()
    if hist.empty:
        return empty

    today = dt.date.today()
    cut = None
    if period == "WTD":
        cut = pd.Timestamp(today - dt.timedelta(days=today.weekday()))   # this Monday
    elif period == "MTD":
        cut = pd.Timestamp(today.replace(day=1))                         # 1st of month

    rows, begin_total, end_total = [], 0.0, 0.0
    for t, sym, shares, fx in legs:
        if sym not in hist.columns:
            continue
        s = hist[sym]
        if s.empty:
            continue
        if cut is not None:
            base = s[s.index < cut]
            p_begin = float(base.iloc[-1]) if not base.empty else float(s.iloc[0])
        else:
            p_begin = float(s.iloc[0])
        p_end = float(s.iloc[-1])
        bv, ev = shares * p_begin * fx, shares * p_end * fx
        begin_total += bv
        end_total += ev
        rows.append({"ticker": t, "begin_value": bv, "end_value": ev,
                     "gain": ev - bv, "ret": (p_end / p_begin - 1) if p_begin else 0.0})
    if begin_total <= 0:
        return empty
    for r in rows:
        r["weight"] = r["begin_value"] / begin_total
        r["contribution"] = r["gain"] / begin_total      # to portfolio return
    rows.sort(key=lambda r: r["contribution"], reverse=True)
    return {"rows": rows, "gain_total": end_total - begin_total,
            "port_return": (end_total - begin_total) / begin_total, "period": period}


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


# --- Currency basis -----------------------------------------------------------
def _is_cad(symbol: str) -> bool:
    """CAD-denominated if TSX-listed; everything else here trades in USD."""
    return str(symbol).endswith(".TO")


def _fx_for(index, period="5y") -> pd.Series:
    """USD→CAD series aligned to a price index; constant fallback if offline."""
    fx = pricelib.get_fx_series(period)
    if fx.empty:
        return pd.Series(pricelib.get_usd_cad(), index=index)
    return fx.reindex(index).ffill().bfill()


def hist_in_cad(hist: pd.DataFrame, period="5y") -> pd.DataFrame:
    """Convert USD-listed columns of a price history to CAD."""
    if hist.empty:
        return hist
    usd_cols = [c for c in hist.columns if not _is_cad(c)]
    if not usd_cols:
        return hist
    out = hist.copy()
    fx = _fx_for(hist.index, period)
    for c in usd_cols:
        out[c] = out[c] * fx
    return out


def hist_in_usd(hist: pd.DataFrame, period="5y") -> pd.DataFrame:
    """Convert CAD-listed columns of a price history to USD."""
    if hist.empty:
        return hist
    cad_cols = [c for c in hist.columns if _is_cad(c)]
    if not cad_cols:
        return hist
    out = hist.copy()
    fx = _fx_for(hist.index, period)
    for c in cad_cols:
        out[c] = out[c] / fx
    return out


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
    px = hist_in_usd(px, period)     # FF factors are USD; CAD funds must match
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


# --- Black-Litterman target on the full opportunity set (§6, AQR views) -----
# AQR 2026 Capital Market Assumptions (Alt Thinking 2026 Issue 1, as of Dec 31
# 2025). Medium-term (5-10Y) expected LOCAL REAL EXCESS-OF-CASH returns from
# Exhibit A1; real estate from Exhibit A5. Editable — these are the view inputs.
AQR_REGION_ER = {"US": 0.026, "Canada": 0.035, "Intl Dev": 0.045, "EM": 0.041}
AQR_REALESTATE_ER = 0.018            # US real estate real return 3.1% less ~1.3% real cash
# Per-unit factor premia, calibrated from AQR's stated long-only style premia
# (value +0.5%, integrated multifactor +1.0%) and the US Small−Large gap (+1.2%).
AQR_LAMBDA = {"SMB": 0.012, "HML": 0.017, "RMW": 0.010, "Mom": 0.010}
# Style-based fallback tilt (excess return) when a fund's regression is unavailable.
_STYLE_FALLBACK = {"Value": 0.005, "Small": 0.012, "Small Value": 0.017,
                   "Real Estate": 0.0, "Market": 0.0}

# Full opportunity set: (label, price proxy, region, style). Broad regional beta
# is carried by AVUS/AVDE/AVEM/Canada; the rest are genuine factor tilts.
BL_UNIVERSE = [
    ("AVUS", "AVUS", "US", "Market"),
    ("Canada Market", "XIC.TO", "Canada", "Market"),
    ("AVDE", "AVDE", "Intl Dev", "Market"),
    ("AVEM", "AVEM", "EM", "Market"),
    ("AVLV", "AVLV", "US", "Value"),
    ("AVUV", "AVUV", "US", "Small Value"),
    ("AVSC", "AVSC", "US", "Small"),
    ("AVMV", "AVMV", "US", "Value"),
    ("AVRE", "AVRE", "US", "Real Estate"),
    ("AVIV", "AVIV", "Intl Dev", "Value"),
    ("AVES", "AVES", "EM", "Value"),
]
# Market-cap prior weights (the BL anchor): regional beta only, no factor tilt.
BL_PRIOR = {"AVUS": 0.63, "Canada Market": 0.03, "AVDE": 0.24, "AVEM": 0.10}


def _bl_expected_returns(assets, ff, period):
    """Excess-of-cash expected return per asset: AQR regional base + factor tilt.
    Tilt uses the fund's FF5+momentum loadings where a regression is available,
    else a style-based fallback."""
    er = {}
    for label, sym, region, style in BL_UNIVERSE:
        if label not in assets:
            continue
        if style == "Real Estate":
            er[label] = AQR_REALESTATE_ER
            continue
        base = AQR_REGION_ER.get(region, 0.03)
        if style == "Market":
            er[label] = base
            continue
        reg = _regress_fund(sym, ff, period)
        if reg:
            tilt = sum(AQR_LAMBDA.get(f, 0.0) * reg["betas"].get(f, 0.0)
                       for f in AQR_LAMBDA)
        else:
            tilt = _STYLE_FALLBACK.get(style, 0.0)
        er[label] = base + tilt
    return er


def black_litterman_target(period="5y", confidence=0.5) -> dict:
    """Return-aware target weights over the full opportunity set via Black-
    Litterman: market-cap prior blended with AQR capital-market-assumption views.

    confidence in [0,1]: 0 collapses to the market-cap prior, 1 lets the AQR
    views dominate. Long-only, weights sum to 1.
    """
    labels = [u[0] for u in BL_UNIVERSE]
    proxies = {u[0]: u[1] for u in BL_UNIVERSE}
    hist = pricelib.get_history(list(proxies.values()), period=period)
    hist = hist_in_cad(hist, period)          # common CAD basis across the universe
    if hist.empty:
        return {}
    monthly = hist.resample("ME").last().pct_change(fill_method=None)
    assets = [l for l in labels if proxies[l] in monthly.columns
              and int(monthly[proxies[l]].notna().sum()) >= 12]
    if len(assets) < 3:
        return {}
    cols = [proxies[l] for l in assets]
    arr = monthly[cols].to_numpy(dtype=float)
    common = arr[~np.isnan(arr).any(axis=1)]
    n = len(assets)
    if len(common) < n + 2:
        return {}
    cov = np.cov(common, rowvar=False) * 12.0                  # annualized
    cov = 0.75 * cov + 0.25 * np.diag(np.diag(cov))            # Ledoit-Wolf-style shrink

    pi = np.array([BL_PRIOR.get(l, 0.0) for l in assets])
    if pi.sum() <= 0:
        return {}
    pi = pi / pi.sum()

    ff = _get_ff_factors()
    er = _bl_expected_returns(assets, ff, period)
    q = np.array([er.get(l, 0.0) for l in assets])

    # risk aversion δ anchored so the prior reproduces its cap-weighted AQR return
    mkt_var = float(pi @ cov @ pi)
    target_excess = float(pi @ q)
    delta = target_excess / mkt_var if mkt_var > 0 else 3.0
    implied = delta * cov @ pi                                 # equilibrium returns Π

    idx = pd.Index(assets)
    c = min(max(float(confidence), 0.0), 1.0)
    if c <= 0:                                                 # no views → market-cap anchor
        w = pi.copy()
    else:
        tau = 0.05
        tau_cov = tau * cov
        omega = np.diag(np.diag(tau_cov)) / c                  # view uncertainty (P=I)
        inv_tau = np.linalg.pinv(tau_cov)
        inv_om = np.linalg.pinv(omega)
        mu = np.linalg.pinv(inv_tau + inv_om) @ (inv_tau @ implied + inv_om @ q)
        w = np.linalg.pinv(delta * cov) @ mu
        w = np.clip(w, 0.0, None)
        w = w / w.sum() if w.sum() > 0 else pi
    return {"assets": assets,
            "target": pd.Series(w, index=idx),
            "prior": pd.Series(pi, index=idx),
            "implied": pd.Series(implied, index=idx),
            "views": pd.Series(q, index=idx),
            "cov_months": int(len(common)), "delta": float(delta)}


# --- Custom policy benchmark (IPS §6.1) --------------------------------------
# Policy weights applied to investable proxies, fixed weights, daily rebalance.
POLICY_BENCHMARK = {
    "AVUS": 0.35, "AVDE": 0.22, "AVIV": 0.16, "XIC.TO": 0.11,
    "AVEM": 0.08, "AVES": 0.05, "AVUV": 0.015, "AVLV": 0.015,
}


def policy_benchmark(period="1y") -> pd.Series:
    """Return series of the §6.1 policy portfolio, rebased to 100.

    Actual − policy = implementation gap (should sit near zero);
    policy − S&P 500 = the allocation choice (intentional, judged over years).
    """
    syms = list(POLICY_BENCHMARK)
    hist = pricelib.get_history(syms, period=period)
    if hist.empty:
        return pd.Series(dtype=float)
    have = [s for s in syms if s in hist.columns and hist[s].notna().sum() > 20]
    if not have:
        return pd.Series(dtype=float)
    rets = hist[have].pct_change(fill_method=None).dropna()
    if rets.empty:
        return pd.Series(dtype=float)
    w = np.array([POLICY_BENCHMARK[s] for s in have])
    w = w / w.sum()
    port = (rets * w).sum(axis=1)
    value = (1 + port).cumprod()
    return value / value.iloc[0] * 100


# --- Performance measurement (PM suite) --------------------------------------
def twr_series(snapshots: pd.DataFrame) -> pd.Series:
    """True time-weighted return index from recorded daily P&L percentages.

    daily_pnl_pct is price-move-only (shares × price change), so contributions
    never register as return. Chain-linked, rebased to 100 at tracking start.
    """
    if snapshots is None or snapshots.empty or "daily_pnl_pct" not in snapshots:
        return pd.Series(dtype=float)
    s = snapshots.dropna(subset=["daily_pnl_pct"]).copy()
    if s.empty:
        return pd.Series(dtype=float)
    s["date"] = pd.to_datetime(s["date"])
    s = s.sort_values("date").set_index("date")
    return (1 + s["daily_pnl_pct"].astype(float)).cumprod() * 100


def xirr(flows) -> float:
    """Annualized money-weighted return via bisection.

    flows: list of (date, amount); negative = money in, positive = value out.
    Returns None when a root can't be bracketed (e.g. all one sign).
    """
    flows = sorted(flows, key=lambda f: f[0])
    if len(flows) < 2:
        return None
    amts = [a for _, a in flows]
    if all(a >= 0 for a in amts) or all(a <= 0 for a in amts):
        return None
    d0 = flows[0][0]

    def npv(rate):
        return sum(a / (1 + rate) ** ((d - d0).days / 365.25) for d, a in flows)

    lo, hi = -0.95, 10.0
    f_lo, f_hi = npv(lo), npv(hi)
    if f_lo * f_hi > 0:
        return None
    for _ in range(100):
        mid = (lo + hi) / 2
        f_mid = npv(mid)
        if abs(f_mid) < 1e-8:
            return mid
        if f_lo * f_mid < 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    return (lo + hi) / 2


def money_weighted_return(contribs: pd.DataFrame, market_value: float) -> dict:
    """XIRR from recorded contributions to today's total value."""
    if contribs is None or contribs.empty or not market_value:
        return {}
    flows = [(pd.Timestamp(d).date(), -float(a))
             for d, a in zip(contribs["date"], contribs["amount"])]
    today = dt.date.today()
    flows = [f for f in flows if f[0] <= today]
    flows.append((today, float(market_value)))
    total_in = float(contribs["amount"].sum())
    return {"xirr": xirr(flows), "contributed": total_in,
            "growth": float(market_value) - total_in,
            "since": min(f[0] for f in flows)}


def calendar_returns(tx, instruments, period="5y") -> pd.DataFrame:
    """Monthly return grid (years × months + Year column), modeled: current
    holdings backtested in CAD, OPO excluded."""
    pp = portfolio_performance(tx, instruments, period)
    if pp.empty:
        return pd.DataFrame()
    m = pp.resample("ME").last().pct_change().dropna()
    if m.empty:
        return pd.DataFrame()
    df = pd.DataFrame({"year": m.index.year, "month": m.index.month, "ret": m.values})
    grid = df.pivot(index="year", columns="month", values="ret")
    grid = grid.reindex(columns=range(1, 13))
    grid.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    grid["Year"] = (1 + grid.fillna(0)).prod(axis=1) - 1
    return grid


# --- Next-dollar allocator (trader tool) --------------------------------------
# §6.1 policy on the sleeve basis used by current_block_weights.
POLICY_SLEEVES = {"AVUS": 0.35, "AVDE": 0.22, "AVIV": 0.16, "Canada Market": 0.11,
                  "AVEM": 0.08, "AVES": 0.05, "AVUV": 0.015, "AVLV": 0.015}
SLEEVE_VEHICLE = {"AVUS": "XUS / AVUS", "AVDE": "AVDE", "AVIV": "AVIV",
                  "Canada Market": "XIC", "AVEM": "AVEM", "AVES": "AVES",
                  "AVUV": "AVUV", "AVLV": "AVLV"}


def next_dollar(positions, instruments, amount: float) -> pd.DataFrame:
    """Buy-only allocation of new cash toward the §6.1 policy weights.

    Computes each sleeve's dollar shortfall against the policy at the post-
    contribution total and splits the contribution pro-rata across shortfalls —
    the fastest buy-only path back toward policy (never sells).
    """
    if amount <= 0:
        return pd.DataFrame()
    cur = current_block_weights(positions, instruments)
    inst = instruments.set_index("ticker")
    mkt_mv = float(sum(
        p["Market Value"] for _, p in positions.iterrows()
        if p["Ticker"] in inst.index and not inst.loc[p["Ticker"], "is_private"]))
    if mkt_mv <= 0:
        return pd.DataFrame()
    total_after = mkt_mv + amount
    rows = []
    for sleeve, w_pol in POLICY_SLEEVES.items():
        cur_d = cur.get(sleeve, 0.0) * mkt_mv
        need = max(0.0, w_pol * total_after - cur_d)
        rows.append({"sleeve": sleeve, "current_w": cur_d / mkt_mv,
                     "policy_w": w_pol, "need": need})
    total_need = sum(r["need"] for r in rows)
    for r in rows:
        r["buy"] = amount * r["need"] / total_need if total_need > 0 else 0.0
        r["vehicle"] = SLEEVE_VEHICLE.get(r["sleeve"], r["sleeve"])
    out = pd.DataFrame(rows).sort_values("buy", ascending=False)
    return out[out["buy"] > 0.005]


# --- Risk & return statistics (PortfolioVisualizer-style suite) -------------
def risk_stats(tx, instruments, period="3y") -> dict:
    """Monthly risk/return metrics for the modeled portfolio vs the benchmark.

    Backtests current holdings (OPO excluded), resamples to monthly total
    returns, and computes the standard institutional suite. Risk-free is the
    Fama-French RF series where the windows overlap, else 0.
    """
    pp = portfolio_performance(tx, instruments, period)
    if pp.empty or len(pp) < 40:
        return {}
    bench_hist = pricelib.get_history([BENCHMARK_SYMBOL], period=period)
    b_daily = bench_hist[BENCHMARK_SYMBOL] if BENCHMARK_SYMBOL in bench_hist else None

    def _monthly(series):
        m = series.resample("ME").last().pct_change().dropna()
        m.index = m.index.to_period("M").to_timestamp("M")
        return m

    p = _monthly(pp)
    if len(p) < 6:
        return {}
    b = _monthly(b_daily) if b_daily is not None else pd.Series(dtype=float)
    both = pd.concat([p.rename("p"), b.rename("b")], axis=1).dropna()

    try:
        rf = _get_ff_factors()["RF"].reindex(p.index).fillna(0.0)
    except Exception:
        rf = pd.Series(0.0, index=p.index)
    ex = p - rf

    r = p.values
    mean_mo = float(np.mean(r))
    geo_mo = float(np.prod(1 + r) ** (1 / len(r)) - 1)
    std_mo = float(np.nanstd(r, ddof=1))
    downside = np.minimum(r, 0.0)
    dd_mo = float(np.sqrt(np.mean(downside ** 2)))
    ann = 12.0
    geo_ann = (1 + geo_mo) ** ann - 1
    std_ann = std_mo * np.sqrt(ann)
    mdd = max_drawdown(pp)

    ex_ann = float(np.mean(ex.values)) * ann
    sharpe = ex_ann / std_ann if std_ann > 0 else None
    sortino = (ex_ann / (dd_mo * np.sqrt(ann))) if dd_mo > 0 else None
    calmar = (geo_ann / abs(mdd)) if mdd else None

    z = (r - mean_mo) / std_mo if std_mo > 0 else np.zeros_like(r)
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4) - 3.0)
    var_h = float(-np.percentile(r, 5))
    var_a = float(-(mean_mo - 1.645 * std_mo))
    tail = r[r <= np.percentile(r, 5)]
    cvar = float(-np.mean(tail)) if len(tail) else None

    beta = alpha_ann = r2 = corr = te = ir = active = treynor = m2 = None
    up_cap = down_cap = bench_sharpe = None
    if len(both) >= 6:
        pv, bv = both["p"].values, both["b"].values
        rf_b = rf.reindex(both.index).fillna(0.0).values
        pe, be = pv - rf_b, bv - rf_b
        vb = float(np.nanstd(bv, ddof=1))
        cov_pb = float(np.cov(pe, be)[0, 1])
        var_b = float(np.var(be, ddof=1))
        if var_b > 0:
            beta = cov_pb / var_b
            alpha_ann = (np.mean(pe) - beta * np.mean(be)) * ann
            treynor = ex_ann / beta if beta else None
        corr = float(np.corrcoef(pv, bv)[0, 1])
        r2 = corr ** 2 if corr is not None else None
        diff = pv - bv
        te = float(np.nanstd(diff, ddof=1)) * np.sqrt(ann)
        geo_b = float(np.prod(1 + bv) ** (ann / len(bv)) - 1)
        active = geo_ann - geo_b
        ir = active / te if te and te > 0 else None
        if sharpe is not None and vb > 0:
            m2 = sharpe * vb * np.sqrt(ann) + float(np.mean(rf_b)) * ann
        if vb > 0:
            bench_sharpe = float(np.mean(be)) * ann / (vb * np.sqrt(ann))
        up_m, dn_m = bv > 0, bv < 0
        if up_m.any() and float(np.mean(bv[up_m])):
            up_cap = float(np.mean(pv[up_m]) / np.mean(bv[up_m]))
        if dn_m.any() and float(np.mean(bv[dn_m])):
            down_cap = float(np.mean(pv[dn_m]) / np.mean(bv[dn_m]))

    gains, losses = r[r > 0], r[r < 0]
    gain_loss = (float(np.mean(gains) / abs(np.mean(losses)))
                 if len(gains) and len(losses) else None)

    return {
        "months": int(len(r)),
        "Arithmetic mean (monthly)": mean_mo,
        "Arithmetic mean (annualized)": mean_mo * ann,
        "Geometric mean (annualized)": geo_ann,
        "Standard deviation (annualized)": std_ann,
        "Downside deviation (monthly)": dd_mo,
        "Maximum drawdown": mdd,
        "Benchmark correlation": corr, "Beta": beta,
        "Alpha (annualized)": alpha_ann, "R²": r2,
        "Sharpe ratio": sharpe, "Benchmark Sharpe": bench_sharpe,
        "Sortino ratio": sortino,
        "Treynor ratio (%)": treynor * 100 if treynor is not None else None,
        "Calmar ratio": calmar,
        "Modigliani–Modigliani (M²)": m2,
        "Active return (annualized)": active,
        "Tracking error (annualized)": te,
        "Information ratio": ir,
        "Skewness": skew, "Excess kurtosis": kurt,
        "Historical VaR 5% (monthly)": var_h,
        "Analytical VaR 5% (monthly)": var_a,
        "Conditional VaR 5% (monthly)": cvar,
        "Upside capture": up_cap, "Downside capture": down_cap,
        "Positive periods": f"{int((r > 0).sum())} of {len(r)} "
                            f"({(r > 0).mean():.0%})",
        "Gain/loss ratio": gain_loss,
    }


def block_correlation(period="5y") -> pd.DataFrame:
    """Correlation matrix of monthly returns across the BL opportunity set."""
    proxies = {u[0]: u[1] for u in BL_UNIVERSE}
    hist = pricelib.get_history(list(proxies.values()), period=period)
    hist = hist_in_cad(hist, period)          # common CAD basis across the universe
    if hist.empty:
        return pd.DataFrame()
    monthly = hist.resample("ME").last().pct_change(fill_method=None)
    cols = {sym: lbl for lbl, sym in proxies.items() if sym in monthly.columns}
    m = monthly[list(cols)].rename(columns=cols)
    m = m.dropna(axis=1, thresh=12)
    return m.corr()


def efficient_frontier(period="5y", n_samples=6000, current_weights=None) -> dict:
    """Long-only efficient frontier over the full opportunity set.

    Expected returns are the AQR-based views (real, excess of cash) used by the
    BL target; covariance is historical monthly, annualized. The frontier is
    traced from closed-form minimum-variance solutions per target return
    (clipped long-only) sharpened with a Dirichlet sample cloud.
    """
    labels = [u[0] for u in BL_UNIVERSE]
    proxies = {u[0]: u[1] for u in BL_UNIVERSE}
    hist = pricelib.get_history(list(proxies.values()), period=period)
    hist = hist_in_cad(hist, period)          # common CAD basis across the universe
    if hist.empty:
        return {}
    monthly = hist.resample("ME").last().pct_change(fill_method=None)
    assets = [l for l in labels if proxies[l] in monthly.columns
              and int(monthly[proxies[l]].notna().sum()) >= 12]
    if len(assets) < 3:
        return {}
    arr = monthly[[proxies[l] for l in assets]].to_numpy(dtype=float)
    common = arr[~np.isnan(arr).any(axis=1)]
    n = len(assets)
    if len(common) < n + 2:
        return {}
    # realized vol from each asset's own full history (the common window is
    # only as long as the youngest fund and understates seasoned funds' risk);
    # correlations from the common window, lightly shrunk toward identity
    vol_own = np.nanstd(arr, axis=0, ddof=1) * np.sqrt(12.0)
    corr = np.corrcoef(common, rowvar=False)
    corr = 0.75 * corr + 0.25 * np.eye(n)
    cov = np.outer(vol_own, vol_own) * corr
    ff = _get_ff_factors()
    er = _bl_expected_returns(assets, ff, period)
    mu = np.array([er.get(l, 0.0) for l in assets])

    cands = [np.eye(n)[i] for i in range(n)]                    # single assets
    inv = np.linalg.pinv(cov)
    ones = np.ones(n)
    A = np.array([[ones @ inv @ ones, ones @ inv @ mu],
                  [mu @ inv @ ones, mu @ inv @ mu]])
    Ainv = np.linalg.pinv(A)
    for t in np.linspace(mu.min(), mu.max(), 60):               # closed-form per target
        lam = Ainv @ np.array([1.0, t])
        w = inv @ (lam[0] * ones + lam[1] * mu)
        w = np.clip(w, 0, None)
        if w.sum() > 0:
            cands.append(w / w.sum())
    rng = np.random.default_rng(7)
    cands.extend(rng.dirichlet(0.35 * ones, size=n_samples))
    W = np.array(cands)
    rets = W @ mu
    vols = np.sqrt(np.einsum("ij,jk,ik->i", W, cov, W))

    # Upper concave hull of the candidate cloud — the true frontier shape. A
    # running-max envelope keeps every marginally-better sample and draws a
    # staircase; the hull keeps only the vertices that dominate.
    order = np.argsort(vols)
    hull = []
    for i in order:
        x, y = float(vols[i]), float(rets[i])
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1) >= 0:
                hull.pop()
            else:
                break
        hull.append((x, y))
    peak = max(range(len(hull)), key=lambda j: hull[j][1])
    hull = hull[:peak + 1]                       # stop at max return; no downslope
    f = pd.DataFrame(hull, columns=["vol", "ret"])
    tang = f.loc[(f["ret"] / f["vol"]).idxmax()] if len(f) else None

    def point(w_map):
        w = np.array([w_map.get(l, 0.0) for l in assets])
        if w.sum() <= 0:
            return None
        w = w / w.sum()
        return {"vol": float(np.sqrt(w @ cov @ w)), "ret": float(w @ mu)}

    return {"assets": assets, "frontier": f,
            "asset_pts": pd.DataFrame({"label": assets, "vol": np.sqrt(np.diag(cov)),
                                       "ret": mu}),
            "tangency": None if tang is None else
                        {"vol": float(tang["vol"]), "ret": float(tang["ret"])},
            "prior_pt": point(BL_PRIOR),
            "current_pt": point(current_weights) if current_weights else None,
            "months": int(len(common))}


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
    legs = []  # (yf_symbol, shares)
    for _, p in pos.iterrows():
        t = p["ticker"]
        if t not in inst.index or p["shares"] == 0:
            continue
        meta = inst.loc[t]
        if meta["is_private"] or not meta["yf_symbol"]:
            continue
        legs.append((meta["yf_symbol"], p["shares"]))
    if not legs:
        return pd.Series(dtype=float)
    hist = pricelib.get_history([s for s, _ in legs], period=period)
    if hist.empty:
        return pd.Series(dtype=float)
    hist = hist_in_cad(hist, period)     # true CAD basis: FX moves are returns
    value = None
    for sym, shares in legs:
        if sym in hist.columns:
            leg = hist[sym] * shares
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
