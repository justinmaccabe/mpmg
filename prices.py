"""Market data via yfinance. All functions degrade gracefully if offline."""
import time

import pandas as pd
import yfinance as yf


def get_history(symbols, period="1y", retries=2) -> pd.DataFrame:
    """Daily close prices for a list of yfinance symbols. Columns = symbols.

    Retries transient failures (common on a cold container / first load) and
    returns an empty frame rather than raising, so callers never crash.
    """
    symbols = [s for s in symbols if s]
    if not symbols:
        return pd.DataFrame()
    for attempt in range(retries + 1):
        try:
            data = yf.download(symbols, period=period, auto_adjust=True,
                               progress=False, group_by="column")
            if data is None or data.empty:
                raise ValueError("empty response")
            if isinstance(data.columns, pd.MultiIndex):
                close = data["Close"]
            else:  # single symbol -> flat columns
                close = data[["Close"]].rename(columns={"Close": symbols[0]})
            return close.dropna(how="all")
        except Exception:
            if attempt < retries:
                time.sleep(1.5)
            else:
                return pd.DataFrame()


def get_current_and_prev(symbols) -> pd.DataFrame:
    """Return DataFrame indexed by symbol with columns: price, prev_close.

    'price' is the latest close, 'prev_close' the one before it (for daily P&L).
    """
    hist = get_history(symbols, period="5d")
    rows = {}
    for s in symbols:
        if s and s in hist.columns:
            col = hist[s].dropna()
            if len(col) >= 2:
                rows[s] = dict(price=float(col.iloc[-1]), prev_close=float(col.iloc[-2]))
            elif len(col) == 1:
                rows[s] = dict(price=float(col.iloc[-1]), prev_close=float(col.iloc[-1]))
    return pd.DataFrame.from_dict(rows, orient="index")


def get_fx_series(period="5y") -> pd.Series:
    """Daily USD→CAD rate history. Empty series if offline."""
    hist = get_history(["USDCAD=X"], period=period)
    if "USDCAD=X" in hist.columns:
        return hist["USDCAD=X"].dropna()
    return pd.Series(dtype=float)


def get_usd_cad() -> float:
    """Current USD->CAD rate. Falls back to a recent constant if offline."""
    try:
        col = get_history(["USDCAD=X"], period="5d")["USDCAD=X"].dropna()
        if len(col):
            return float(col.iloc[-1])
    except Exception:
        pass
    return 1.40
