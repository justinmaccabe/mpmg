"""Record one daily portfolio snapshot. Run by the GitHub Action each weekday at
9:35am ET ("open") and ~4–5pm ET ("close") — or manually.

The open targets 9:35am ET year-round. GitHub cron is UTC and DST-blind, so the
workflow schedules the open twice (13:35 UTC for EDT, 14:35 UTC for EST) and
passes the triggering cron string in $SCHEDULE_CRON. This script keeps only the
open cron matching the current ET offset and skips the other, so exactly one
open is recorded regardless of season — and the decision is based on the
*scheduled* cron, not the actual run time, so GitHub's scheduling delays can't
misclassify it. Both points are kept: the morning sets market_value_open, the
evening sets market_value (close) while preserving the morning's open.
"""
import os
import sys
import datetime as dt
from zoneinfo import ZoneInfo

import pandas as pd

ET = ZoneInfo("America/New_York")
OPEN_CRON_HOURS = {13, 14}          # the EDT/EST UTC variants of the 9:35 ET open


def _resolve_slot():
    """Return 'open', 'close', or None (skip) for this invocation.

    Scheduled runs are classified by the cron that fired ($SCHEDULE_CRON): the
    DST-correct open cron → 'open', the wrong-DST open cron → None (skip), any
    other cron → 'close'. Manual runs (no cron) are classified by ET hour.
    """
    cron = os.environ.get("SCHEDULE_CRON", "").strip()
    if not cron:
        return "open" if dt.datetime.now(ET).hour < 12 else "close"
    cron_hour = int(cron.split()[1])
    if cron_hour in OPEN_CRON_HOURS:
        # UTC hour at which 9:35 ET falls today, given the current DST offset
        want = (dt.datetime.now(ET).replace(hour=9, minute=35, second=0, microsecond=0)
                .astimezone(dt.timezone.utc).hour)
        return "open" if cron_hour == want else None
    return "close"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import db
import portfolio
import prices as pricelib


def _persist_last_prices():
    """Save each holding's current live price as its last-known fallback."""
    inst = db.get_instruments_df()
    syms = [s for s in inst["yf_symbol"].dropna().tolist()]
    q = pricelib.get_current_and_prev(syms)
    for _, r in inst.iterrows():
        sym = r["yf_symbol"]
        if sym and sym in q.index and pd.notna(q.loc[sym, "price"]):
            db.set_last_price(r["ticker"], float(q.loc[sym, "price"]))


def _today_row():
    snaps = db.get_snapshots_df()
    if snaps.empty:
        return None
    today = dt.date.today()
    match = snaps[pd.to_datetime(snaps["date"]).dt.date == today]
    return match.iloc[-1] if len(match) else None


def main():
    slot = _resolve_slot()
    if slot is None:
        print(f"Skipping: {os.environ.get('SCHEDULE_CRON', '').strip()} is not the "
              "DST-correct open run for the current ET offset.")
        return

    db.init_db()
    tx = db.get_transactions_df()
    inst = db.get_instruments_df()
    _, totals = portfolio.build_portfolio(tx, inst)
    if not totals:
        print("No positions; nothing to record.")
        return
    _persist_last_prices()
    mv = totals["market_value"]
    is_open = slot == "open"
    prev = _today_row()

    if is_open:
        mv_open, mv_close = mv, None              # close stays empty until the close run
    else:
        mv_close = mv
        prev_open = None if prev is None else prev.get("market_value_open")
        mv_open = float(prev_open) if prev_open is not None and pd.notna(prev_open) else mv

    row = {
        "date": dt.date.today(),
        "market_value": mv_close,
        "market_value_open": mv_open,
        "book_value": totals["book_value"],
        "gain_loss": totals["gain_loss"],
        "gain_loss_pct": totals["gain_loss_pct"],
        "daily_pnl": totals["daily_pnl"],
        "daily_pnl_pct": totals["daily_pnl_pct"],
        "benchmark_pct": portfolio.benchmark_daily_pct(),
    }
    db.upsert_snapshot(row)
    close_str = "pending" if mv_close is None else f"{mv_close:.2f}"
    print(f"Recorded {row['date']} ({slot}): open={mv_open:.2f} close={close_str}")


if __name__ == "__main__":
    main()
