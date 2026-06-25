"""Record one daily portfolio snapshot. Run by the GitHub Action twice each
weekday — ~10am ET ("open") and ~5pm ET ("close") — or manually.

Open vs close is inferred from the UTC hour (the 14:05 UTC run is the open; the
21:05 UTC run is the close). Both are kept: the morning sets market_value_open,
the evening sets market_value (close) while preserving the morning's open.
"""
import os
import sys
import datetime as dt

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import db
import portfolio


def _today_row():
    snaps = db.get_snapshots_df()
    if snaps.empty:
        return None
    today = dt.date.today()
    match = snaps[pd.to_datetime(snaps["date"]).dt.date == today]
    return match.iloc[-1] if len(match) else None


def main():
    db.init_db()
    tx = db.get_transactions_df()
    inst = db.get_instruments_df()
    _, totals = portfolio.build_portfolio(tx, inst)
    if not totals:
        print("No positions; nothing to record.")
        return
    mv = totals["market_value"]
    is_open = dt.datetime.utcnow().hour < 18      # 14:05 UTC = open, 21:05 = close
    prev = _today_row()

    if is_open:
        mv_open, mv_close = mv, mv                # close placeholder until the close run
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
    print(f"Recorded {row['date']} ({'open' if is_open else 'close'}): "
          f"open={mv_open:.2f} close={mv_close:.2f}")


if __name__ == "__main__":
    main()
