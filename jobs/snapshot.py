"""Record one daily portfolio snapshot. Run by the GitHub Action each weekday,
or manually:  python jobs/snapshot.py

Idempotent: re-running on the same day overwrites that day's row.
"""
import os
import sys
import datetime as dt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import db
import portfolio


def main():
    db.init_db()
    tx = db.get_transactions_df()
    inst = db.get_instruments_df()
    _, totals = portfolio.build_portfolio(tx, inst)
    if not totals:
        print("No positions; nothing to record.")
        return
    row = {
        "date": dt.date.today(),
        "market_value": totals["market_value"],
        "book_value": totals["book_value"],
        "gain_loss": totals["gain_loss"],
        "gain_loss_pct": totals["gain_loss_pct"],
        "daily_pnl": totals["daily_pnl"],
        "daily_pnl_pct": totals["daily_pnl_pct"],
        "benchmark_pct": portfolio.benchmark_daily_pct(),
    }
    db.upsert_snapshot(row)
    print(f"Recorded {row['date']}: MV={row['market_value']:.2f} "
          f"P&L day={row['daily_pnl']:.2f}")


if __name__ == "__main__":
    main()
