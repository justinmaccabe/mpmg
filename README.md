# ◆ Maccabe Portfolio Management Group

A Streamlit portfolio dashboard that replaces the spreadsheet:

- **Holdings & gain/loss** — live prices via `yfinance`, valued in CAD (USD holdings converted at the current rate).
- **Add trades** — a form for Buy / Sell / Dividend; share counts and average cost (ACB) recompute automatically.
- **Correlations** — a heatmap of daily-return correlations across your ETFs.
- **Daily performance** — a snapshot row recorded **automatically every weekday** by a GitHub Action, plotted over time.

Seeded with your current holdings (ZMMK, XEQT, XUS, AVGE, OPO).

---

## Run it locally (no signup, 1 minute)

```bash
cd stonks-tracker
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`. Locally it uses a SQLite file (`stonks.db`) — fine for trying it out. To add today's performance point by hand: `python jobs/snapshot.py`.

---

## Deploy so you can open it from anywhere

The deployed app needs a **persistent database** because Streamlit Cloud wipes its
local disk on every restart (a SQLite file would lose your data). Use a free Postgres.

**1. Push to GitHub**
```bash
cd stonks-tracker
git init && git add . && git commit -m "Stonks tracker"
gh repo create stonks-tracker --private --source=. --push   # or push manually
```

**2. Create a free Postgres** at [neon.tech](https://neon.tech) (or supabase.com).
Copy the connection string — looks like `postgresql://user:pass@host/db`.

**3. Deploy the UI** at [share.streamlit.io](https://share.streamlit.io):
- New app → pick your repo → main file `app.py`.
- **Settings → Secrets**, paste:
  ```toml
  DATABASE_URL = "postgresql://user:pass@host/db"
  ```
- **Settings → Sharing** → restrict viewers to your Google account (it's your money 🙂).

You now have a private URL that works on your phone and laptop.

**4. Turn on the automatic daily snapshot**
- In the GitHub repo: **Settings → Secrets and variables → Actions → New secret**
  → name `DATABASE_URL`, value = the same connection string.
- The workflow in `.github/workflows/daily.yml` runs ~5 min after the TSX close
  every weekday and writes one row. Trigger it once now from the **Actions** tab
  (**Run workflow**) to confirm it works.

That's the part the spreadsheet couldn't do: it records even when nothing is open.

---

## Notes & assumptions

- **Tickers**: Canadian listings use the `.TO` suffix (`XEQT.TO`). `AVGE` is US-listed (USD). `OPO` is private — set its price in `db.py` (`manual_price`) or add a field later.
- **FX**: USD holdings are converted at *today's* USD/CAD. Cost basis uses the same rate (a simplification — not purchase-date FX).
- **Benchmark**: `XEQT.TO` by default — change `BENCHMARK_SYMBOL` in `db.py`.
- **Editing holdings**: the seed lives in `db.py` (`SEED_INSTRUMENTS`, `SEED_TRANSACTIONS`) and only loads into an empty database.
