-- Reference only. The app creates these automatically via db.init_db().

CREATE TABLE IF NOT EXISTS instruments (
    ticker       TEXT PRIMARY KEY,
    yf_symbol    TEXT,
    currency     TEXT NOT NULL DEFAULT 'CAD',
    is_private   BOOLEAN NOT NULL DEFAULT FALSE,
    manual_price DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS transactions (
    id      SERIAL PRIMARY KEY,
    date    DATE NOT NULL,
    ticker  TEXT NOT NULL,
    account TEXT NOT NULL,          -- FHSA / TFSA
    action  TEXT NOT NULL,          -- Buy / Sell / Dividend
    shares  DOUBLE PRECISION NOT NULL DEFAULT 0,
    price   DOUBLE PRECISION NOT NULL DEFAULT 0,
    fees    DOUBLE PRECISION NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS snapshots (
    date               DATE PRIMARY KEY,
    market_value       DOUBLE PRECISION,   -- day's close (latest)
    market_value_open  DOUBLE PRECISION,   -- mid-morning open snapshot
    book_value         DOUBLE PRECISION,
    gain_loss     DOUBLE PRECISION,
    gain_loss_pct DOUBLE PRECISION,
    daily_pnl     DOUBLE PRECISION,
    daily_pnl_pct DOUBLE PRECISION,
    benchmark_pct DOUBLE PRECISION
);
