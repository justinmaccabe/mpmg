"""Maccabe Portfolio Management Group — portfolio dashboard.

Run locally:   streamlit run app.py        (uses local SQLite, no setup)
Deployed:      set DATABASE_URL secret      (uses Postgres so data persists)
"""
import datetime as dt
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import db
import portfolio

st.set_page_config(page_title="Maccabe Portfolio Management Group",
                   page_icon="◆", layout="wide")
db.init_db()

# ----------------------------------------------------------------- palette
GOLD = "#C9A227"
BLUE = "#6E8CA0"      # muted steel — secondary accent
POS, NEG = "#16C784", "#E0533D"    # reserved for P&L semantics only
SERIF = "'Times New Roman', Georgia, serif"
# refined, desaturated categorical palette (gold-led, "private wealth")
PALETTE = ["#C9A227", "#6E8CA0", "#6FA287", "#B07B53", "#8E83A8", "#5F94A0", "#9CA3AB"]
SAGE = "#6FA287"      # muted green for gain/contribution bars (paired with NEG)

st.markdown(f"""
<style>
html, body, [class*="css"], .stApp, h1, h2, h3, h4, h5, p, div, span,
.stMarkdown, [data-testid="stMetricValue"], [data-testid="stMetricLabel"],
[data-testid="stMetricDelta"], button, input, select, textarea {{
    font-family: {SERIF} !important;
}}
/* tabular, lining figures so numbers align in columns — an institutional tell */
[data-testid="stMetricValue"], [data-testid="stMetricDelta"],
[data-testid="stDataFrame"], .stDataFrame, table, .mpmg-num {{
    font-variant-numeric: tabular-nums lining-nums !important;
    font-feature-settings: "tnum" 1, "lnum" 1 !important;
}}
.stApp {{ background: radial-gradient(1200px 600px at 50% -10%, #161B27 0%, #0B0E14 55%); }}
#MainMenu, footer {{ visibility: hidden; }}

/* monogram + header */
.mpmg-mark {{ display:inline-flex; align-items:center; justify-content:center;
    width:46px; height:46px; border:1.5px solid {GOLD}; border-radius:9px;
    color:{GOLD}; font-family:{SERIF}; font-style:italic; font-size:1.7rem;
    margin-right:.85rem; flex:0 0 auto; }}
.mpmg-header {{ display:flex; align-items:center; padding:.25rem 0 .15rem; }}
.mpmg-title {{ font-size:2.5rem; font-weight:700; letter-spacing:.04em;
    color:#F4F4F4; line-height:1.05; }}
.mpmg-title .amp {{ color:{GOLD}; }}
.mpmg-sub {{ color:{GOLD}; letter-spacing:.42em; text-transform:uppercase;
    font-size:.72rem; margin-top:.45rem; }}
.mpmg-asof {{ color:#8A8F9A; font-size:.9rem; margin-top:.2rem; font-style:italic; }}
.mpmg-rule {{ border:none; border-top:1px solid rgba(201,162,39,.35); margin:.35rem 0 1.1rem; }}

[data-testid="stToggle"] label p, [data-testid="stCheckbox"] label p {{
    color:{GOLD} !important; font-family:{SERIF} !important;
    text-transform:uppercase; letter-spacing:.16em; font-size:.72rem !important; }}

/* metric cards */
[data-testid="stMetric"] {{ background:rgba(255,255,255,.025);
    border:1px solid rgba(201,162,39,.18); border-radius:10px; padding:14px 18px; }}
[data-testid="stMetricLabel"] {{ color:#9AA0AB !important; text-transform:uppercase;
    letter-spacing:.12em; font-size:.72rem !important; min-height:2.1em; }}
[data-testid="stMetricLabel"] p {{ white-space:normal !important; overflow:visible !important;
    text-overflow:unset !important; line-height:1.3 !important; }}
[data-testid="stMetricValue"] {{ font-size:1.7rem !important; color:#F4F4F4; }}

h2, h3 {{ color:#EDEDED; border-left:3px solid {GOLD}; padding-left:.55rem; }}
.stTabs [data-baseweb="tab"] {{ font-size:1rem; letter-spacing:.03em; }}
.stTabs [aria-selected="true"] {{ color:{GOLD} !important; }}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------- data
@st.cache_data(ttl=900)
def load_portfolio():
    return portfolio.build_portfolio(db.get_transactions_df(), db.get_instruments_df())


def holdings_sig():
    """A signature of current holdings, used as a cache key so holdings-dependent
    views recompute when positions change (instead of waiting out a long TTL)."""
    pos, _ = load_portfolio()
    if pos.empty:
        return ()
    return tuple(sorted((str(r["Ticker"]), round(float(r["Shares"]), 4))
                        for _, r in pos.iterrows()))


@st.cache_data(ttl=3600)
def load_corr(sig):
    pos, _ = load_portfolio()
    held = set(pos[pos["Shares"] > 0]["Ticker"]) if not pos.empty else set()
    inst = db.get_instruments_df()
    return portfolio.correlation_matrix(inst[inst["ticker"].isin(held)])


@st.cache_data(ttl=3600)
def perf_all(period):
    inst = db.get_instruments_df()
    holds = inst[(~inst["is_private"]) & inst["yf_symbol"].notna()]
    syms = list(holds["yf_symbol"]) + list(portfolio.BENCHMARKS.values())
    return portfolio.normalized_performance(syms, period)


@st.cache_data(ttl=3600)
def perf_portfolio(period):
    return portfolio.portfolio_performance(db.get_transactions_df(),
                                           db.get_instruments_df(), period)


@st.cache_data(ttl=86400)
def load_factors(sig):
    pos, _ = load_portfolio()
    return portfolio.factor_exposure(pos, db.get_instruments_df())


@st.cache_data(ttl=86400)
def load_yield(sig):
    pos, _ = load_portfolio()
    return portfolio.portfolio_dividend_yield(pos, db.get_instruments_df())


@st.cache_data(ttl=86400)
def load_sharpe(sig):
    return portfolio.sharpe_ratios(db.get_transactions_df(), db.get_instruments_df())


@st.cache_data(ttl=86400)
def load_optim(period):
    return portfolio.optimize_blocks(period)


@st.cache_data(ttl=3600)
def load_period_returns(sig):
    return portfolio.period_returns(db.get_transactions_df(), db.get_instruments_df())


@st.cache_data(ttl=3600)
def load_attribution(sig, period):
    return portfolio.return_attribution(db.get_transactions_df(),
                                        db.get_instruments_df(), period)


HIDE = False          # set in the main flow; when True, dollar amounts are masked
GUEST = False
MASK = "$ •••••"


def money(x):
    return MASK if HIDE else f"${x:,.2f}"


def fmt_money0(v):
    if pd.isna(v):
        return "—"
    return MASK if HIDE else f"${v:,.0f}"


def fmt_signed(v):
    if pd.isna(v):
        return "—"
    return MASK if HIDE else f"{'+' if v >= 0 else '−'}${abs(v):,.0f}"


def logo_html(box, m, w):
    """The M·WEALTH mark: gold-gradient serif M over white WEALTH in a gold square.
    Font is inherited from the global serif rule (avoids quoting issues)."""
    rad = round(box * 0.2)
    return (
        f"<div style='width:{box}px; height:{box}px; border:2px solid {GOLD};"
        f" border-radius:{rad}px; display:inline-flex; flex-direction:column;"
        " align-items:center; justify-content:center; line-height:1; flex:0 0 auto;'>"
        f"<span style='font-weight:600; font-size:{m}rem;"
        " background:linear-gradient(175deg,#EAD888 0%,#C9A227 50%,#A87E1C 100%);"
        " -webkit-background-clip:text; background-clip:text; color:transparent;'>M</span>"
        f"<span style='color:#FFFFFF; letter-spacing:.4em; font-size:{w}rem;"
        " text-indent:.4em; margin-top:.12em;'>WEALTH</span></div>")


def style_fig(fig, height=360, legend=True):
    fig.update_layout(
        height=height, template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=SERIF, color="#D8D8D8", size=14),
        colorway=PALETTE, margin=dict(l=10, r=10, t=20, b=10),
        showlegend=legend, legend=dict(orientation="h", y=-0.18),
        hoverlabel=dict(bgcolor="#11161F", bordercolor="rgba(201,162,39,.45)",
                        font=dict(family=SERIF, color="#EAEAEA", size=13)),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,.06)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,.06)", zeroline=False)
    return fig


def show(fig):
    """Render with our own theme (theme=None) and no modebar clutter."""
    st.plotly_chart(fig, use_container_width=True, theme=None,
                    config={"displayModeBar": False})


def flag(kind, text):
    """Render a themed status flag (kind: 'ok' | 'warn' | 'bad')."""
    c = {"ok": POS, "warn": GOLD, "bad": NEG}.get(kind, GOLD)
    st.markdown(
        f"<div style=\"border-left:3px solid {c}; background:{c}14; color:#E8E8E8;"
        f" font-family:{SERIF}; padding:.65rem 1rem; border-radius:6px;"
        f" margin:.3rem 0; line-height:1.5;\">{text}</div>",
        unsafe_allow_html=True)


def color_pnl(v):
    if isinstance(v, (int, float)):
        if v > 0:
            return f"color:{POS}"
        if v < 0:
            return f"color:{NEG}"
    return "color:#9AA0AB"


def kpi_ribbon():
    """Persistent summary band shown above every tab."""
    pos, totals = load_portfolio()
    if not totals:
        return

    def amt(x):
        return "•••••" if HIDE else f"${x:,.0f}"

    gl, glp = totals["gain_loss"], totals["gain_loss_pct"]
    dp, dpp = totals["daily_pnl"], totals["daily_pnl_pct"]
    items = [
        ("Total Value", amt(totals["market_value"]), "", "#F4F4F4"),
        ("Today", amt(dp), f"{dpp:+.2%}", POS if dp >= 0 else NEG),
        ("Total Gain / Loss", amt(gl), f"{glp:+.2%}", POS if gl >= 0 else NEG),
        ("Book Value", amt(totals["book_value"]), "", "#F4F4F4"),
    ]
    cells = ""
    for i, (label, value, sub, color) in enumerate(items):
        border = "" if i == 0 else "border-left:1px solid rgba(201,162,39,.18);"
        subhtml = (f"<span style='font-size:.85rem; color:{color};'> {sub}</span>"
                   if sub else "")
        cells += (
            f"<div style='flex:1; padding:.35rem 1.3rem; {border}'>"
            f"<div style='color:#9AA0AB; text-transform:uppercase; letter-spacing:.14em;"
            f" font-size:.62rem;'>{label}</div>"
            f"<div class='mpmg-num' style='font-family:{SERIF}; font-size:1.55rem;"
            f" color:{color};'>{value}{subhtml}</div></div>")
    st.markdown(
        f"<div style='display:flex; background:rgba(255,255,255,.022);"
        f" border:1px solid rgba(201,162,39,.18); border-radius:11px;"
        f" margin-bottom:.6rem;'>{cells}</div>", unsafe_allow_html=True)


def period_ribbon():
    """Week-to-date and month-to-date returns (portfolio vs S&P 500). Percentages,
    so always shown — not masked by Hide Balances."""
    try:
        pr = load_period_returns(holdings_sig())
    except Exception:
        return                       # e.g. stale module pre-reboot — skip silently
    if not pr:
        return

    def pct(x):
        return "—" if x is None else f"{x:+.2%}"

    def tone(x):
        return "#9AA0AB" if x is None else (POS if x >= 0 else NEG)

    def cell(label, port, bench, first):
        border = "" if first else "border-left:1px solid rgba(201,162,39,.18);"
        return (
            f"<div style='flex:1; padding:.3rem 1.3rem; {border}'>"
            f"<div style='color:#9AA0AB; text-transform:uppercase; letter-spacing:.14em;"
            f" font-size:.6rem;'>{label}</div>"
            f"<div class='mpmg-num' style='font-size:1.05rem;'>"
            f"<span style='color:#9AA0AB; font-size:.8rem;'>Portfolio</span> "
            f"<span style='color:{tone(port)};'>{pct(port)}</span>"
            f"<span style='color:#5A5F6A;'> &nbsp;·&nbsp; </span>"
            f"<span style='color:#9AA0AB; font-size:.8rem;'>S&amp;P 500</span> "
            f"<span style='color:{tone(bench)};'>{pct(bench)}</span></div></div>")

    cells = (cell("Week to Date", pr["wtd_port"], pr["wtd_bench"], True)
             + cell("Month to Date", pr["mtd_port"], pr["mtd_bench"], False))
    st.markdown(
        f"<div style='display:flex; background:rgba(255,255,255,.015);"
        f" border:1px solid rgba(201,162,39,.12); border-radius:11px;"
        f" margin-bottom:1.1rem;'>{cells}</div>", unsafe_allow_html=True)


def opo_pending_month(contribs_df, today):
    """Earliest month (from Jan 2025) whose OPO buy is due (≥15th) but unrecorded."""
    recorded = set()
    if contribs_df is not None and not contribs_df.empty:
        opo = contribs_df[contribs_df["note"].astype(str).str.contains("OPO")]
        for d in pd.to_datetime(opo["date"]):
            recorded.add((d.year, d.month))
    months, y, m = [], 2025, 1
    while (y, m) < (today.year, today.month):
        months.append((y, m))
        m += 1
        if m > 12:
            m, y = 1, y + 1
    if today.day >= 15:
        months.append((today.year, today.month))
    for ym in months:
        if ym not in recorded:
            return ym
    return None


@st.dialog("Record your OPO monthly buy")
def opo_buy_dialog(ym):
    label = dt.date(ym[0], ym[1], 1).strftime("%B %Y")
    st.write(f"Your **OPO** contribution for **{label}** invested around the 15th. "
             "Enter the actual fill so your units, ACB, and contribution room stay accurate.")
    try:
        default_price = float(db.get_instruments_df().set_index("ticker")
                              .loc["OPO", "manual_price"] or 0)
    except Exception:
        default_price = 0.0
    qty = st.number_input("Units bought", min_value=0.0, step=0.0001, format="%.4f")
    price = st.number_input("Price per unit", min_value=0.0, value=default_price, format="%.4f")
    amt = st.number_input("Cash contributed ($)", min_value=0.0, value=500.0,
                          step=50.0, format="%.2f")
    c1, c2 = st.columns(2)
    if c1.button("Record buy", type="primary"):
        d = dt.date(ym[0], ym[1], 15)
        db.add_transaction(d, "OPO", "TFSA", "Buy", qty, price, 0.0)
        db.add_contribution(d, "TFSA", amt, "OPO (Optimize TFSA) $500/mo")
        if price > 0:
            db.set_manual_price("OPO", price)
        st.cache_data.clear()
        st.rerun()
    if c2.button("Remind me later"):
        st.session_state["opo_skip"] = True
        st.rerun()


def _secret(name, default=""):
    try:
        v = st.secrets.get(name, "")
    except Exception:
        v = ""
    return str(v or os.environ.get(name, "") or default)


def require_passcode():
    """Gate the dashboard. The main passcode (APP_PASSCODE) gives full access;
    the guest passcode opens the full app locked in hidden-balances mode."""
    code = _secret("APP_PASSCODE")
    guest_code = _secret("APP_GUEST_PASSCODE", "guest")
    if not code or st.session_state.get("authed"):
        return
    st.markdown(
        "<div style='text-align:center; margin-top:6vh;'>"
        + logo_html(124, 3.5, 0.72) +
        f"<div style='font-family:{SERIF}; font-size:2rem; color:#F4F4F4; margin-top:1rem;'>"
        f"Maccabe Portfolio Management <span style='color:{GOLD};'>Group</span></div>"
        f"<div style='color:{GOLD}; letter-spacing:.4em; text-transform:uppercase;"
        " font-size:.7rem; margin-top:.5rem;'>Private Wealth &nbsp;·&nbsp; Quantitative Strategy</div>"
        "</div>", unsafe_allow_html=True)
    c = st.columns([1, 1, 1])
    with c[1]:
        entered = st.text_input("Passcode", type="password",
                                label_visibility="collapsed", placeholder="Enter passcode")
        if entered:
            if entered == code:
                st.session_state["authed"], st.session_state["guest"] = True, False
                st.rerun()
            elif entered == guest_code:
                st.session_state["authed"], st.session_state["guest"] = True, True
                st.rerun()
            else:
                st.error("Incorrect passcode.")
    st.stop()


# ============================================================ tab renderers
def render_overview():
    pos, totals = load_portfolio()
    if not pos.empty:
        left, right = st.columns(2)
        with left:
            st.subheader("Allocation by Holding")
            fig = go.Figure(go.Pie(
                labels=pos["Ticker"], values=pos["Market Value"], hole=.6,
                marker=dict(colors=PALETTE), textinfo="label+percent",
                hovertemplate="%{label}: %{percent}<extra></extra>"))
            fig.update_layout(annotations=[dict(
                text="" if HIDE else money(totals["market_value"]), x=.5, y=.5,
                font=dict(family=SERIF, size=16, color="#F4F4F4"), showarrow=False)])
            show(style_fig(fig, 360, legend=False))
        with right:
            metric = "Gain/Loss %" if HIDE else "Gain/Loss $"
            st.subheader("Gain / Loss by Holding" + (" (%)" if HIDE else ""))
            g = pos.sort_values(metric)
            htmpl = ("%{y}: %{x:.2%}<extra></extra>" if HIDE
                     else "%{y}: $%{x:,.2f}<extra></extra>")
            fig = go.Figure(go.Bar(
                x=g[metric], y=g["Ticker"], orientation="h",
                marker_color=[NEG if v < 0 else SAGE for v in g[metric]],
                hovertemplate=htmpl))
            fig = style_fig(fig, 360, legend=False)
            fig.update_xaxes(tickformat=".0%" if HIDE else "$,.0f")
            show(fig)

        st.subheader("Holdings")
        cols = ["Ticker", "Account", "Shares", "ACB", "Price", "Cur",
                "Market Value", "Book Value", "Gain/Loss $", "Gain/Loss %",
                "Daily P&L $"]
        styler = (pos[cols].style
                  .format({"Shares": "{:,.2f}", "ACB": "${:,.2f}", "Price": "${:,.2f}",
                           "Market Value": fmt_money0, "Book Value": fmt_money0,
                           "Gain/Loss $": fmt_money0, "Gain/Loss %": "{:.2%}",
                           "Daily P&L $": fmt_money0})
                  .map(color_pnl, subset=["Gain/Loss $", "Gain/Loss %", "Daily P&L $"]))
        st.dataframe(styler, width="stretch", hide_index=True)

    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Performance History")
        snaps = db.get_snapshots_df()
        if len(snaps):
            snaps = snaps.assign(date=pd.to_datetime(snaps["date"]))
            freq = st.segmented_control(
                "Frequency", ["Daily", "Weekly", "Monthly"], default="Daily",
                label_visibility="collapsed", key="perf_freq") or "Daily"
            rule = {"Weekly": "W", "Monthly": "ME"}.get(freq)
            s = snaps.set_index("date")
            if rule:
                s = s.resample(rule).last().dropna(how="all")
            s = s.reset_index()
            # evenly-spaced category labels — one slot per snapshot, so weekends
            # never create gaps and every point (incl. today) is labelled
            if rule is None:                                   # daily
                s["_lbl"] = s["date"].dt.strftime("%b %d, %Y")
            elif rule == "W":                                  # weekly → Monday
                monday = s["date"] - pd.to_timedelta(s["date"].dt.weekday, unit="D")
                s["_lbl"] = ("<span style='font-size:0.72em'>Week of</span><br>"
                             + monday.dt.strftime("%b %d, %Y"))
            else:                                              # monthly → month name
                s["_lbl"] = s["date"].dt.strftime("%B")
            fig = go.Figure()
            if rule is None and "market_value_open" in s.columns \
                    and s["market_value_open"].notna().any():
                fig.add_trace(go.Scatter(x=s["_lbl"], y=s["market_value_open"],
                              mode="markers", name="Open",
                              marker=dict(size=9, color="#8FB3D9", symbol="circle-open")))
            fig.add_trace(go.Scatter(x=s["_lbl"], y=s["market_value"],
                          mode="lines+markers", name="Close", connectgaps=True,
                          line=dict(color=BLUE, width=2), marker=dict(size=8)))
            if s["book_value"].notna().any():
                fig.add_trace(go.Scatter(x=s["_lbl"], y=s["book_value"],
                              mode="lines+markers", name="Book Value", connectgaps=True,
                              line=dict(color=GOLD, width=1.5, dash="dot"),
                              marker=dict(size=7)))
            fig = style_fig(fig, 340)
            fig.update_yaxes(title="CAD", tickformat="$,.0f")
            fig.update_xaxes(type="category", categoryorder="array",
                             categoryarray=s["_lbl"].tolist())
            ycols = [s["market_value"], s["book_value"]]
            if "market_value_open" in s.columns:
                ycols.append(s["market_value_open"])
            yvals = pd.concat(ycols).dropna()
            if len(yvals):
                lo, hi = float(yvals.min()), float(yvals.max())
                pad = (hi - lo) * 0.25 if hi > lo else max(hi * 0.02, 1.0)
                fig.update_yaxes(range=[lo - pad, hi + pad])
            if HIDE:
                fig.update_yaxes(showticklabels=False, title=None)
                fig.update_traces(hoverinfo="skip", hovertemplate=None)
            show(fig)
            ss = snaps.sort_values("date")
            o1, o2 = st.columns(2)
            for col, field, lbl in ((o1, "market_value_open", "Latest open"),
                                     (o2, "market_value", "Latest close")):
                rows_v = ss[ss[field].notna()]
                if len(rows_v):
                    r = rows_v.iloc[-1]
                    col.metric(f"{lbl} · {pd.to_datetime(r['date']):%b %d, %Y}",
                               fmt_money0(r[field]))
                else:
                    col.metric(lbl, "—")
            st.caption("Open and close values are recorded each trading day; weekly and "
                       "monthly views reflect period-end closes.")
        else:
            st.info("Performance history will populate once the daily snapshot job has run.")
    with c2:
        st.subheader("Currency Exposure")
        if not pos.empty:
            cur = portfolio.currency_exposure(pos)
            cmap = {"CAD": GOLD, "USD": BLUE}
            fig = go.Figure(go.Pie(
                labels=cur.index, values=cur.values, hole=.6,
                marker=dict(colors=[cmap.get(c, "#8A8F9A") for c in cur.index]),
                textinfo="label+percent",
                hovertemplate="%{label}: %{percent}<extra></extra>"))
            fig.update_layout(annotations=[dict(
                text=f"{cur.get('USD', 0.0) / cur.sum():.0%} USD", x=.5, y=.5,
                font=dict(family=SERIF, size=15, color="#F4F4F4"), showarrow=False)])
            show(style_fig(fig, 360, legend=False))
            st.caption("Classified by trading currency. XUS is treated as USD exposure: "
                       "it holds the unhedged S&P 500 despite trading in CAD.")

    # ---- Daily open / close breakdown --------------------------------------
    _sn = db.get_snapshots_df()
    if len(_sn) and "market_value_open" in _sn.columns:
        d = _sn.assign(date=pd.to_datetime(_sn["date"])).sort_values("date")
        d["prev_close"] = d["market_value"].shift(1)
        tbl = pd.DataFrame({
            "Date": d["date"].dt.strftime("%b %d, %Y"),
            "Open": d["market_value_open"],
            "Intraday": d["market_value"] - d["market_value_open"],
            "Close": d["market_value"],
            "Overnight": d["market_value_open"] - d["prev_close"],
            "24h Return": d["market_value"] / d["prev_close"] - 1,
        }).iloc[::-1].reset_index(drop=True)
        st.subheader("Daily Open & Close")
        st.dataframe(
            tbl.style.format({
                "Open": fmt_money0, "Close": fmt_money0,
                "Intraday": fmt_signed, "Overnight": fmt_signed,
                "24h Return": lambda v: "—" if pd.isna(v) else f"{v:+.2%}"},
                na_rep="—")
            .map(color_pnl, subset=["Intraday", "Overnight", "24h Return"]),
            width="stretch", hide_index=True)
        st.caption("Intraday = close − open · Overnight = open − prior close · "
                   "24h Return = close vs prior close (overnight + intraday).")

    # ---- Return attribution -------------------------------------------------
    st.subheader("Return Attribution")
    ap = st.segmented_control(
        "Period", ["WTD", "MTD", "3M", "1Y", "3Y"], default="MTD",
        label_visibility="collapsed", key="attrib_period") or "MTD"
    attr = load_attribution(holdings_sig(), ap)
    if not attr["rows"]:
        st.info("Attribution will appear once price history is available for the holdings.")
    else:
        rows = attr["rows"]
        contrib = [r["contribution"] for r in rows]
        fig = go.Figure(go.Bar(
            x=contrib, y=[r["ticker"] for r in rows], orientation="h",
            marker_color=[SAGE if c >= 0 else NEG for c in contrib],
            text=[f"{c:+.2%}" for c in contrib], textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x:.2%} contribution<extra></extra>"))
        fig = style_fig(fig, 340, legend=False)
        fig.update_xaxes(title="Contribution to portfolio return", tickformat=".1%")
        fig.update_yaxes(autorange="reversed")          # largest contributor on top
        show(fig)
        st.caption(
            f"Portfolio {ap} return of {attr['port_return']:+.2%}, decomposed into each "
            "holding's contribution (beginning weight × period return, in CAD). "
            "Contributions sum to the total. Based on current holdings and current FX; "
            "OPO is excluded (no market data).")


def render_accounts():
    pos, totals = load_portfolio()
    if not totals:
        st.info("No positions yet.")
        return
    amv = totals["account_mv"]
    tot = amv["FHSA"] + amv["TFSA"]
    c1, c2, c3 = st.columns(3)
    c1.metric("FHSA Balance", money(amv["FHSA"]),
              f"{amv['FHSA'] / tot:.1%} of total" if tot else None, delta_color="off")
    c2.metric("TFSA Balance", money(amv["TFSA"]),
              f"{amv['TFSA'] / tot:.1%} of total" if tot else None, delta_color="off")
    c3.metric("Combined", money(tot))

    left, right = st.columns(2)
    with left:
        st.subheader("Balance by Account")
        fig = go.Figure(go.Pie(
            labels=["FHSA", "TFSA"], values=[amv["FHSA"], amv["TFSA"]],
            hole=.6, marker=dict(colors=[GOLD, BLUE]), textinfo="label+percent",
            hovertemplate="%{label}: %{percent}<extra></extra>"))
        show(style_fig(fig, 360, legend=False))
    with right:
        st.subheader("Holdings by Account")
        fig = go.Figure()
        fig.add_trace(go.Bar(name="FHSA", x=pos["Ticker"], y=pos["MV FHSA"],
                             marker_color=GOLD))
        fig.add_trace(go.Bar(name="TFSA", x=pos["Ticker"], y=pos["MV TFSA"],
                             marker_color=BLUE))
        fig = style_fig(fig, 360)
        fig.update_layout(barmode="stack")
        fig.update_yaxes(title="CAD", tickformat="$,.0f")
        if HIDE:
            fig.update_yaxes(showticklabels=False, title=None)
            fig.update_traces(hoverinfo="skip", hovertemplate=None)
        show(fig)

    st.subheader("Per-Account Detail")
    detail = pos[["Ticker", "MV FHSA", "MV TFSA"]].copy()
    detail["Total"] = detail["MV FHSA"] + detail["MV TFSA"]
    total_row = pd.DataFrame([{
        "Ticker": "Total", "MV FHSA": detail["MV FHSA"].sum(),
        "MV TFSA": detail["MV TFSA"].sum(), "Total": detail["Total"].sum()}])
    detail = pd.concat([detail, total_row], ignore_index=True)
    money_cols = ["MV FHSA", "MV TFSA", "Total"]
    if HIDE:
        grand = detail["Total"].iloc[-1] or 1
        pct = detail.copy()
        pct[money_cols] = pct[money_cols] / grand
        st.dataframe(pct.style.format({c: "{:.1%}" for c in money_cols}),
                     width="stretch", hide_index=True)
    else:
        st.dataframe(detail.style.format({c: "${:,.0f}" for c in money_cols}),
                     width="stretch", hide_index=True)


def render_contributions():
    st.subheader("Contribution Room")
    cdf = db.get_contributions_df()
    yr = dt.date.today().year
    if not cdf.empty:
        cdf = cdf.assign(year=pd.to_datetime(cdf["date"]).dt.year)

    def by_year(acct):
        if cdf.empty:
            return {}
        return cdf[cdf["account"] == acct].groupby("year")["amount"].sum().to_dict()

    tfsa_by, fhsa_by = by_year("TFSA"), by_year("FHSA")
    tfsa_room = portfolio.tfsa_cumulative_room(yr)
    tfsa_used = sum(tfsa_by.values())
    tfsa_ytd = tfsa_by.get(yr, 0.0)
    fh = portfolio.fhsa_status(fhsa_by, yr)
    fhsa_ytd = fhsa_by.get(yr, 0.0)

    a, b = st.columns(2)
    with a:
        st.markdown("#### TFSA")
        st.metric("Room remaining (all-time)", money(tfsa_room - tfsa_used),
                  None if HIDE else f"{tfsa_used / tfsa_room:.0%} of ${tfsa_room:,.0f} used",
                  delta_color="off")
        if not HIDE:
            st.progress(min(1.0, max(0.0, tfsa_used / tfsa_room)))
        st.caption(f"Contributed {money(tfsa_used)} all-time · {money(tfsa_ytd)} in "
                   f"{yr}. Cumulative room since age 18 ({db.USER_BIRTH_YEAR + 18}): "
                   f"${tfsa_room:,.0f}. All TFSAs (RBC + Optimize) share this limit.")
        if tfsa_room - tfsa_used < 0:
            st.error("Over-contributed — CRA charges 1%/month on the excess.")
    with b:
        st.markdown("#### FHSA")
        st.metric("Room remaining this year", money(fh["available_this_year"]),
                  "Maxed for the year" if fh["available_this_year"] <= 0 else None,
                  delta_color="off")
        if not HIDE:
            st.progress(min(1.0, max(0.0, fh["used_lifetime"] / db.FHSA_LIFETIME_LIMIT)))
        st.caption(f"Contributed {money(fhsa_ytd)} in {yr}. Lifetime "
                   f"{money(fh['used_lifetime'])} of ${db.FHSA_LIFETIME_LIMIT:,.0f} "
                   f"used · {money(fh['lifetime_remaining'])} remaining.")

    if not cdf.empty:
        st.subheader("Contributions by Year")
        piv = cdf.pivot_table(index="year", columns="account", values="amount",
                              aggfunc="sum", fill_value=0)
        piv["Total"] = piv.sum(axis=1)
        st.dataframe(piv.style.format(fmt_money0), width="stretch")

    if GUEST:
        st.caption("Guest view — adding and removing contributions is disabled.")
        return
    st.subheader("Log a Contribution")
    with st.form("contrib", clear_on_submit=True):
        cc = st.columns([1, 1, 1, 2])
        cdate = cc[0].date_input("Date", dt.date.today(), key="cdate")
        cacct = cc[1].selectbox("Account", ["TFSA", "FHSA"], key="cacct")
        camt = cc[2].number_input("Amount", min_value=0.0, step=100.0, format="%.2f")
        cnote = cc[3].text_input("Note", "")
        if st.form_submit_button("Add contribution"):
            db.add_contribution(cdate, cacct, camt, cnote)
            st.success(f"Added {cacct} contribution.")
            st.rerun()

    if not cdf.empty:
        st.subheader("Remove a Contribution")
        clabels = {
            f"#{int(r.id)} · {r.date} · {r.account} · {money(r.amount)} · {r.note}":
            int(r.id) for r in cdf.itertuples()
        }
        cpick = st.selectbox("Select a contribution", list(clabels))
        if st.button("Delete selected contribution", type="primary"):
            db.delete_contribution(clabels[cpick])
            st.success("Deleted.")
            st.rerun()


def render_benchmarks():
    st.subheader("Performance vs Benchmarks")
    inst = db.get_instruments_df()
    holds = inst[(~inst["is_private"]) & inst["yf_symbol"].notna()]
    name_by_sym = dict(zip(holds["yf_symbol"], holds["ticker"]))

    ctrl = st.columns([2, 1, 1, 1])
    period = ctrl[0].selectbox("Period", ["3mo", "6mo", "1y", "2y", "5y"], index=2)
    show_pf = ctrl[1].checkbox("Portfolio", value=True)
    show_tm = ctrl[2].checkbox("Total Market", value=True)
    show_sp = ctrl[3].checkbox("S&P 500", value=True)
    chosen = st.multiselect("Holdings to plot", list(name_by_sym.values()),
                            default=list(name_by_sym.values()))

    perf = perf_all(period)
    if perf.empty:
        st.warning("Price history is currently unavailable. Please retry shortly.")
        return
    fig = go.Figure()
    for sym, name in name_by_sym.items():
        if name in chosen and sym in perf.columns:
            fig.add_trace(go.Scatter(x=perf.index, y=perf[sym], mode="lines", name=name))
    if show_pf:
        pp = perf_portfolio(period)
        if not pp.empty:
            fig.add_trace(go.Scatter(x=pp.index, y=pp.values, mode="lines",
                          name="Portfolio", line=dict(width=4, color=BLUE)))
    bench_color = {"Total Market": GOLD, "S&P 500": "#F4F4F4"}
    bench_on = {"Total Market": show_tm, "S&P 500": show_sp}
    for label, sym in portfolio.BENCHMARKS.items():
        if bench_on[label] and sym in perf.columns:
            fig.add_trace(go.Scatter(x=perf.index, y=perf[sym], mode="lines", name=label,
                          line=dict(width=3, dash="dash", color=bench_color[label])))
    fig = style_fig(fig, 480)
    fig.update_yaxes(title="Growth of $100")
    fig.update_xaxes(type="date", tickformat="%b %Y")
    show(fig)
    st.caption("Indexed to 100 at the period start; relative price performance only. "
               "Portfolio reflects current holdings backtested on price history "
               "(OPO excluded — no market data). Benchmarks: Total Market (VT) and "
               "S&P 500 (^GSPC).")


def render_factor():
    st.subheader("Factor Exposure")
    st.caption("Returns-based Fama-French five-factor plus momentum loadings, "
               "market-value-weighted across holdings.")
    try:
        fx = load_factors(holdings_sig())
    except Exception as e:
        st.warning(f"Factor data is currently unavailable: {e}")
        return
    if not fx:
        return
    port = fx["portfolio"]
    vals = [port[f] for f in fx["factors"]]
    fig = go.Figure(go.Bar(
        x=fx["factors"], y=vals,
        marker_color=[GOLD if v >= 0 else NEG for v in vals],
        text=[f"{v:+.2f}" for v in vals], textposition="outside"))
    fig = style_fig(fig, 420, legend=False)
    fig.update_yaxes(title="Loading (β)")
    show(fig)
    if fx["unattributed"] > 0.001:
        st.caption(f"Loadings cover the marketable sleeve; {fx['unattributed']:.0%} of the "
                   "portfolio (OPO, private) has no return history and is excluded.")
    rows = []
    for tk, d in fx["per_fund"].items():
        row = {"Ticker": tk, "Weight": d["weight"], "R²": d["r2"],
               "Months": d["n"], "α (mo)": d["alpha"]}
        row.update({f: d["betas"][f] for f in fx["factors"]})
        rows.append(row)
    fdf = pd.DataFrame(rows)
    st.dataframe(fdf.style.format(
        {"Weight": "{:.0%}", "R²": "{:.2f}", "α (mo)": "{:+.2%}",
         **{f: "{:+.2f}" for f in fx["factors"]}}),
        width="stretch", hide_index=True)
    st.caption(f"Factor window {fx['window'][0]}–{fx['window'][1]} (Developed factors, monthly). "
               "CAD-listed funds show market β below 1.0 partly owing to CAD/USD drift against "
               "the USD-denominated factor set; AVGE (USD) carries the value (HML) and "
               "profitability tilts of its Avantis design. Estimates for funds under two years "
               "old are less reliable. Updated daily.")


def render_correlations():
    st.subheader("Correlation of Daily Returns (~1Y)")
    corr = load_corr(holdings_sig())
    if corr.empty:
        st.warning("Price history is currently unavailable. Please retry shortly.")
        return
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu",
                    zmin=-1, zmax=1, aspect="auto")
    show(style_fig(fig, 440, legend=False))
    st.caption("1.0 = perfectly correlated · 0 = uncorrelated · negative = inversely "
               "correlated. OPO (private) is excluded — no market data.")


# Factor / exposure each underlying building block targets (display labels).
BLOCK_FACTOR = {
    "AVUS": "US Quality", "AVLV": "US Large Value", "AVUV": "US Small-Cap Value",
    "AVSC": "US Small-Cap", "AVMV": "US Mid-Cap Value", "AVRE": "US Real Estate",
    "AVDE": "Intl Developed", "AVIV": "Intl Developed Value",
    "AVEM": "Emerging Markets", "AVES": "EM Value",
    "US Market": "US Market", "Canada Market": "Canada Market",
    "Intl Dev Market": "Intl Developed", "EM Market": "Emerging Markets",
}

# Global market-cap weights (~MSCI ACWI) — the forecast-free "truth" anchor and
# Black-Litterman prior per IPS §6. The market portfolio expresses regional beta
# through the broad blocks, so dedicated factor sleeves sit at 0: their weight is
# a deliberate tilt, not part of the cap-weight truth. Approximate; editable.
MKT_CAP_TRUTH = {
    "US Market": 0.63, "Intl Dev Market": 0.24, "EM Market": 0.10, "Canada Market": 0.03,
}


def render_lookthrough():
    st.subheader("Portfolio Look-Through")
    pos, _ = load_portfolio()
    if pos.empty:
        st.info("No positions yet.")
        return
    lt = portfolio.look_through(pos, db.get_instruments_df())
    if not lt["blocks"]:
        st.info("No market holdings to decompose.")
        return
    reg, sty = lt["region"], lt["style"]
    tilt = 1 - sty.get("Market", 0.0)
    st.caption(
        "The equity sleeve (OPO excluded), seen through to its underlying funds: "
        f"**{reg.get('US', 0):.0%} US · {reg.get('Intl Dev', 0):.0%} Intl Developed · "
        f"{reg.get('EM', 0):.0%} EM · {reg.get('Canada', 0):.0%} Canada**. Only "
        f"**{tilt:.0%}** is factor-tilted (value/small) versus market-cap weighting, as the "
        "Avantis allocation (AVGE) is a small and predominantly US-market position.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### By Region")
        order = ["US", "Canada", "Intl Dev", "EM", "Other"]
        items = [(k, reg[k]) for k in order if k in reg]
        fig = go.Figure(go.Pie(labels=[k for k, _ in items], values=[v for _, v in items],
                        hole=.6, marker=dict(colors=PALETTE), textinfo="label+percent"))
        show(style_fig(fig, 340, legend=False))
    with c2:
        st.markdown("##### By Style")
        items = sorted(sty.items(), key=lambda x: x[1])
        fig = go.Figure(go.Bar(
            x=[v for _, v in items], y=[k for k, _ in items], orientation="h",
            marker_color=GOLD, text=[f"{v:.0%}" for _, v in items],
            textposition="outside", cliponaxis=False))
        fig = style_fig(fig, 340, legend=False)
        fig.update_xaxes(tickformat=".0%")
        show(fig)

    st.markdown("##### Underlying Building Blocks")
    bdf = pd.DataFrame([{"Building block": b,
                         "Factor": BLOCK_FACTOR.get(b, "—"),
                         "Weight": w,
                         "Mkt-cap truth": MKT_CAP_TRUTH.get(b, 0.0),
                         "Δ vs truth": w - MKT_CAP_TRUTH.get(b, 0.0)}
                        for b, w in sorted(lt["blocks"].items(), key=lambda x: -x[1])])
    st.dataframe(
        bdf.style.format({"Weight": "{:.1%}", "Mkt-cap truth": "{:.1%}",
                          "Δ vs truth": "{:+.1%}"}),
        width="stretch", hide_index=True)
    st.caption("AVGE is decomposed into its ten Avantis sleeves (renormalized to 100%); "
               "XEQT into iShares regional weights (approximate); XUS as US large-cap.")
    st.caption("**Mkt-cap truth** = each block's weight in the global market-cap portfolio "
               "(≈MSCI ACWI: US 63%, Intl Developed 24%, EM 10%, Canada 3%) — the "
               "forecast-free Black-Litterman prior per IPS §6. It carries regional beta in "
               "the broad blocks, so the Avantis factor sleeves sit at 0%: that weight is a "
               "deliberate tilt. **Δ vs truth** is your active bet (＋ overweight / − under).")

    # ---- Phase 2: optimized target & gaps -------------------------------
    st.divider()
    st.subheader("Optimized Target & Gaps (§6)")
    method = st.segmented_control(
        "Method", ["Min-Variance", "Risk Parity (inv-vol)", "Equal Weight"],
        default="Min-Variance", label_visibility="collapsed", key="opt_method") \
        or "Min-Variance"
    try:
        o = load_optim("5y")
    except Exception as e:
        st.warning(f"Optimizer is currently unavailable ({type(e).__name__}: {e}).")
        return
    if not o:
        st.warning("Optimizer data is currently unavailable. Please retry shortly.")
        return
    key = {"Min-Variance": "minvar", "Risk Parity (inv-vol)": "invvol",
           "Equal Weight": "equal"}[method]
    target = o[key]

    cur = portfolio.current_block_weights(pos, db.get_instruments_df())
    canada = cur.pop("Canada Market", 0.0)
    cur_s = pd.Series({a: cur.get(a, 0.0) for a in o["assets"]}, dtype=float)
    if cur_s.sum() > 0:
        cur_s = cur_s / cur_s.sum()
    comp = pd.DataFrame({"Current": cur_s, "Target": target}).fillna(0.0)
    comp["Gap"] = comp["Target"] - comp["Current"]
    comp = comp.sort_values("Target", ascending=False)

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Current", x=comp.index, y=comp["Current"], marker_color=BLUE))
    fig.add_trace(go.Bar(name="Target", x=comp.index, y=comp["Target"], marker_color=GOLD))
    fig = style_fig(fig, 380)
    fig.update_layout(barmode="group")
    fig.update_yaxes(tickformat=".0%")
    show(fig)
    st.dataframe(comp.style.format("{:.1%}").map(color_pnl, subset=["Gap"]),
                 width="stretch")

    under = comp[comp["Gap"] > 0.03].sort_values("Gap", ascending=False)
    over = comp[comp["Gap"] < -0.03].sort_values("Gap")
    add_txt = ", ".join(f"**{i}** (+{comp.loc[i, 'Gap']:.0%})" for i in under.index[:4])
    trim_txt = ", ".join(f"**{i}** ({comp.loc[i, 'Gap']:.0%})" for i in over.index[:2])
    if add_txt:
        flag("warn", f"{method} target versus current — **increase**: {add_txt}"
             + (f"; **reduce**: {trim_txt}" if trim_txt else "")
             + ". The underweights are value, small-cap, and international sleeves "
               "carrying negligible current exposure.")
    st.caption(
        f"Target weights over AVGE's ten Avantis sleeves; covariance estimated over "
        f"{o['cov_months']} months with shrinkage. Canada ({canada:.0%}) is excluded — no "
        "corresponding Avantis sleeve. Implementation: these tilts cannot be achieved through "
        "XEQT, XUS, or AVGE alone (all market-cap weighted); a material value, small-cap, or "
        "international tilt requires holding the Avantis sleeves directly — the §6 strategic "
        "question. Min-Variance concentrates in low-volatility, diversifying sleeves; Risk "
        "Parity weights by inverse volatility; Equal Weight is the unweighted reference.")


def render_leverage():
    st.subheader("Leverage")
    pos, totals = load_portfolio()
    if not totals:
        st.info("No positions yet.")
        return
    s = db.get_settings()
    if not GUEST:
        with st.form("loc"):
            lc = st.columns(3)
            loc = lc[0].number_input("LOC balance ($)", value=float(s["loc_balance"]),
                                     min_value=0.0, step=500.0)
            prime = lc[1].number_input("Prime rate (%)", value=float(s["prime_rate"]),
                                       min_value=0.0, step=0.05, format="%.2f")
            spread = lc[2].number_input("Spread over prime (%)", value=float(s["loc_spread"]),
                                        min_value=0.0, step=0.05, format="%.2f")
            if st.form_submit_button("Save"):
                db.set_settings({"loc_balance": loc, "prime_rate": prime,
                                 "loc_spread": spread})
                st.cache_data.clear()
                st.rerun()
    else:
        loc, prime, spread = s["loc_balance"], s["prime_rate"], s["loc_spread"]

    sig = holdings_sig()
    yld_pct = load_yield(sig) * 100.0
    sh = load_sharpe(sig)
    snaps = db.get_snapshots_df()
    peak = snaps["market_value"].max() if len(snaps) else totals["market_value"]
    lev = portfolio.leverage_metrics(totals["market_value"], loc, prime, spread, yld_pct, peak)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Leverage factor", f"{lev['leverage']:.2f}×")
    c2.metric("Equity", money(lev["equity"]))
    c3.metric("Gross exposure", money(lev["gross_exposure"]))
    c4.metric("LOC rate", f"{lev['loc_rate'] * 100:.2f}%")
    try:
        pp = perf_portfolio("3y")
        mdd = portfolio.max_drawdown(pp) if not pp.empty else None
    except Exception:
        mdd = None
    # actual experienced drawdown — worst peak-to-trough of the recorded
    # close-value snapshot history (vs the 3Y backtest of current holdings)
    exp_mdd = portfolio.max_drawdown(snaps["market_value"].dropna()) if len(snaps) else None
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Annual interest", money(lev["annual_interest"]))
    d2.metric("Monthly interest", money(lev["monthly_interest"]))
    d3.metric("Drawdown from peak", f"{lev['drawdown']:.1%}")
    d4.metric("Max drawdown (recorded)", f"{exp_mdd:.1%}" if exp_mdd is not None else "—")
    d5.metric("Max drawdown (3Y, modeled)", f"{mdd:.1%}" if mdd is not None else "—")
    e1, e2, e3 = st.columns(3)
    e1.metric("Est. portfolio yield", f"{yld_pct:.2f}%")
    p_sh, b_sh = sh.get("portfolio"), sh.get("benchmark")
    e2.metric("Portfolio Sharpe", f"{p_sh:.2f}" if p_sh is not None else "—")
    e3.metric(f"{sh.get('benchmark_symbol', 'Benchmark')} Sharpe",
              f"{b_sh:.2f}" if b_sh is not None else "—")

    st.markdown("##### IPS flags")
    if lev["drawdown_trigger"]:
        flag("bad", "Drawdown ≥ 50% from peak — IPS §8 mandatory strategy review.")
    else:
        flag("ok", f"Drawdown {lev['drawdown']:.1%} — within the IPS §8 50% review trigger.")
    cost_pct = lev["loc_rate"] * 100
    if not lev["yield_le_cost"]:
        flag("ok", f"Portfolio yield ({yld_pct:.2f}%) exceeds LOC cost ({cost_pct:.2f}%).")
    elif p_sh is not None and b_sh is not None and p_sh > b_sh:
        flag("warn", f"Yield ({yld_pct:.2f}%) ≤ LOC cost ({cost_pct:.2f}%) — IPS §7 flag — but "
             f"portfolio Sharpe ({p_sh:.2f}) still exceeds the market ({b_sh:.2f}), so the "
             "§7 Sharpe override holds: leverage remains justified.")
    else:
        flag("bad", f"Yield ({yld_pct:.2f}%) ≤ LOC cost ({cost_pct:.2f}%) and no Sharpe "
             "advantage — IPS §7/§8 suggest reassessing the leverage position.")
    st.caption("Yield = MV-weighted trailing-12-month distribution yield of the ETFs. Sharpe = "
               "annualized, monthly total returns over the FF risk-free (current holdings "
               "backtested; OPO excluded). Leverage = gross exposure ÷ equity. Prime is an "
               "input — keep it current per IPS §7.")


def render_trade():
    inst = db.get_instruments_df()
    if not GUEST:
        if st.button("➕ Record OPO monthly buy"):
            today = dt.date.today()
            ym = (opo_pending_month(db.get_contributions_df(), today)
                  or (today.year, today.month))
            st.session_state["opo_skip"] = False
            opo_buy_dialog(ym)

        priv = inst[inst["is_private"].astype(bool)]
        if not priv.empty:
            st.subheader("Private Holdings — Update Mark")
            st.caption("Privates aren't covered by the market feed. Enter the latest "
                       "valuation whenever your manager publishes one — it drives market "
                       "value and performance until you update it again.")
            pmap = priv.set_index("ticker")["manual_price"].to_dict()
            pc = st.columns([2, 2, 1])
            pt = pc[0].selectbox("Holding", list(pmap), key="priv_tkr")
            newp = pc[1].number_input(
                f"Current price / unit  ·  now ${float(pmap[pt] or 0):,.4f}",
                min_value=0.0, value=float(pmap[pt] or 0), step=0.0001,
                format="%.4f", key=f"priv_px_{pt}")
            pc[2].markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
            if pc[2].button("Update mark", type="primary"):
                db.set_manual_price(pt, newp)
                st.cache_data.clear()
                st.success(f"{pt} marked at ${newp:,.4f}.")
                st.rerun()

        st.subheader("Log a Transaction")
        with st.form("trade", clear_on_submit=True):
            col = st.columns(3)
            date = col[0].date_input("Date", dt.date.today())
            ticker = col[1].selectbox("Ticker", inst["ticker"].tolist())
            account = col[2].selectbox("Account", ["FHSA", "TFSA"])
            col2 = st.columns(4)
            action = col2[0].selectbox("Action", ["Buy", "Sell", "Dividend"])
            shares = col2[1].number_input("Shares", min_value=0.0, step=1.0, format="%.4f")
            price = col2[2].number_input("Price / share", min_value=0.0, step=0.01, format="%.4f")
            fees = col2[3].number_input("Fees", min_value=0.0, step=0.01, format="%.2f")
            if st.form_submit_button("Add transaction"):
                db.add_transaction(date, ticker, account, action, shares, price, fees)
                st.cache_data.clear()
                st.success(f"Added {action} {shares} {ticker} ({account}).")

    st.subheader("Ledger")
    ledger = db.get_transactions_df()
    st.dataframe(ledger, width="stretch", hide_index=True)

    if GUEST:
        st.caption("Guest view — adding and removing transactions is disabled.")
    elif not ledger.empty:
        st.subheader("Remove a Transaction")
        labels = {
            f"#{int(r.id)} · {r.date} · {r.action} {r.shares:g} {r.ticker} "
            f"({r.account}) @ ${r.price:g}": int(r.id)
            for r in ledger.itertuples()
        }
        pick = st.selectbox("Select a transaction", list(labels))
        if st.button("Delete selected transaction", type="primary"):
            db.delete_transaction(labels[pick])
            st.cache_data.clear()
            st.success(f"Deleted transaction {pick.split(' · ')[0]}.")
            st.rerun()


def render_ips():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ips.md")
    try:
        with open(path, encoding="utf-8") as f:
            st.markdown(f.read())
    except FileNotFoundError:
        st.info("Investment Policy Statement not found (ips.md missing).")


# ============================================================ main
require_passcode()

st.markdown(
    "<div class='mpmg-header'>" + logo_html(58, 1.85, 0.30) +
    "<div class='mpmg-title' style='margin-left:.85rem;'>Maccabe Portfolio Management "
    "<span class='amp'>Group</span></div></div>", unsafe_allow_html=True)

hc1, hc2 = st.columns([5, 2], vertical_alignment="center")
with hc1:
    st.markdown(
        "<div class='mpmg-sub'>Private Wealth &nbsp;·&nbsp; Quantitative Strategy</div>"
        f"<div class='mpmg-asof'><em>As of {dt.date.today():%B %d, %Y}</em></div>",
        unsafe_allow_html=True)
GUEST = st.session_state.get("guest", False)
with hc2:
    HIDE = st.toggle("Hide Balances", value=GUEST, disabled=GUEST,
                     help="Mask dollar amounts; percentages stay visible.") or GUEST
    if GUEST:
        st.caption("Guest view — balances locked")
st.markdown("<hr class='mpmg-rule'>", unsafe_allow_html=True)

kpi_ribbon()
period_ribbon()

# Prompt for this month's OPO buy once it's due (skippable for the session)
if not GUEST and not st.session_state.get("opo_skip"):
    _opo_due = opo_pending_month(db.get_contributions_df(), dt.date.today())
    if _opo_due:
        opo_buy_dialog(_opo_due)

t_over, t_acct, t_analytics, t_risk, t_manage, t_ips = st.tabs(
    ["Overview", "Accounts", "Analytics", "Risk", "Manage", "IPS"])

with t_over:
    render_overview()
with t_acct:
    ac1, ac2 = st.tabs(["Balances", "Contributions"])
    with ac1:
        render_accounts()
    with ac2:
        render_contributions()
with t_analytics:
    an1, an2, an3, an4 = st.tabs(
        ["Benchmarks", "Factor Exposure", "Correlations", "Policy (§6)"])
    with an1:
        render_benchmarks()
    with an2:
        render_factor()
    with an3:
        render_correlations()
    with an4:
        render_lookthrough()
with t_risk:
    render_leverage()
with t_manage:
    render_trade()
with t_ips:
    render_ips()
