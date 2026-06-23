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

# ----------------------------------------------------------------- styling
GOLD = "#C9A227"
SERIF = "'Times New Roman', Georgia, serif"
PALETTE = ["#C9A227", "#16C784", "#4C9AFF", "#E0533D", "#9B8AFF", "#52B7C9", "#B8B8B8"]

st.markdown(f"""
<style>
html, body, [class*="css"], .stApp, h1, h2, h3, h4, h5, p, div, span,
.stMarkdown, [data-testid="stMetricValue"], [data-testid="stMetricLabel"],
[data-testid="stMetricDelta"], button, input, select, textarea {{
    font-family: {SERIF} !important;
}}
.stApp {{ background: radial-gradient(1200px 600px at 50% -10%, #161B27 0%, #0B0E14 55%); }}
#MainMenu, footer {{ visibility: hidden; }}

/* branded header */
.mpmg-header {{ padding: .25rem 0 .15rem; }}
.mpmg-rule {{ border: none; border-top: 1px solid rgba(201,162,39,.35);
    margin: .35rem 0 1.3rem; }}
[data-testid="stToggle"] label p, [data-testid="stCheckbox"] label p {{
    color: {GOLD} !important; font-family: {SERIF} !important;
    text-transform: uppercase; letter-spacing: .16em; font-size: .72rem !important; }}
.mpmg-title {{ font-size: 2.55rem; font-weight: 700; letter-spacing: .04em;
    color: #F4F4F4; line-height: 1.05; }}
.mpmg-title .amp {{ color: {GOLD}; }}
.mpmg-sub {{ color: {GOLD}; letter-spacing: .42em; text-transform: uppercase;
    font-size: .72rem; margin-top: .45rem; }}
.mpmg-asof {{ color: #8A8F9A; font-size: .85rem; margin-top: .2rem; font-style: italic; }}

/* metric cards */
[data-testid="stMetric"] {{ background: rgba(255,255,255,.025);
    border: 1px solid rgba(201,162,39,.18); border-radius: 10px;
    padding: 14px 18px; }}
[data-testid="stMetricLabel"] {{ color: #9AA0AB !important; text-transform: uppercase;
    letter-spacing: .12em; font-size: .72rem !important; }}
[data-testid="stMetricValue"] {{ font-size: 1.7rem !important; color: #F4F4F4; }}

h2, h3 {{ color: #EDEDED; border-left: 3px solid {GOLD}; padding-left: .55rem; }}
.stTabs [data-baseweb="tab"] {{ font-size: 1rem; letter-spacing: .03em; }}
.stTabs [aria-selected="true"] {{ color: {GOLD} !important; }}
</style>

<div class="mpmg-header">
  <div class="mpmg-title">Maccabe Portfolio Management <span class="amp">Group</span></div>
</div>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------- data
@st.cache_data(ttl=900)
def load_portfolio():
    return portfolio.build_portfolio(db.get_transactions_df(), db.get_instruments_df())


@st.cache_data(ttl=3600)
def load_corr():
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
def load_factors():
    pos, _ = load_portfolio()
    return portfolio.factor_exposure(pos, db.get_instruments_df())


@st.cache_data(ttl=86400)
def load_yield():
    pos, _ = load_portfolio()
    return portfolio.portfolio_dividend_yield(pos, db.get_instruments_df())


@st.cache_data(ttl=86400)
def load_sharpe():
    return portfolio.sharpe_ratios(db.get_transactions_df(), db.get_instruments_df())


HIDE = False          # toggled below; when True, dollar amounts are masked
MASK = "$ •••••"


def money(x):
    return MASK if HIDE else f"${x:,.2f}"


def fmt_money0(v):
    return MASK if HIDE else f"${v:,.0f}"


def style_fig(fig, height=360, legend=True):
    fig.update_layout(
        height=height, template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=SERIF, color="#D8D8D8", size=14),
        colorway=PALETTE, margin=dict(l=10, r=10, t=20, b=10),
        showlegend=legend, legend=dict(orientation="h", y=-0.18),
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
    palette = {"ok": "#16C784", "warn": GOLD, "bad": "#E0533D"}
    c = palette.get(kind, GOLD)
    st.markdown(
        f"<div style=\"border-left:3px solid {c}; background:{c}14; color:#E8E8E8;"
        f" font-family:{SERIF}; padding:.65rem 1rem; border-radius:6px;"
        f" margin:.3rem 0; line-height:1.5;\">{text}</div>",
        unsafe_allow_html=True)


def color_pnl(v):
    if isinstance(v, (int, float)):
        if v > 0:
            return "color:#16C784"
        if v < 0:
            return "color:#E0533D"
    return "color:#9AA0AB"


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
            db.set_manual_price("OPO", price)   # keep the balance current too
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
    the guest passcode opens the full app locked in hidden-balances mode.
    If no main passcode is configured, the gate is disabled (local dev)."""
    code = _secret("APP_PASSCODE")
    guest_code = _secret("APP_GUEST_PASSCODE", "guest")
    if not code or st.session_state.get("authed"):
        return
    entered = st.text_input("Enter passcode to view", type="password")
    if entered:
        if entered == code:
            st.session_state["authed"], st.session_state["guest"] = True, False
            st.rerun()
        elif entered == guest_code:
            st.session_state["authed"], st.session_state["guest"] = True, True
            st.rerun()
        else:
            st.error("Incorrect passcode.")
    st.caption("This dashboard is private. Enter the passcode to continue.")
    st.stop()


require_passcode()

hc1, hc2 = st.columns([5, 2], vertical_alignment="center")
with hc1:
    st.markdown(
        "<div class='mpmg-sub'>Private Wealth &nbsp;·&nbsp; Quantitative Strategy</div>"
        f"<div class='mpmg-asof'>As of {dt.date.today():%B %d, %Y}</div>",
        unsafe_allow_html=True)
GUEST = st.session_state.get("guest", False)
with hc2:
    HIDE = st.toggle("Hide Balances", value=GUEST, disabled=GUEST,
                     help="Mask dollar amounts; percentages stay visible.") or GUEST
    if GUEST:
        st.caption("Guest view — balances locked")
st.markdown("<hr class='mpmg-rule'>", unsafe_allow_html=True)

# Prompt for this month's OPO buy once it's due (skippable for the session)
if not GUEST and not st.session_state.get("opo_skip"):
    _opo_due = opo_pending_month(db.get_contributions_df(), dt.date.today())
    if _opo_due:
        opo_buy_dialog(_opo_due)

(tab_overview, tab_accounts, tab_bench, tab_contrib, tab_trade, tab_corr,
 tab_lev, tab_factor, tab_ips) = st.tabs(
    ["Overview", "Accounts", "Benchmarks", "Contributions", "Add trade",
     "Correlations", "Leverage", "Factor Exposure", "IPS"])

# ----------------------------------------------------------------- Overview
with tab_overview:
    pos, totals = load_portfolio()
    if totals:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Market Value", money(totals["market_value"]))
        c2.metric("Total Gain / Loss", money(totals["gain_loss"]),
                  f"{totals['gain_loss_pct']:.2%}")
        c3.metric("Today's P&L", money(totals["daily_pnl"]),
                  f"{totals['daily_pnl_pct']:.2%}")
        c4.metric("Book Value", money(totals["book_value"]))

    if not pos.empty:
        left, right = st.columns(2)
        with left:
            st.subheader("Allocation by Holding")
            fig = go.Figure(go.Pie(
                labels=pos["Ticker"], values=pos["Market Value"], hole=.6,
                marker=dict(colors=PALETTE), textinfo="label+percent"))
            fig.update_layout(annotations=[dict(
                text="" if HIDE else money(totals["market_value"]), x=.5, y=.5,
                font=dict(family=SERIF, size=16, color="#F4F4F4"), showarrow=False)])
            show(style_fig(fig, 360, legend=False))
        with right:
            # show % when balances are hidden, $ otherwise
            metric = "Gain/Loss %" if HIDE else "Gain/Loss $"
            st.subheader("Gain / Loss by Holding" + (" (%)" if HIDE else ""))
            g = pos.sort_values(metric)
            txt = ([f"{v:.2%}" for v in g[metric]] if HIDE
                   else [money(v) for v in g[metric]])
            fig = go.Figure(go.Bar(
                x=g[metric], y=g["Ticker"], orientation="h",
                marker_color=["#E0533D" if v < 0 else "#16C784" for v in g[metric]],
                text=txt, textposition="auto"))
            fig = style_fig(fig, 360, legend=False)
            if HIDE:
                fig.update_xaxes(tickformat=".0%")
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
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=snaps["date"], y=snaps["market_value"],
                          mode="lines+markers", name="Market Value",
                          line=dict(color="#4C9AFF", width=2), marker=dict(size=8)))
            if snaps["book_value"].notna().any():
                fig.add_trace(go.Scatter(x=snaps["date"], y=snaps["book_value"],
                              mode="lines+markers", name="Book Value",
                              line=dict(color=GOLD, width=1.5, dash="dot"),
                              marker=dict(size=7)))
            fig = style_fig(fig, 360)
            fig.update_yaxes(title="CAD", tickformat="$,.0f")
            fig.update_xaxes(type="date", tickformat="%b %d, %Y")
            # zoom y to the data so daily moves are visible (not anchored at $0)
            yvals = pd.concat([snaps["market_value"], snaps["book_value"]]).dropna()
            if len(yvals):
                lo, hi = float(yvals.min()), float(yvals.max())
                pad = (hi - lo) * 0.25 if hi > lo else max(hi * 0.02, 1.0)
                fig.update_yaxes(range=[lo - pad, hi + pad])
            if len(snaps) == 1:           # give a lone point some horizontal room
                d = snaps["date"].iloc[0]
                fig.update_xaxes(range=[d - pd.Timedelta(days=4),
                                        d + pd.Timedelta(days=4)])
            if HIDE:
                fig.update_yaxes(showticklabels=False, title=None)
            show(fig)
            st.caption("Recorded each weekday — a mid-morning snapshot, finalized at "
                       "the close. The y-axis zooms to the data, so the gap between the "
                       "lines is your unrealized gain.")
        else:
            st.info("No snapshots yet — they appear once the daily job runs.")
    with c2:
        st.subheader("Currency Exposure")
        cur = portfolio.currency_exposure(pos)
        cmap = {"CAD": "#C9A227", "USD": "#4C9AFF"}
        fig = go.Figure(go.Pie(
            labels=cur.index, values=cur.values, hole=.6,
            marker=dict(colors=[cmap.get(c, "#8A8F9A") for c in cur.index]),
            textinfo="label+percent"))
        fig.update_layout(annotations=[dict(
            text=f"{cur.get('USD', 0.0) / cur.sum():.0%} USD", x=.5, y=.5,
            font=dict(family=SERIF, size=15, color="#F4F4F4"), showarrow=False)])
        show(style_fig(fig, 360, legend=False))
        st.caption("By trading currency, except unhedged XUS is counted as USD "
                   "(it holds the S&P 500 though it trades in CAD).")

# ----------------------------------------------------------------- Accounts
with tab_accounts:
    pos, totals = load_portfolio()
    if not totals:
        st.info("No positions yet.")
    else:
        amv = totals["account_mv"]
        tot = amv["FHSA"] + amv["TFSA"]
        c1, c2, c3 = st.columns(3)
        c1.metric("FHSA Balance", money(amv["FHSA"]),
                  f"{amv['FHSA'] / tot:.1%} of total" if tot else None,
                  delta_color="off")
        c2.metric("TFSA Balance", money(amv["TFSA"]),
                  f"{amv['TFSA'] / tot:.1%} of total" if tot else None,
                  delta_color="off")
        c3.metric("Combined", money(tot))

        left, right = st.columns(2)
        with left:
            st.subheader("Balance by Account")
            fig = go.Figure(go.Pie(
                labels=["FHSA", "TFSA"], values=[amv["FHSA"], amv["TFSA"]],
                hole=.6, marker=dict(colors=[GOLD, "#4C9AFF"]),
                textinfo="label+percent"))
            show(style_fig(fig, 360, legend=False))
        with right:
            st.subheader("Holdings by Account")
            fig = go.Figure()
            fig.add_trace(go.Bar(name="FHSA", x=pos["Ticker"], y=pos["MV FHSA"],
                                 marker_color=GOLD))
            fig.add_trace(go.Bar(name="TFSA", x=pos["Ticker"], y=pos["MV TFSA"],
                                 marker_color="#4C9AFF"))
            fig = style_fig(fig, 360)
            fig.update_layout(barmode="stack")
            fig.update_yaxes(title="CAD", tickformat="$,.0f")
            if HIDE:
                fig.update_yaxes(showticklabels=False, title=None)
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

# ----------------------------------------------------------------- Benchmarks
with tab_bench:
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
        st.warning("Couldn't load price history (offline?). Try again shortly.")
    else:
        fig = go.Figure()
        for sym, name in name_by_sym.items():
            if name in chosen and sym in perf.columns:
                fig.add_trace(go.Scatter(x=perf.index, y=perf[sym],
                              mode="lines", name=name))
        if show_pf:
            pp = perf_portfolio(period)
            if not pp.empty:
                fig.add_trace(go.Scatter(x=pp.index, y=pp.values, mode="lines",
                              name="Portfolio", line=dict(width=4, color="#4C9AFF")))
        bench_color = {"Total Market": GOLD, "S&P 500": "#F4F4F4"}
        bench_on = {"Total Market": show_tm, "S&P 500": show_sp}
        for label, sym in portfolio.BENCHMARKS.items():
            if bench_on[label] and sym in perf.columns:
                fig.add_trace(go.Scatter(x=perf.index, y=perf[sym], mode="lines",
                              name=label,
                              line=dict(width=3, dash="dash", color=bench_color[label])))
        fig = style_fig(fig, 480)
        fig.update_yaxes(title="Growth of $100")
        fig.update_xaxes(type="date", tickformat="%b %Y")
        show(fig)
        st.caption("Each line rebased to 100 at the period start — relative price "
                   "performance (no balances shown). Portfolio = your current holdings "
                   "backtested on price history (OPO excluded, no market data). "
                   "Toggle Portfolio / Total Market (VT) / S&P 500 (^GSPC) above.")

# ----------------------------------------------------------------- Contributions
with tab_contrib:
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
                  f"{tfsa_used / tfsa_room:.0%} of ${tfsa_room:,.0f} used",
                  delta_color="off")
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
    else:
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

# ----------------------------------------------------------------- Add trade
with tab_trade:
    inst = db.get_instruments_df()
    if not GUEST:
        if st.button("➕ Record OPO monthly buy"):
            today = dt.date.today()
            ym = (opo_pending_month(db.get_contributions_df(), today)
                  or (today.year, today.month))
            st.session_state["opo_skip"] = False
            opo_buy_dialog(ym)

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

# ----------------------------------------------------------------- Leverage
with tab_lev:
    st.subheader("Leverage")
    pos, totals = load_portfolio()
    if not totals:
        st.info("No positions yet.")
    else:
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

        yld_pct = load_yield() * 100.0           # weighted trailing ETF dividend yield
        sh = load_sharpe()
        snaps = db.get_snapshots_df()
        peak = snaps["market_value"].max() if len(snaps) else totals["market_value"]
        lev = portfolio.leverage_metrics(totals["market_value"], loc, prime, spread, yld_pct, peak)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Leverage factor", f"{lev['leverage']:.2f}×")
        c2.metric("Equity", money(lev["equity"]))
        c3.metric("Gross exposure", money(lev["gross_exposure"]))
        c4.metric("LOC rate", f"{lev['loc_rate'] * 100:.2f}%")
        d1, d2, d3 = st.columns(3)
        d1.metric("Annual interest", money(lev["annual_interest"]))
        d2.metric("Monthly interest", money(lev["monthly_interest"]))
        d3.metric("Drawdown from peak", f"{lev['drawdown']:.1%}")
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

# ----------------------------------------------------------------- Factor Exposure
with tab_factor:
    st.subheader("Factor Exposure")
    st.caption("Returns-based Fama-French 5 + momentum loadings, MV-weighted across holdings.")
    try:
        fx = load_factors()
    except Exception as e:
        fx = None
        st.warning(f"Couldn't load factor data (offline or source unavailable): {e}")
    if fx:
        port = fx["portfolio"]
        vals = [port[f] for f in fx["factors"]]
        fig = go.Figure(go.Bar(
            x=fx["factors"], y=vals,
            marker_color=[GOLD if v >= 0 else "#E0533D" for v in vals],
            text=[f"{v:+.2f}" for v in vals], textposition="outside"))
        fig = style_fig(fig, 420, legend=False)
        fig.update_yaxes(title="Loading (β)")
        show(fig)
        if fx["unattributed"] > 0.001:
            st.caption(f"Loadings cover the market-holding sleeve; {fx['unattributed']:.0%} of the "
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
                   "CAD-listed funds show market β below 1.0 partly from CAD-vs-USD drift against the "
                   "USD factor set; AVGE (USD) carries the value (HML)/profitability tilts of its "
                   "Avantis design; ZMMK (cash) reads ~0. Funds under ~2 years old have noisier "
                   "estimates. Updated daily.")

# ----------------------------------------------------------------- IPS
with tab_ips:
    _ips_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ips.md")
    try:
        with open(_ips_path, encoding="utf-8") as f:
            st.markdown(f.read())
    except FileNotFoundError:
        st.info("Investment Policy Statement not found (ips.md missing).")

# ----------------------------------------------------------------- Correlations
with tab_corr:
    st.subheader("Correlation of Daily Returns (~1Y)")
    corr = load_corr()
    if corr.empty:
        st.warning("Couldn't load price history (offline?). Try again shortly.")
    else:
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu",
                        zmin=-1, zmax=1, aspect="auto")
        show(style_fig(fig, 440, legend=False))
        st.caption("1.0 = move together · 0 = unrelated · negative = move opposite. "
                   "Private holdings (OPO) are excluded — no market data.")
