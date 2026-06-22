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
    return portfolio.correlation_matrix(db.get_instruments_df())


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


def color_pnl(v):
    if isinstance(v, (int, float)):
        if v > 0:
            return "color:#16C784"
        if v < 0:
            return "color:#E0533D"
    return "color:#9AA0AB"


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

(tab_overview, tab_accounts, tab_bench, tab_contrib,
 tab_trade, tab_corr) = st.tabs(
    ["Overview", "Accounts", "Benchmarks", "Contributions",
     "Add trade", "Correlations"])

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
                          line=dict(color="#4C9AFF", width=2),
                          marker=dict(size=8),
                          fill="tozeroy", fillcolor="rgba(76,154,255,.08)"))
            if snaps["book_value"].notna().any():
                fig.add_trace(go.Scatter(x=snaps["date"], y=snaps["book_value"],
                              mode="lines+markers", name="Book Value",
                              line=dict(color=GOLD, width=1.5, dash="dot"),
                              marker=dict(size=7)))
            fig = style_fig(fig, 360)
            fig.update_yaxes(title="CAD", tickformat="$,.0f")
            if HIDE:
                fig.update_yaxes(showticklabels=False, title=None)
            fig.update_xaxes(type="date", tickformat="%b %d, %Y")
            if len(snaps) == 1:           # avoid a degenerate sub-second axis
                d = snaps["date"].iloc[0]
                fig.update_xaxes(range=[d - pd.Timedelta(days=4),
                                        d + pd.Timedelta(days=4)])
                fig.update_yaxes(range=[0, snaps["market_value"].iloc[0] * 1.15])
            show(fig)
            st.caption("Recorded automatically each weekday by the snapshot job — "
                       "the line fills in as history builds.")
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
