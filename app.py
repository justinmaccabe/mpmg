"""Maccabe Portfolio Management Group — portfolio dashboard.

Run locally:   streamlit run app.py        (uses local SQLite, no setup)
Deployed:      set DATABASE_URL secret      (uses Postgres so data persists)
"""
import datetime as dt

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
.mpmg-header {{ border-bottom: 1px solid rgba(201,162,39,.35); padding: .25rem 0 1rem;
    margin-bottom: 1.4rem; }}
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
  <div class="mpmg-sub">Private Wealth &nbsp;·&nbsp; Quantitative Strategy</div>
  <div class="mpmg-asof">As of {dt.date.today():%B %d, %Y}</div>
</div>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------- data
@st.cache_data(ttl=900)
def load_portfolio():
    return portfolio.build_portfolio(db.get_transactions_df(), db.get_instruments_df())


@st.cache_data(ttl=3600)
def load_corr():
    return portfolio.correlation_matrix(db.get_instruments_df())


def money(x):
    return f"${x:,.2f}"


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


tab_overview, tab_accounts, tab_trade, tab_corr = st.tabs(
    ["Overview", "Accounts", "Add trade", "Correlations"])

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
                text=f"{money(totals['market_value'])}", x=.5, y=.5,
                font=dict(family=SERIF, size=16, color="#F4F4F4"), showarrow=False)])
            show(style_fig(fig, 360, legend=False))
        with right:
            st.subheader("Gain / Loss by Holding")
            g = pos.sort_values("Gain/Loss $")
            fig = go.Figure(go.Bar(
                x=g["Gain/Loss $"], y=g["Ticker"], orientation="h",
                marker_color=["#E0533D" if v < 0 else "#16C784" for v in g["Gain/Loss $"]],
                text=[money(v) for v in g["Gain/Loss $"]], textposition="auto"))
            show(style_fig(fig, 360, legend=False))

        st.subheader("Holdings")
        cols = ["Ticker", "Account", "Shares", "ACB", "Price", "Cur",
                "Market Value", "Book Value", "Gain/Loss $", "Gain/Loss %",
                "Daily P&L $"]
        styler = (pos[cols].style
                  .format({"Shares": "{:,.2f}", "ACB": "${:,.2f}", "Price": "${:,.2f}",
                           "Market Value": "${:,.0f}", "Book Value": "${:,.0f}",
                           "Gain/Loss $": "${:,.0f}", "Gain/Loss %": "{:.2%}",
                           "Daily P&L $": "${:,.0f}"})
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
        cur = pos.groupby("Cur")["Market Value"].sum()
        fig = go.Figure(go.Pie(
            labels=cur.index, values=cur.values, hole=.6,
            marker=dict(colors=["#C9A227", "#4C9AFF", "#16C784"]),
            textinfo="label+percent"))
        usd = cur.get("USD", 0.0)
        fig.update_layout(annotations=[dict(
            text=f"{usd / cur.sum():.0%} USD", x=.5, y=.5,
            font=dict(family=SERIF, size=15, color="#F4F4F4"), showarrow=False)])
        show(style_fig(fig, 360, legend=False))
        st.caption("Share of the book denominated in each currency — your FX risk.")

# ----------------------------------------------------------------- Accounts
with tab_accounts:
    pos, totals = load_portfolio()
    if not totals:
        st.info("No positions yet.")
    else:
        amv = totals["account_mv"]
        c1, c2, c3 = st.columns(3)
        c1.metric("FHSA Balance", money(amv["FHSA"]))
        c2.metric("TFSA Balance", money(amv["TFSA"]))
        tot = amv["FHSA"] + amv["TFSA"]
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
            show(fig)

        st.subheader("Per-Account Detail")
        detail = pos[["Ticker", "MV FHSA", "MV TFSA"]].copy()
        detail["Total"] = detail["MV FHSA"] + detail["MV TFSA"]
        total_row = pd.DataFrame([{
            "Ticker": "Total", "MV FHSA": detail["MV FHSA"].sum(),
            "MV TFSA": detail["MV TFSA"].sum(), "Total": detail["Total"].sum()}])
        detail = pd.concat([detail, total_row], ignore_index=True)
        st.dataframe(detail.style.format(
            {"MV FHSA": "${:,.0f}", "MV TFSA": "${:,.0f}", "Total": "${:,.0f}"}),
            width="stretch", hide_index=True)

# ----------------------------------------------------------------- Add trade
with tab_trade:
    st.subheader("Log a Transaction")
    inst = db.get_instruments_df()
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
    st.dataframe(db.get_transactions_df(), width="stretch", hide_index=True)

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
