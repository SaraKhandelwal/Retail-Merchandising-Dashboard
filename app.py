"""
app.py — Luxury Retail Merchandising Analytics Dashboard
A synthetic portfolio project demonstrating luxury retail BI & merchandising analytics.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from helper_functions import (
    generate_synthetic_data,
    apply_filters,
    generate_insights,
    classify_inventory_health,
    compute_opportunity_score,
    fmt_currency, fmt_pct, fmt_number,
    plotly_layout, PALETTE, LUXURY_COLORS,
)

# ──────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Maison Analytics | Luxury Retail Intelligence",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
#  GLOBAL CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;0,600;1,300;1,400&family=Jost:wght@300;400;500&display=swap');

/* ── Root & background ── */
html, body, [class*="css"] {
    font-family: 'Jost', sans-serif;
    background-color: #FDFAF5;
    color: #1C1C1C;
}
.stApp { background-color: #FDFAF5; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1400px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #1C1C1C !important;
    border-right: 1px solid #2E2E2E;
}
[data-testid="stSidebar"] * {
    color: #F5F2EC !important;
    font-family: 'Jost', sans-serif !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stDateInput label {
    font-size: 0.72rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #9B9184 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] {
    background-color: #2E2E2E !important;
    border-color: #3E3E3E !important;
}
[data-testid="stSidebar"] .stRadio label span {
    font-size: 0.85rem !important;
}
[data-testid="stSidebarContent"] { padding: 1.5rem 1.2rem; }

/* ── KPI cards ── */
.kpi-card {
    background: #FFFFFF;
    border: 1px solid #E8E4DC;
    border-top: 3px solid #C9A84C;
    border-radius: 2px;
    padding: 1.4rem 1.2rem 1.2rem;
    margin-bottom: 0.5rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
}
.kpi-label {
    font-family: 'Jost', sans-serif;
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #8B7D6B;
    margin-bottom: 0.5rem;
}
.kpi-value {
    font-family: 'Cormorant Garamond', Georgia, serif;
    font-size: 2.0rem;
    font-weight: 500;
    color: #1C1C1C;
    line-height: 1;
    margin-bottom: 0.2rem;
}
.kpi-sub {
    font-size: 0.72rem;
    color: #9B9184;
    letter-spacing: 0.04em;
}
.kpi-delta-pos { color: #4A6741; font-size: 0.78rem; }
.kpi-delta-neg { color: #8B3A3A; font-size: 0.78rem; }

/* ── Section headers ── */
.section-title {
    font-family: 'Cormorant Garamond', Georgia, serif;
    font-size: 1.55rem;
    font-weight: 400;
    color: #1C1C1C;
    letter-spacing: 0.02em;
    margin-bottom: 0.1rem;
    margin-top: 0.5rem;
}
.section-rule {
    border: none;
    border-top: 1px solid #C9A84C;
    margin: 0.3rem 0 1.5rem 0;
    width: 4rem;
}
.section-sub {
    font-size: 0.82rem;
    color: #8B7D6B;
    letter-spacing: 0.06em;
    margin-bottom: 1.5rem;
}

/* ── Page title ── */
.page-masthead {
    border-bottom: 1px solid #E8E4DC;
    padding-bottom: 1.2rem;
    margin-bottom: 2rem;
}
.page-title {
    font-family: 'Cormorant Garamond', Georgia, serif;
    font-size: 2.2rem;
    font-weight: 300;
    letter-spacing: 0.06em;
    color: #1C1C1C;
    margin-bottom: 0.15rem;
}
.page-subtitle {
    font-family: 'Jost', sans-serif;
    font-size: 0.78rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #8B7D6B;
}
.gold-dot { color: #C9A84C; }

/* ── Insight cards ── */
.insight-box {
    background: #FFFFFF;
    border-left: 3px solid #C9A84C;
    border-radius: 0 2px 2px 0;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.7rem;
    font-size: 0.88rem;
    color: #2C2C2C;
    line-height: 1.55;
    box-shadow: 0 1px 6px rgba(0,0,0,0.04);
}

/* ── Table styling ── */
[data-testid="stDataFrame"] {
    border: 1px solid #E8E4DC;
    border-radius: 2px;
}

/* ── Chart containers ── */
.chart-wrapper {
    background: #FFFFFF;
    border: 1px solid #E8E4DC;
    border-radius: 2px;
    padding: 1rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.03);
}

/* ── Sidebar nav radio ── */
.stRadio > div { gap: 0.3rem; }
.stRadio label {
    background: transparent;
    border-radius: 1px;
    padding: 0.4rem 0;
    cursor: pointer;
}

/* ── Divider ── */
.luxury-divider {
    text-align: center;
    color: #C9A84C;
    font-size: 1.1rem;
    letter-spacing: 0.8em;
    margin: 1.5rem 0;
    opacity: 0.6;
}

/* ── Sidebar logo area ── */
.sidebar-brand {
    text-align: center;
    padding: 1rem 0 1.5rem;
    border-bottom: 1px solid #2E2E2E;
    margin-bottom: 1.5rem;
}
.sidebar-brand-name {
    font-family: 'Cormorant Garamond', Georgia, serif;
    font-size: 1.3rem;
    font-weight: 400;
    letter-spacing: 0.25em;
    color: #F5F2EC !important;
}
.sidebar-brand-sub {
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #9B9184 !important;
    margin-top: 0.2rem;
}
.sidebar-gold { color: #C9A84C !important; }

/* ── Download button ── */
.stDownloadButton button {
    background-color: transparent !important;
    color: #C9A84C !important;
    border: 1px solid #C9A84C !important;
    border-radius: 1px !important;
    font-family: 'Jost', sans-serif !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    padding: 0.4rem 1rem !important;
}
.stDownloadButton button:hover {
    background-color: #C9A84C !important;
    color: #1C1C1C !important;
}

/* ── Metric delta override ── */
[data-testid="stMetricDelta"] { display: none; }

/* ── Tag badge ── */
.badge {
    display: inline-block;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.15rem 0.5rem;
    border-radius: 1px;
    margin-right: 0.3rem;
}
.badge-gold   { background: #F5F0E0; color: #8B6914; border: 1px solid #E8D5A3; }
.badge-sage   { background: #EDF2EE; color: #3D6B44; border: 1px solid #BDD4C0; }
.badge-blush  { background: #F5EDEA; color: #7A4040; border: 1px solid #DEC4BC; }
.badge-taupe  { background: #F0EDE8; color: #5C504A; border: 1px solid #D0C8BC; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
#  DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────
DATA_PATH = "synthetic_luxury_retail_data.csv"

@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    else:
        df = generate_synthetic_data(2500)
        df.to_csv(DATA_PATH, index=False)
    return df

with st.spinner("Loading intelligence platform…"):
    df = load_data()


# ──────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
      <div class="sidebar-brand-name">✦ MAISON</div>
      <div class="sidebar-brand-sub">Analytics Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "NAVIGATION",
        ["Executive Overview", "Merchandising Analytics", "Advanced Analytics"],
        label_visibility="visible",
    )

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.65rem;letter-spacing:0.15em;text-transform:uppercase;color:#9B9184;margin-bottom:0.8rem;'>Filters</p>", unsafe_allow_html=True)

    all_regions = sorted(df["Region"].unique())
    sel_regions = st.multiselect("Region", all_regions, default=all_regions)

    available_markets = sorted(df[df["Region"].isin(sel_regions)]["Market"].unique()) if sel_regions else sorted(df["Market"].unique())
    sel_markets = st.multiselect("Market", available_markets, default=available_markets)

    all_cats = sorted(df["Product Category"].unique())
    sel_cats = st.multiselect("Category", all_cats, default=all_cats)

    all_colls = sorted(df["Collection Type"].unique())
    sel_colls = st.multiselect("Collection", all_colls, default=all_colls)

    all_channels = sorted(df["Channel"].unique())
    sel_channels = st.multiselect("Channel", all_channels, default=all_channels)

    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()
    date_range = st.date_input("Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    if len(date_range) == 2:
        filters = {
            "regions": sel_regions or all_regions,
            "markets": sel_markets or available_markets,
            "categories": sel_cats or all_cats,
            "collections": sel_colls or all_colls,
            "channels": sel_channels or all_channels,
            "date_range": date_range,
        }
    else:
        filters = {}

    dff = apply_filters(df, filters)

    st.download_button(
        "↓  Export Dataset",
        data=dff.to_csv(index=False).encode("utf-8"),
        file_name="maison_analytics_export.csv",
        mime="text/csv",
    )

    st.markdown(f"""
    <div style='margin-top:2rem;font-size:0.62rem;color:#5C5C5C;letter-spacing:0.06em;text-align:center;'>
    {len(dff):,} records · Synthetic Portfolio Data<br>
    <span style='color:#C9A84C;'>✦</span> 2023 – 2024
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
#  HELPERS FOR CHARTS
# ──────────────────────────────────────────────────────────────────────────────
def kpi_card(label: str, value: str, sub: str = "", delta: str = "", pos: bool = True) -> str:
    delta_class = "kpi-delta-pos" if pos else "kpi-delta-neg"
    delta_html  = f'<div class="{delta_class}">{delta}</div>' if delta else ""
    return f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      {delta_html}
      <div class="kpi-sub">{sub}</div>
    </div>"""


def section_header(title: str, subtitle: str = ""):
    st.markdown(f'<div class="section-title">{title}</div><hr class="section-rule">', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-sub">{subtitle}</div>', unsafe_allow_html=True)


def chart_wrap(fig):
    """Render a plotly chart inside a styled wrapper."""
    st.markdown('<div class="chart-wrapper">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — EXECUTIVE OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Executive Overview":

    st.markdown("""
    <div class="page-masthead">
      <div class="page-title">Executive <span class="gold-dot">Overview</span></div>
      <div class="page-subtitle">Global Retail Performance Intelligence · Maison Analytics</div>
    </div>
    """, unsafe_allow_html=True)

    if dff.empty:
        st.warning("No data matches the current filters. Please adjust your selections.")
        st.stop()

    # ── KPIs ─────────────────────────────────────────────────────────────────
    total_rev   = dff["Revenue"].sum()
    units_sold  = dff["Units Sold"].sum()
    avg_asp     = dff["Average Selling Price"].mean()
    avg_str     = dff["Sell Through Rate"].mean()
    avg_gm      = dff["Gross Margin %"].mean()
    avg_ret     = dff["Return Rate"].mean()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: st.markdown(kpi_card("Total Revenue", fmt_currency(total_rev), "Gross retail"), unsafe_allow_html=True)
    with c2: st.markdown(kpi_card("Units Sold", fmt_number(units_sold), "Across all channels"), unsafe_allow_html=True)
    with c3: st.markdown(kpi_card("Avg Selling Price", fmt_currency(avg_asp, abbrev=False), "Weighted mean ASP"), unsafe_allow_html=True)
    with c4: st.markdown(kpi_card("Sell-Through Rate", fmt_pct(avg_str), "Inventory efficiency"), unsafe_allow_html=True)
    with c5: st.markdown(kpi_card("Gross Margin", fmt_pct(avg_gm), "Before operating costs"), unsafe_allow_html=True)
    with c6: st.markdown(kpi_card("Return Rate", fmt_pct(avg_ret), "All channels blended"), unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # ── Revenue Trend ─────────────────────────────────────────────────────────
    section_header("Revenue Trend", "Weekly retail revenue performance across selected markets")

    weekly = dff.set_index("Date").resample("W")["Revenue"].sum().reset_index()
    weekly.columns = ["Week", "Revenue"]

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=weekly["Week"], y=weekly["Revenue"],
        mode="lines", line=dict(color="#C9A84C", width=2),
        fill="tozeroy", fillcolor="rgba(201,168,76,0.08)",
        hovertemplate="<b>%{x|%d %b %Y}</b><br>€%{y:,.0f}<extra></extra>",
    ))
    fig_trend.update_layout(**plotly_layout("", 300))
    chart_wrap(fig_trend)

    # ── Region & Category Revenue ─────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        section_header("Revenue by Region")
        reg_rev = dff.groupby("Region")["Revenue"].sum().reset_index().sort_values("Revenue", ascending=True)
        fig_reg = go.Figure(go.Bar(
            x=reg_rev["Revenue"], y=reg_rev["Region"],
            orientation="h",
            marker_color=PALETTE[:len(reg_rev)],
            text=[fmt_currency(v) for v in reg_rev["Revenue"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>€%{x:,.0f}<extra></extra>",
        ))
        fig_reg.update_layout(**plotly_layout("", 300))
        chart_wrap(fig_reg)

    with col_right:
        section_header("Revenue by Category")
        cat_rev = dff.groupby("Product Category")["Revenue"].sum().reset_index().sort_values("Revenue", ascending=False)
        fig_cat = go.Figure(go.Bar(
            x=cat_rev["Product Category"], y=cat_rev["Revenue"],
            marker_color=PALETTE[:len(cat_rev)],
            text=[fmt_currency(v) for v in cat_rev["Revenue"]],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>€%{y:,.0f}<extra></extra>",
        ))
        fig_cat.update_layout(**plotly_layout("", 300))
        chart_wrap(fig_cat)

    # ── Market Comparison ────────────────────────────────────────────────────
    section_header("Market Comparison", "Revenue, ASP, and sell-through by market")

    mkt = dff.groupby("Market").agg(
        Revenue=("Revenue", "sum"),
        ASP=("Average Selling Price", "mean"),
        SellThrough=("Sell Through Rate", "mean"),
        GrossMargin=("Gross Margin %", "mean"),
    ).reset_index().sort_values("Revenue", ascending=False)

    fig_mkt = make_subplots(specs=[[{"secondary_y": True}]])
    fig_mkt.add_trace(go.Bar(
        x=mkt["Market"], y=mkt["Revenue"],
        name="Revenue", marker_color="#C9A84C", opacity=0.85,
    ), secondary_y=False)
    fig_mkt.add_trace(go.Scatter(
        x=mkt["Market"], y=mkt["ASP"],
        name="Avg ASP", mode="lines+markers",
        line=dict(color="#8B7D6B", width=2),
        marker=dict(size=7, symbol="diamond"),
    ), secondary_y=True)
    fig_mkt.update_layout(**plotly_layout("", 340))
    fig_mkt.update_yaxes(title_text="Revenue (€)", secondary_y=False, showgrid=True, gridcolor="#E8E4DC")
    fig_mkt.update_yaxes(title_text="Avg Selling Price (€)", secondary_y=True, showgrid=False)
    chart_wrap(fig_mkt)

    # ── Top Product Families ─────────────────────────────────────────────────
    col_a, col_b = st.columns([3, 2])

    with col_a:
        section_header("Top-Performing Product Families")
        fam_rev = dff.groupby(["Product Family", "Product Category"]).agg(
            Revenue=("Revenue", "sum"),
            Units=("Units Sold", "sum"),
            ASP=("Average Selling Price", "mean"),
        ).reset_index().sort_values("Revenue", ascending=False).head(12)

        fig_fam = go.Figure(go.Bar(
            x=fam_rev["Revenue"],
            y=fam_rev["Product Family"],
            orientation="h",
            marker=dict(color=fam_rev["Revenue"], colorscale=[[0, "#E8D5A3"], [1, "#C9A84C"]]),
            text=[fmt_currency(v) for v in fam_rev["Revenue"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Revenue: €%{x:,.0f}<extra></extra>",
        ))
        fig_fam.update_layout(**plotly_layout("", 380))
        chart_wrap(fig_fam)

    with col_b:
        section_header("Channel Mix")
        ch_rev = dff.groupby("Channel")["Revenue"].sum().reset_index()
        fig_pie = go.Figure(go.Pie(
            labels=ch_rev["Channel"],
            values=ch_rev["Revenue"],
            hole=0.65,
            marker=dict(colors=PALETTE[:3], line=dict(color="#FDFAF5", width=2)),
            textinfo="label+percent",
            textfont=dict(family="Jost, sans-serif", size=11),
            hovertemplate="<b>%{label}</b><br>€%{value:,.0f}<br>%{percent}<extra></extra>",
        ))
        fig_pie.update_layout(
            **plotly_layout("", 380),
            annotations=[dict(text="Channel<br>Mix", x=0.5, y=0.5, font=dict(
                family="Cormorant Garamond, serif", size=14, color="#1C1C1C"), showarrow=False)],
        )
        chart_wrap(fig_pie)

    # ── Key Business Insights ────────────────────────────────────────────────
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    section_header("Key Business Insights", "Automated intelligence derived from current filter selection")

    insights = generate_insights(dff)
    for ins in insights:
        st.markdown(f'<div class="insight-box">✦ &nbsp;{ins}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — MERCHANDISING & COLLECTION ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Merchandising Analytics":

    st.markdown("""
    <div class="page-masthead">
      <div class="page-title">Merchandising <span class="gold-dot">&amp; Collection</span> Analytics</div>
      <div class="page-subtitle">Assortment Intelligence · Inventory Health · Collection Performance</div>
    </div>
    """, unsafe_allow_html=True)

    if dff.empty:
        st.warning("No data matches the current filters.")
        st.stop()

    # ── Collection Performance ───────────────────────────────────────────────
    section_header("Collection Performance Analysis", "Revenue, sell-through, and margin by collection type")

    coll = dff.groupby("Collection Type").agg(
        Revenue=("Revenue", "sum"),
        Units=("Units Sold", "sum"),
        SellThrough=("Sell Through Rate", "mean"),
        GrossMargin=("Gross Margin %", "mean"),
        ASP=("Average Selling Price", "mean"),
        ReturnRate=("Return Rate", "mean"),
    ).reset_index()

    col1, col2 = st.columns(2)

    with col1:
        fig_coll_rev = go.Figure(go.Bar(
            x=coll["Collection Type"], y=coll["Revenue"],
            marker_color=PALETTE[:len(coll)],
            text=[fmt_currency(v) for v in coll["Revenue"]],
            textposition="outside",
        ))
        fig_coll_rev.update_layout(**plotly_layout("Revenue by Collection", 320))
        chart_wrap(fig_coll_rev)

    with col2:
        fig_coll_str = go.Figure()
        fig_coll_str.add_trace(go.Bar(
            name="Sell-Through %", x=coll["Collection Type"],
            y=coll["SellThrough"] * 100, marker_color="#C9A84C", opacity=0.85,
        ))
        fig_coll_str.add_trace(go.Bar(
            name="Gross Margin %", x=coll["Collection Type"],
            y=coll["GrossMargin"] * 100, marker_color="#7A8C7E", opacity=0.85,
        ))
        fig_coll_str.update_layout(**plotly_layout("Sell-Through vs Gross Margin by Collection", 320), barmode="group")
        chart_wrap(fig_coll_str)

    # ── Sell-Through Heatmap ─────────────────────────────────────────────────
    section_header("Sell-Through Heatmap", "Sell-through rate by category and market")

    pivot_str = dff.pivot_table(
        index="Product Category", columns="Market",
        values="Sell Through Rate", aggfunc="mean"
    ).fillna(0)

    fig_heat = go.Figure(go.Heatmap(
        z=pivot_str.values * 100,
        x=pivot_str.columns.tolist(),
        y=pivot_str.index.tolist(),
        colorscale=[[0, "#F5F0E0"], [0.5, "#E8D5A3"], [1, "#C9A84C"]],
        text=[[f"{v:.1f}%" for v in row] for row in pivot_str.values * 100],
        texttemplate="%{text}",
        textfont=dict(size=11),
        hovertemplate="<b>%{y} · %{x}</b><br>Sell-Through: %{z:.1f}%<extra></extra>",
        colorbar=dict(title=dict(text="STR %", side="right"), thickness=12),
    ))
    fig_heat.update_layout(**plotly_layout("", 340))
    chart_wrap(fig_heat)

    # ── Inventory Health ─────────────────────────────────────────────────────
    section_header("Inventory Health & Risk Analysis", "Classification of inventory conditions by product family")

    dff_inv = dff.copy()
    dff_inv["Health"] = dff_inv.apply(classify_inventory_health, axis=1)

    health_summary = dff_inv.groupby(["Product Category", "Health"]).size().reset_index(name="Count")
    fig_health = px.bar(
        health_summary, x="Product Category", y="Count", color="Health",
        color_discrete_map={
            "Healthy":        "#7A8C7E",
            "Stockout Risk":  "#8B3A3A",
            "Critical Low":   "#C9A84C",
            "Overstock":      "#8B7D6B",
            "Monitor":        "#C4A99A",
        },
        barmode="stack",
    )
    fig_health.update_layout(**plotly_layout("", 340))
    chart_wrap(fig_health)

    # ── Overstock vs Stockout ────────────────────────────────────────────────
    col_os, col_so = st.columns(2)

    with col_os:
        section_header("Overstock Distribution")
        overstock = dff[dff["Inventory Units"] > 20].groupby("Product Family")["Inventory Units"].sum().reset_index()
        overstock = overstock.sort_values("Inventory Units", ascending=True).tail(10)
        fig_os = go.Figure(go.Bar(
            x=overstock["Inventory Units"], y=overstock["Product Family"],
            orientation="h", marker_color="#C4A99A",
            text=overstock["Inventory Units"], textposition="outside",
        ))
        fig_os.update_layout(**plotly_layout("", 320))
        chart_wrap(fig_os)

    with col_so:
        section_header("Stockout Exposure by Market")
        stockout_mkt = dff.groupby("Market")["Stockout Flag"].mean().reset_index()
        stockout_mkt.columns = ["Market", "Stockout Rate"]
        stockout_mkt = stockout_mkt.sort_values("Stockout Rate", ascending=True)
        fig_so = go.Figure(go.Bar(
            x=stockout_mkt["Stockout Rate"] * 100,
            y=stockout_mkt["Market"],
            orientation="h",
            marker=dict(color=stockout_mkt["Stockout Rate"], colorscale=[[0, "#E8D5A3"], [1, "#8B3A3A"]]),
            text=[f"{v:.1f}%" for v in stockout_mkt["Stockout Rate"] * 100],
            textposition="outside",
        ))
        fig_so.update_layout(**plotly_layout("", 320))
        chart_wrap(fig_so)

    # ── Visual Merchandising Impact ─────────────────────────────────────────
    section_header("Visual Merchandising Impact", "Correlation between VM score and revenue performance")

    dff_vm = dff.copy()
    dff_vm["VM_Bucket"] = pd.cut(dff_vm["Visual Merchandising Score"], bins=[0, 5, 6.5, 8, 10],
                                  labels=["Low (<5)", "Moderate (5–6.5)", "Good (6.5–8)", "Excellent (>8)"])
    vm_perf = dff_vm.groupby("VM_Bucket", observed=True).agg(
        Revenue=("Revenue", "mean"),
        Units=("Units Sold", "mean"),
        SellThrough=("Sell Through Rate", "mean"),
    ).reset_index()

    col_vm1, col_vm2 = st.columns([3, 2])

    with col_vm1:
        fig_vm = go.Figure()
        fig_vm.add_trace(go.Bar(
            x=vm_perf["VM_Bucket"].astype(str), y=vm_perf["Revenue"],
            name="Avg Revenue / Transaction",
            marker_color=PALETTE[:len(vm_perf)],
            text=[fmt_currency(v, abbrev=False) for v in vm_perf["Revenue"]],
            textposition="outside",
        ))
        fig_vm.update_layout(**plotly_layout("Avg Revenue by VM Score Band", 310))
        chart_wrap(fig_vm)

    with col_vm2:
        fig_vm2 = go.Figure(go.Scatter(
            x=dff.sample(min(600, len(dff)), random_state=1)["Visual Merchandising Score"],
            y=dff.sample(min(600, len(dff)), random_state=1)["Revenue"],
            mode="markers",
            marker=dict(color="#C9A84C", size=5, opacity=0.45),
            hovertemplate="VM Score: %{x}<br>Revenue: €%{y:,.0f}<extra></extra>",
        ))
        fig_vm2.update_layout(**plotly_layout("Revenue vs VM Score", 310))
        chart_wrap(fig_vm2)

    # ── Margin vs Revenue Scatter ────────────────────────────────────────────
    section_header("Margin vs Revenue by Category", "Identifying high-value, high-margin product opportunities")

    margin_cat = dff.groupby(["Product Category", "Collection Type"]).agg(
        Revenue=("Revenue", "sum"),
        GrossMargin=("Gross Margin %", "mean"),
        Units=("Units Sold", "sum"),
    ).reset_index()

    fig_scatter = px.scatter(
        margin_cat,
        x="GrossMargin", y="Revenue",
        color="Product Category",
        size="Units",
        hover_name="Product Category",
        symbol="Collection Type",
        color_discrete_sequence=PALETTE,
        labels={"GrossMargin": "Gross Margin %", "Revenue": "Revenue (€)"},
    )
    fig_scatter.update_traces(marker=dict(opacity=0.75, line=dict(width=0.5, color="white")))
    fig_scatter.update_layout(**plotly_layout("", 380))
    chart_wrap(fig_scatter)

    # ── Product Opportunity Score ────────────────────────────────────────────
    section_header("Product Opportunity Scoring", "Composite score identifying highest-potential product families")

    opp_df = compute_opportunity_score(dff)

    fig_opp = go.Figure(go.Bar(
        x=opp_df["opportunity_score"].head(12),
        y=opp_df["Product Family"].head(12),
        orientation="h",
        marker=dict(color=opp_df["opportunity_score"].head(12),
                    colorscale=[[0, "#E8D5A3"], [1, "#C9A84C"]]),
        text=[f"{v:.0f}" for v in opp_df["opportunity_score"].head(12)],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Opportunity Score: %{x:.1f}<extra></extra>",
    ))
    fig_opp.update_layout(**plotly_layout("", 360))
    chart_wrap(fig_opp)

    # ── Buying Strategy Table ────────────────────────────────────────────────
    section_header("Buying Strategy Recommendations", "Data-driven assortment guidance by product family")

    opp_display = opp_df[["Product Family", "revenue", "sell_through", "gross_margin",
                            "stockout_rate", "return_rate", "opportunity_score"]].copy()
    opp_display.columns = ["Product Family", "Revenue (€)", "Sell-Through", "Gross Margin",
                            "Stockout Rate", "Return Rate", "Opp. Score"]

    def buying_action(row):
        if row["Opp. Score"] >= 75:  return "🟢 Increase Depth"
        if row["Opp. Score"] >= 55:  return "🟡 Maintain"
        if row["Stockout Rate"] > 0.15: return "🔴 Replenish"
        return "🔵 Review & Rationalise"

    opp_display["Action"] = opp_display.apply(buying_action, axis=1)
    opp_display["Revenue (€)"] = opp_display["Revenue (€)"].apply(lambda x: f"€{x:,.0f}")
    opp_display["Sell-Through"] = opp_display["Sell-Through"].apply(fmt_pct)
    opp_display["Gross Margin"] = opp_display["Gross Margin"].apply(fmt_pct)
    opp_display["Stockout Rate"] = opp_display["Stockout Rate"].apply(fmt_pct)
    opp_display["Return Rate"]   = opp_display["Return Rate"].apply(fmt_pct)

    st.dataframe(opp_display, use_container_width=True, hide_index=True, height=380)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — ADVANCED ANALYTICS & FORECASTING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Advanced Analytics":

    st.markdown("""
    <div class="page-masthead">
      <div class="page-title">Advanced Analytics <span class="gold-dot">&amp; Forecasting</span></div>
      <div class="page-subtitle">Predictive Intelligence · Demand Signals · Strategic Recommendations</div>
    </div>
    """, unsafe_allow_html=True)

    if dff.empty:
        st.warning("No data matches the current filters.")
        st.stop()

    # ── Revenue Forecasting ──────────────────────────────────────────────────
    section_header("Revenue Forecasting", "8-week forward revenue projection using trend decomposition")

    weekly_rev = dff.set_index("Date").resample("W")["Revenue"].sum().reset_index()
    weekly_rev.columns = ["ds", "y"]

    if len(weekly_rev) >= 8:
        # Simple linear trend + seasonality forecast
        from sklearn.linear_model import LinearRegression

        weekly_rev["t"] = np.arange(len(weekly_rev))
        weekly_rev["sin52"] = np.sin(2 * np.pi * weekly_rev["t"] / 52)
        weekly_rev["cos52"] = np.cos(2 * np.pi * weekly_rev["t"] / 52)
        weekly_rev["sin26"] = np.sin(2 * np.pi * weekly_rev["t"] / 26)

        X = weekly_rev[["t", "sin52", "cos52", "sin26"]].values
        y = weekly_rev["y"].values
        model = LinearRegression().fit(X, y)

        last_t = weekly_rev["t"].max()
        last_date = weekly_rev["ds"].max()
        future_dates = [last_date + pd.Timedelta(weeks=i+1) for i in range(8)]
        future_t = np.arange(last_t + 1, last_t + 9)
        X_future = np.column_stack([
            future_t,
            np.sin(2 * np.pi * future_t / 52),
            np.cos(2 * np.pi * future_t / 52),
            np.sin(2 * np.pi * future_t / 26),
        ])
        y_pred = model.predict(X_future)
        y_pred = np.maximum(y_pred, 0)

        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(
            x=weekly_rev["ds"], y=weekly_rev["y"],
            mode="lines", name="Historical",
            line=dict(color="#C9A84C", width=2),
        ))
        fig_fc.add_trace(go.Scatter(
            x=future_dates, y=y_pred,
            mode="lines+markers", name="Forecast",
            line=dict(color="#8B7D6B", width=2, dash="dot"),
            marker=dict(symbol="diamond", size=7, color="#8B7D6B"),
        ))
        fig_fc.add_trace(go.Scatter(
            x=future_dates + future_dates[::-1],
            y=list(y_pred * 1.08) + list(y_pred * 0.92)[::-1],
            fill="toself", fillcolor="rgba(139,125,107,0.10)",
            line=dict(color="rgba(0,0,0,0)"), name="Confidence Band",
        ))
        fig_fc.update_layout(**plotly_layout("", 340))
        chart_wrap(fig_fc)
    else:
        st.info("Insufficient data for forecasting. Please expand the date range.")

    # ── Demand by Category ───────────────────────────────────────────────────
    section_header("Demand Analysis by Category", "Rolling revenue trend per product category")

    cat_weekly = dff.groupby([pd.Grouper(key="Date", freq="W"), "Product Category"])["Revenue"].sum().reset_index()

    fig_cat_trend = px.line(
        cat_weekly, x="Date", y="Revenue", color="Product Category",
        color_discrete_sequence=PALETTE,
        line_shape="spline",
    )
    fig_cat_trend.update_traces(line=dict(width=1.8))
    fig_cat_trend.update_layout(**plotly_layout("", 340))
    chart_wrap(fig_cat_trend)

    # ── Customer Segment Analysis ────────────────────────────────────────────
    section_header("Customer Segment Analysis", "Revenue contribution and behavioural profile by client tier")

    col_seg1, col_seg2 = st.columns(2)

    seg = dff.groupby("Customer Segment").agg(
        Revenue=("Revenue", "sum"),
        ASP=("Average Selling Price", "mean"),
        Units=("Units Sold", "sum"),
        ReturnRate=("Return Rate", "mean"),
        SellThrough=("Sell Through Rate", "mean"),
    ).reset_index()

    with col_seg1:
        fig_seg_rev = go.Figure(go.Pie(
            labels=seg["Customer Segment"], values=seg["Revenue"],
            hole=0.60,
            marker=dict(colors=PALETTE[:4], line=dict(color="#FDFAF5", width=2)),
            textinfo="label+percent",
            textfont=dict(family="Jost, sans-serif", size=11),
        ))
        fig_seg_rev.update_layout(
            **plotly_layout("Revenue by Client Segment", 320),
            annotations=[dict(text="Client<br>Tiers", x=0.5, y=0.5,
                              font=dict(family="Cormorant Garamond, serif", size=13, color="#1C1C1C"),
                              showarrow=False)],
        )
        chart_wrap(fig_seg_rev)

    with col_seg2:
        fig_seg_asp = go.Figure(go.Bar(
            x=seg["Customer Segment"], y=seg["ASP"],
            marker_color=PALETTE[:len(seg)],
            text=[fmt_currency(v, False) for v in seg["ASP"]],
            textposition="outside",
        ))
        fig_seg_asp.update_layout(**plotly_layout("Avg Selling Price by Segment", 320))
        chart_wrap(fig_seg_asp)

    # ── Correlation Heatmap ──────────────────────────────────────────────────
    section_header("Metric Correlation Analysis", "Identifying key relationships across retail performance indicators")

    numeric_cols = ["Revenue", "Units Sold", "Average Selling Price", "Sell Through Rate",
                    "Gross Margin %", "Return Rate", "Visual Merchandising Score",
                    "Inventory Units", "Stockout Flag"]
    corr = dff[numeric_cols].corr()

    fig_corr = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale=[[0, "#8B3A3A"], [0.5, "#F5F2EC"], [1, "#C9A84C"]],
        zmid=0, zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in corr.values],
        texttemplate="%{text}",
        textfont=dict(size=9),
        colorbar=dict(title=dict(text="r", side="right"), thickness=12),
    ))
    fig_corr.update_layout(**plotly_layout("", 400))
    chart_wrap(fig_corr)

    # ── Anomaly Detection ────────────────────────────────────────────────────
    section_header("Revenue Anomaly Detection", "Statistical identification of unusual performance patterns")

    weekly_anom = dff.set_index("Date").resample("W")["Revenue"].sum().reset_index()
    weekly_anom.columns = ["Week", "Revenue"]

    if len(weekly_anom) > 4:
        roll_mean = weekly_anom["Revenue"].rolling(4, min_periods=1).mean()
        roll_std  = weekly_anom["Revenue"].rolling(4, min_periods=1).std().fillna(0)
        weekly_anom["upper"] = roll_mean + 1.8 * roll_std
        weekly_anom["lower"] = (roll_mean - 1.8 * roll_std).clip(lower=0)
        weekly_anom["anomaly"] = (
            (weekly_anom["Revenue"] > weekly_anom["upper"]) |
            (weekly_anom["Revenue"] < weekly_anom["lower"])
        )

        fig_anom = go.Figure()
        fig_anom.add_trace(go.Scatter(
            x=pd.concat([weekly_anom["Week"], weekly_anom["Week"][::-1]]),
            y=pd.concat([weekly_anom["upper"], weekly_anom["lower"][::-1]]),
            fill="toself", fillcolor="rgba(201,168,76,0.08)",
            line=dict(color="rgba(0,0,0,0)"), name="Normal Band",
        ))
        fig_anom.add_trace(go.Scatter(
            x=weekly_anom["Week"], y=weekly_anom["Revenue"],
            mode="lines", name="Revenue", line=dict(color="#1C1C1C", width=1.5),
        ))
        anomalies = weekly_anom[weekly_anom["anomaly"]]
        fig_anom.add_trace(go.Scatter(
            x=anomalies["Week"], y=anomalies["Revenue"],
            mode="markers", name="Anomaly",
            marker=dict(color="#8B3A3A", size=9, symbol="x"),
        ))
        fig_anom.update_layout(**plotly_layout("", 320))
        chart_wrap(fig_anom)

    # ── Market Opportunity Scoring ───────────────────────────────────────────
    section_header("Market Opportunity Index", "Composite score ranking markets by growth potential")

    mkt_opp = dff.groupby("Market").agg(
        Revenue=("Revenue", "sum"),
        SellThrough=("Sell Through Rate", "mean"),
        GrossMargin=("Gross Margin %", "mean"),
        StockoutRate=("Stockout Flag", "mean"),
    ).reset_index()

    for col in ["Revenue", "SellThrough", "GrossMargin"]:
        mn, mx = mkt_opp[col].min(), mkt_opp[col].max()
        mkt_opp[f"{col}_n"] = (mkt_opp[col] - mn) / (mx - mn + 1e-9)

    mkt_opp["OpportunityIndex"] = (
        mkt_opp["Revenue_n"] * 0.40 +
        mkt_opp["SellThrough_n"] * 0.30 +
        mkt_opp["GrossMargin_n"] * 0.20 +
        mkt_opp["StockoutRate"] * 0.10
    ) * 100

    mkt_opp = mkt_opp.sort_values("OpportunityIndex", ascending=True)

    fig_mkt_opp = go.Figure(go.Bar(
        x=mkt_opp["OpportunityIndex"], y=mkt_opp["Market"],
        orientation="h",
        marker=dict(color=mkt_opp["OpportunityIndex"],
                    colorscale=[[0, "#E8D5A3"], [1, "#C9A84C"]]),
        text=[f"{v:.1f}" for v in mkt_opp["OpportunityIndex"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Opportunity Index: %{x:.1f}<extra></extra>",
    ))
    fig_mkt_opp.update_layout(**plotly_layout("", 340))
    chart_wrap(fig_mkt_opp)

    # ── Growth Trend ────────────────────────────────────────────────────────
    section_header("Revenue Growth Trajectory", "Month-over-month performance vs. prior period")

    monthly = dff.set_index("Date").resample("ME")["Revenue"].sum().reset_index()
    monthly.columns = ["Month", "Revenue"]
    monthly["MoM"] = monthly["Revenue"].pct_change() * 100

    if len(monthly) > 1:
        fig_mom = make_subplots(specs=[[{"secondary_y": True}]])
        fig_mom.add_trace(go.Bar(
            x=monthly["Month"], y=monthly["Revenue"],
            name="Revenue", marker_color="#C9A84C", opacity=0.7,
        ), secondary_y=False)
        fig_mom.add_trace(go.Scatter(
            x=monthly["Month"], y=monthly["MoM"],
            name="MoM Growth %", mode="lines+markers",
            line=dict(color="#2C3E50", width=2),
            marker=dict(size=6),
        ), secondary_y=True)
        fig_mom.update_layout(**plotly_layout("", 340))
        fig_mom.update_yaxes(title_text="Revenue (€)", secondary_y=False, showgrid=True, gridcolor="#E8E4DC")
        fig_mom.update_yaxes(title_text="MoM Growth (%)", secondary_y=True, showgrid=False, zeroline=True, zerolinecolor="#E8E4DC")
        chart_wrap(fig_mom)

    # ── Strategic Recommendations ────────────────────────────────────────────
    section_header("Strategic Recommendations", "AI-generated business priorities based on current analytics")

    st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)
    recs = []

    top_opp_market = mkt_opp.sort_values("OpportunityIndex", ascending=False).iloc[0]["Market"]
    recs.append(f"**Prioritise {top_opp_market}** for next-season investment: the market ranks highest on the composite opportunity index, combining strong sell-through, margin performance, and demand signals.")

    highest_str_coll = dff.groupby("Collection Type")["Sell Through Rate"].mean().idxmax()
    recs.append(f"**Expand {highest_str_coll} collection depth**: sell-through rates confirm strong demand alignment. Recommend increasing buy quantities by 10–15% for core replenishment SKUs.")

    ec_ret = dff[dff["Channel"] == "E-commerce"]["Return Rate"].mean() if "E-commerce" in dff["Channel"].values else 0
    if ec_ret > 0.12:
        recs.append(f"**Address e-commerce return friction**: return rates at {fmt_pct(ec_ret)} exceed the luxury benchmark threshold. Recommended actions: enhanced product content, virtual try-on, and post-purchase clienteling touchpoints.")

    stockout_rate = dff["Stockout Flag"].mean()
    if stockout_rate > 0.05:
        recs.append(f"**Replenishment alert**: {fmt_pct(stockout_rate)} of transactions show stockout conditions. Immediate inventory review across high-velocity SKUs is advised to protect revenue opportunity in peak demand periods.")

    top_vm_impact = dff.groupby("Store Type")["Visual Merchandising Score"].mean().idxmin()
    recs.append(f"**Elevate visual merchandising in {top_vm_impact} stores**: VM scores in this format lag the fleet average. Investment in in-store presentation is expected to yield a measurable uplift in conversion and ASP.")

    for rec in recs:
        st.markdown(f'<div class="insight-box">✦ &nbsp;{rec}</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="luxury-divider">· · · · ·</div>
    <div style='text-align:center;font-size:0.7rem;color:#9B9184;letter-spacing:0.1em;'>
    MAISON ANALYTICS &nbsp;·&nbsp; SYNTHETIC PORTFOLIO PROJECT &nbsp;·&nbsp; ALL DATA IS ILLUSTRATIVE
    </div>
    """, unsafe_allow_html=True)
