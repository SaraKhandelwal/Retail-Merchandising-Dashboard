"""
helper_functions.py
Utility functions for the Luxury Retail Merchandising Analytics Dashboard.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# ─────────────────────────────────────────────
#  SYNTHETIC DATA GENERATION
# ─────────────────────────────────────────────

def generate_synthetic_data(n_rows: int = 2500, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic synthetic luxury retail dataset."""
    rng = np.random.default_rng(seed)

    # ── Dimension pools ──────────────────────────────────────────────────────
    regions = ["Europe", "North America", "Asia-Pacific", "Middle East"]
    region_market_map = {
        "Europe":        ["France", "Italy", "UK"],
        "North America": ["USA"],
        "Asia-Pacific":  ["China", "Japan", "South Korea"],
        "Middle East":   ["UAE"],
    }
    store_types     = ["Flagship", "Boutique", "Department Store", "E-commerce"]
    categories      = ["Women's Shoes", "Men's Shoes", "Leather Goods", "Accessories", "Ready-to-Wear"]
    collection_types= ["Core", "Seasonal", "Limited Edition", "Runway"]
    product_families= {
        "Women's Shoes":  ["Pump", "Sandal", "Boot", "Sneaker"],
        "Men's Shoes":    ["Oxford", "Loafer", "Derby", "Sneaker"],
        "Leather Goods":  ["Handbag", "Wallet", "Tote", "Clutch"],
        "Accessories":    ["Silk Scarf", "Belt", "Sunglasses", "Jewellery"],
        "Ready-to-Wear":  ["Blazer", "Dress", "Coat", "Knitwear"],
    }
    customer_segs   = ["VIC", "High Value", "Returning Client", "New Client"]
    channels        = ["Retail", "E-commerce", "Clienteling"]

    # ── Realistic weights ────────────────────────────────────────────────────
    region_weights       = [0.35, 0.25, 0.30, 0.10]
    store_type_weights   = [0.25, 0.35, 0.25, 0.15]
    collection_weights   = [0.40, 0.30, 0.20, 0.10]
    customer_seg_weights = [0.10, 0.20, 0.40, 0.30]
    channel_weights      = [0.55, 0.30, 0.15]

    # ── Date range ───────────────────────────────────────────────────────────
    start_date = datetime(2023, 1, 1)
    end_date   = datetime(2024, 12, 31)
    date_range = (end_date - start_date).days

    rows = []
    sku_counter = 1000

    for _ in range(n_rows):
        region       = rng.choice(regions, p=region_weight_norm(region_weights))
        market       = rng.choice(region_market_map[region])
        store_type   = rng.choice(store_types, p=store_type_weights)
        category     = rng.choice(categories)
        collection   = rng.choice(collection_types, p=collection_weights)
        family       = rng.choice(product_families[category])
        customer_seg = rng.choice(customer_segs, p=customer_seg_weights)
        channel_val  = rng.choice(channels, p=channel_weights)
        date         = start_date + timedelta(days=int(rng.integers(0, date_range)))

        # ── Base ASP with realistic modifiers ────────────────────────────────
        base_asp = {
            "Women's Shoes":  950, "Men's Shoes": 850,
            "Leather Goods": 1800, "Accessories": 650, "Ready-to-Wear": 2200,
        }[category]

        asp = base_asp
        if collection == "Limited Edition": asp *= rng.uniform(1.30, 1.60)
        elif collection == "Runway":        asp *= rng.uniform(1.20, 1.45)
        elif collection == "Seasonal":      asp *= rng.uniform(0.90, 1.15)

        if store_type == "Flagship":        asp *= rng.uniform(1.08, 1.18)
        elif store_type == "E-commerce":    asp *= rng.uniform(0.92, 1.00)

        if customer_seg == "VIC":           asp *= rng.uniform(1.15, 1.30)

        if market in ["France", "Italy"]:   asp *= rng.uniform(1.05, 1.12)
        elif market in ["China", "UAE"]:    asp *= rng.uniform(1.02, 1.10)

        asp = round(asp * rng.uniform(0.96, 1.04), 2)

        # ── Units sold ───────────────────────────────────────────────────────
        base_units = int(rng.integers(2, 30))
        if collection == "Limited Edition": base_units = max(1, int(base_units * rng.uniform(0.5, 0.8)))
        if customer_seg == "VIC":           base_units = max(1, int(base_units * rng.uniform(0.6, 0.9)))
        if store_type == "Flagship":        base_units = int(base_units * rng.uniform(1.1, 1.5))
        units_sold = max(1, base_units)

        # ── Seasonal demand boost ─────────────────────────────────────────────
        month = date.month
        seasonal_boost = 1.0
        if month in [11, 12]:   seasonal_boost = 1.25   # Holiday
        elif month in [3, 4]:   seasonal_boost = 1.10   # Spring
        elif month in [6, 7]:   seasonal_boost = 0.90   # Summer lull
        units_sold = max(1, int(units_sold * seasonal_boost))

        revenue = round(units_sold * asp, 2)

        # VIC revenue uplift
        if customer_seg == "VIC":
            revenue = round(revenue * rng.uniform(1.20, 1.50), 2)

        # ── Sell-through rate ────────────────────────────────────────────────
        str_base = 0.72
        if collection == "Limited Edition": str_base = rng.uniform(0.85, 0.97)
        elif collection == "Runway":        str_base = rng.uniform(0.55, 0.75)
        elif collection == "Core":          str_base = rng.uniform(0.65, 0.82)
        else:                               str_base = rng.uniform(0.58, 0.80)
        sell_through = round(min(0.99, max(0.20, float(str_base) + rng.uniform(-0.05, 0.05))), 4)

        # ── Inventory units ──────────────────────────────────────────────────
        inventory = max(0, int(units_sold / sell_through) - units_sold)

        # ── Gross margin ─────────────────────────────────────────────────────
        gm_base = {"Women's Shoes": 0.62, "Men's Shoes": 0.60,
                   "Leather Goods": 0.68, "Accessories": 0.72, "Ready-to-Wear": 0.58}[category]
        if collection == "Limited Edition": gm_base += 0.04
        if store_type == "E-commerce":      gm_base -= 0.03
        gross_margin = round(min(0.85, max(0.35, gm_base + rng.uniform(-0.03, 0.03))), 4)

        # ── Return rate ──────────────────────────────────────────────────────
        return_base = 0.06
        if channel_val == "E-commerce":     return_base = rng.uniform(0.14, 0.22)
        elif store_type == "E-commerce":    return_base = rng.uniform(0.12, 0.20)
        if customer_seg == "VIC":           return_base *= 0.5
        return_rate = round(min(0.35, max(0.00, return_base + rng.uniform(-0.02, 0.02))), 4)

        # ── Visual merchandising score (1-10) ────────────────────────────────
        vm_score = round(rng.uniform(5.0, 10.0), 1)
        if store_type == "Flagship":        vm_score = round(min(10.0, vm_score + rng.uniform(0.5, 1.5)), 1)
        elif store_type == "E-commerce":    vm_score = round(max(1.0, vm_score - rng.uniform(1.0, 2.0)), 1)

        # VM score positively correlates with revenue
        revenue = round(revenue * (1 + (vm_score - 7.0) * 0.015), 2)

        # ── Flags ────────────────────────────────────────────────────────────
        stockout_flag   = 1 if inventory == 0 and sell_through > 0.92 else 0
        promotion_flag  = 1 if (collection == "Seasonal" and month in [6, 7, 12]) or rng.random() < 0.12 else 0

        # Stockouts suppress revenue
        if stockout_flag:
            revenue = round(revenue * rng.uniform(0.75, 0.90), 2)

        sku = f"LX-{category[:2].upper()}-{sku_counter:05d}"
        sku_counter += 1

        rows.append({
            "Date":                    date.strftime("%Y-%m-%d"),
            "Region":                  region,
            "Market":                  market,
            "Store Type":              store_type,
            "Product Category":        category,
            "Collection Type":         collection,
            "Product Family":          family,
            "SKU":                     sku,
            "Units Sold":              units_sold,
            "Revenue":                 revenue,
            "Average Selling Price":   round(asp, 2),
            "Inventory Units":         inventory,
            "Sell Through Rate":       sell_through,
            "Gross Margin %":          gross_margin,
            "Return Rate":             return_rate,
            "Customer Segment":        customer_seg,
            "Channel":                 channel_val,
            "Visual Merchandising Score": vm_score,
            "Stockout Flag":           stockout_flag,
            "Promotion Flag":          promotion_flag,
        })

    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def region_weight_norm(weights):
    a = np.array(weights, dtype=float)
    return a / a.sum()


# ─────────────────────────────────────────────
#  FORMATTING HELPERS
# ─────────────────────────────────────────────

def fmt_currency(value: float, abbrev: bool = True) -> str:
    if abbrev:
        if value >= 1_000_000:
            return f"€{value/1_000_000:.1f}M"
        elif value >= 1_000:
            return f"€{value/1_000:.1f}K"
    return f"€{value:,.0f}"


def fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def fmt_number(value: float) -> str:
    if value >= 1_000_000:
        return f"{value/1_000_000:.1f}M"
    elif value >= 1_000:
        return f"{value/1_000:.1f}K"
    return f"{value:,.0f}"


# ─────────────────────────────────────────────
#  FILTER HELPER
# ─────────────────────────────────────────────

def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    dff = df.copy()
    if filters.get("regions"):
        dff = dff[dff["Region"].isin(filters["regions"])]
    if filters.get("markets"):
        dff = dff[dff["Market"].isin(filters["markets"])]
    if filters.get("categories"):
        dff = dff[dff["Product Category"].isin(filters["categories"])]
    if filters.get("collections"):
        dff = dff[dff["Collection Type"].isin(filters["collections"])]
    if filters.get("channels"):
        dff = dff[dff["Channel"].isin(filters["channels"])]
    if filters.get("date_range"):
        s, e = filters["date_range"]
        dff = dff[(dff["Date"] >= pd.Timestamp(s)) & (dff["Date"] <= pd.Timestamp(e))]
    return dff


# ─────────────────────────────────────────────
#  INSIGHT GENERATION
# ─────────────────────────────────────────────

def generate_insights(df: pd.DataFrame) -> list[str]:
    insights = []
    if df.empty:
        return ["No data available for the selected filters."]

    # Top region
    top_region = df.groupby("Region")["Revenue"].sum().idxmax()
    insights.append(f"**{top_region}** leads revenue performance across all selected markets.")

    # Best collection sell-through
    coll_st = df.groupby("Collection Type")["Sell Through Rate"].mean()
    best_coll = coll_st.idxmax()
    insights.append(
        f"**{best_coll}** collections achieve the highest average sell-through "
        f"at {fmt_pct(coll_st[best_coll])}, reflecting strong demand alignment."
    )

    # E-commerce return rates
    if "E-commerce" in df["Channel"].values:
        ec_ret = df[df["Channel"] == "E-commerce"]["Return Rate"].mean()
        rt_ret = df[df["Channel"] == "Retail"]["Return Rate"].mean() if "Retail" in df["Channel"].values else 0
        if ec_ret > rt_ret:
            insights.append(
                f"E-commerce return rates ({fmt_pct(ec_ret)}) remain elevated vs. "
                f"retail ({fmt_pct(rt_ret)}), warranting focused post-purchase strategy."
            )

    # VIC revenue contribution
    if "VIC" in df["Customer Segment"].values:
        vic_rev = df[df["Customer Segment"] == "VIC"]["Revenue"].sum()
        total_rev = df["Revenue"].sum()
        insights.append(
            f"VIC clients account for **{fmt_pct(vic_rev/total_rev)}** of total revenue, "
            f"underscoring the critical importance of clienteling excellence."
        )

    # Top category
    top_cat = df.groupby("Product Category")["Revenue"].sum().idxmax()
    insights.append(f"**{top_cat}** is the top-performing category by revenue in the current view.")

    # Stockout risk
    stockout_pct = df["Stockout Flag"].mean()
    if stockout_pct > 0.05:
        insights.append(
            f"{fmt_pct(stockout_pct)} of transactions show stockout conditions — "
            f"immediate replenishment is advised to protect revenue opportunity."
        )

    # VM correlation note
    high_vm = df[df["Visual Merchandising Score"] >= 8.0]["Revenue"].mean()
    low_vm  = df[df["Visual Merchandising Score"] < 6.0]["Revenue"].mean()
    if high_vm > low_vm * 1.05:
        insights.append(
            f"Stores with high visual merchandising scores (≥8) generate "
            f"{fmt_pct((high_vm - low_vm) / low_vm)} more revenue per transaction on average."
        )

    return insights[:5]


# ─────────────────────────────────────────────
#  INVENTORY HEALTH CLASSIFICATION
# ─────────────────────────────────────────────

def classify_inventory_health(row) -> str:
    st_rate = row["Sell Through Rate"]
    inv     = row["Inventory Units"]
    stockout= row["Stockout Flag"]
    if stockout == 1:
        return "Stockout Risk"
    if st_rate >= 0.90 and inv < 5:
        return "Critical Low"
    if st_rate >= 0.75:
        return "Healthy"
    if st_rate < 0.50 and inv > 20:
        return "Overstock"
    return "Monitor"


# ─────────────────────────────────────────────
#  OPPORTUNITY SCORING
# ─────────────────────────────────────────────

def compute_opportunity_score(df: pd.DataFrame) -> pd.DataFrame:
    """Score product families by unrealised revenue opportunity."""
    agg = df.groupby("Product Family").agg(
        revenue=("Revenue", "sum"),
        sell_through=("Sell Through Rate", "mean"),
        gross_margin=("Gross Margin %", "mean"),
        stockout_rate=("Stockout Flag", "mean"),
        return_rate=("Return Rate", "mean"),
    ).reset_index()

    # Normalise
    for col in ["revenue", "sell_through", "gross_margin"]:
        mn, mx = agg[col].min(), agg[col].max()
        agg[f"{col}_norm"] = (agg[col] - mn) / (mx - mn + 1e-9)

    agg["opportunity_score"] = (
        agg["revenue_norm"]       * 0.35 +
        agg["sell_through_norm"]  * 0.30 +
        agg["gross_margin_norm"]  * 0.20 +
        agg["stockout_rate"]      * 0.10 +
        (1 - agg["return_rate"])  * 0.05
    ) * 100

    agg["opportunity_score"] = agg["opportunity_score"].round(1)
    return agg.sort_values("opportunity_score", ascending=False)


# ─────────────────────────────────────────────
#  PLOTLY THEME
# ─────────────────────────────────────────────

LUXURY_COLORS = {
    "cream":       "#F5F2EC",
    "charcoal":    "#1C1C1C",
    "gold":        "#C9A84C",
    "gold_light":  "#E8D5A3",
    "taupe":       "#8B7D6B",
    "sage":        "#7A8C7E",
    "blush":       "#C4A99A",
    "midnight":    "#2C3E50",
    "warm_grey":   "#9B9184",
    "ivory":       "#FDFAF5",
}

PALETTE = [
    "#C9A84C", "#8B7D6B", "#7A8C7E", "#C4A99A",
    "#2C3E50", "#9B9184", "#D4C5B0", "#4A6741",
]

def plotly_layout(title: str = "", height: int = 380) -> dict:
    return dict(
        title=dict(text=title, font=dict(family="Cormorant Garamond, Georgia, serif", size=16, color="#1C1C1C")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Cormorant Garamond, Georgia, serif", color="#1C1C1C", size=12),
        height=height,
        margin=dict(l=30, r=30, t=50, b=30),
        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=11)),
        yaxis=dict(showgrid=True, gridcolor="#E8E4DC", zeroline=False, tickfont=dict(size=11)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        colorway=PALETTE,
    )
