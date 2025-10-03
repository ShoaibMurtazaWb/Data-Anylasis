# AnalysisFile.py (Simplified)
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# --- Configuration & Styling ---
st.set_page_config(page_title="Financial Sales Dashboard", page_icon="📈", layout="wide")
st.title("📈 Financial Sales Dashboard")

px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = [
    "#2563eb", "#16a34a", "#dc2626", "#9333ea", "#ea580c", "#0891b2", "#f59e0b"
]

def style_fig(fig, *, height=380, ytitle=None):
    """Apply consistent layout to Plotly figures."""
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=60, b=40),
        hoverlabel=dict(bgcolor="white"),
        font=dict(size=14, color='black'),
    )
    if ytitle:
        fig.update_layout(yaxis_title=ytitle)
    return fig

# Subtle CSS polish (Simplified)
STYLES = """
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 4rem; }
h2, h3 { margin-top: 0.75rem; }
.stMarkdown h3 {
  padding: .35rem .6rem;
  background: linear-gradient(90deg, rgba(37,99,235,.12), rgba(37,99,235,0));
  border-left: 3px solid #2563eb;
  border-radius: 6px;
}
.stMetric {
  color: #000000;
  background: #fff; border: 1px solid rgba(0,0,0,.06); border-radius: 12px;
  padding: .6rem .8rem; box-shadow: 0 1px 2px rgba(0,0,0,.05);
}
.st-emotion-cache-q49buc,.st-emotion-cache-efbu8t{color: #000000}
section[data-testid="stSidebar"] { background: #f8fafc; border-right: 1px solid #e5e7eb; }
.stDownloadButton button { border-radius: 10px; padding: .6rem 1rem; border: 1px solid #e5e7eb; }
div[data-testid="stDataFrame"] { height: max-content; }
</style>
"""
st.markdown(STYLES, unsafe_allow_html=True)

# --- Data Loading & Coercion (Kept mostly as is for correctness) ---
EXPECTED_COLS = [
    "order_id", "customer_id", "product_id", "category", "quantity",
    "price", "discount", "order_date", "region", "payment_method"
]
DATA_PATH = Path(__file__).parent / "data" / "ecommerce_dataset.csv"

@st.cache_data
def load_default() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)

def coerce_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Missing column check omitted for brevity in simplified version,
    # assuming default data is clean or uploader is not used.
    
    if "order_date" in df.columns:
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    for c in ("quantity", "price", "discount"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def add_computed(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if {"quantity", "price"}.issubset(out.columns):
        out["gross_sales"] = out["quantity"] * out["price"]
        out["net_sales"] = out["gross_sales"] * (1 - out.get("discount", 0).clip(0, 0.99))
    else:
        out["gross_sales"], out["net_sales"] = np.nan, np.nan

    if "order_date" in out.columns and pd.api.types.is_datetime64_any_dtype(out["order_date"]):
        out["date"] = out["order_date"].dt.date
        out["month"] = out["order_date"].dt.to_period("M").dt.to_timestamp()
        out["weekday"] = out["order_date"].dt.day_name()
    return out

# --- Load and Prepare Data ---
df = load_default()
df = coerce_schema(df)
df = add_computed(df)
valid_sales_cols = [c for c in ["net_sales", "gross_sales", "quantity"] if c in df.columns]
df = df.dropna(subset=valid_sales_cols)

# --- Sidebar Filters ---
with st.sidebar:
    st.header("🔧 Filters")
    st.caption("Using default dataset: **data/ecommerce_dataset.csv**")
    
    start, end = None, None
    if "order_date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["order_date"]):
        min_d, max_d = df["order_date"].min(), df["order_date"].max()
        start, end = st.date_input(
            "Date range",
            value=(min_d.date(), max_d.date()),
            min_value=min_d.date(),
            max_value=max_d.date(),
        )

    # Use dict comprehension for cleaner filter setup
    filter_cols = {"region": "Region", "category": "Category", "payment_method": "Payment Method"}
    sel_filters = {}
    for col, label in filter_cols.items():
        if col in df.columns:
            options = sorted(df[col].dropna().unique().tolist())
            sel_filters[col] = st.multiselect(label, options, default=options)
        else:
            sel_filters[col] = None

# Apply filters
fdf = df.copy()
if start and end and "order_date" in fdf.columns:
    fdf = fdf[(fdf["order_date"].dt.date >= start) & (fdf["order_date"].dt.date <= end)]
for col, selections in sel_filters.items():
    if selections and col in fdf.columns:
        fdf = fdf[fdf[col].isin(selections)]

if fdf.empty:
    st.warning("No data after filters. Adjust filters.")
    st.stop()

# --- KPI Row ---
total_net   = fdf["net_sales"].sum()
total_gross = fdf["gross_sales"].sum()
units       = fdf["quantity"].sum()
orders      = fdf["order_id"].nunique() if "order_id" in fdf.columns else len(fdf)
aov         = total_net / orders if orders else 0.0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Net Sales", f"${total_net:,.0f}")
col2.metric("Gross Sales", f"${total_gross:,.0f}")
col3.metric("Units Sold", f"{int(units):,}")
col4.metric("Avg Order Value (AOV)", f"${aov:,.2f}")

# --- Chart Generation Helper ---
def plot_sales_breakdown(df, col, chart_type="bar", title_suffix=""):
    st.subheader(f"Sales by {col.replace('_', ' ').title()}{title_suffix}")
    if col in df.columns:
        agg = (
            df.groupby(col, as_index=False)["net_sales"]
            .sum()
            .sort_values("net_sales", ascending=False)
        )
        if chart_type == "bar":
            fig = px.bar(agg, x=col, y="net_sales", text_auto=True, 
                         title=f"Net Sales by {col.title()}")
            fig = style_fig(fig, height=380, ytitle="Net Sales")
        elif chart_type == "pie":
            fig = px.pie(agg, values="net_sales", names=col, 
                         title=f"Net Sales by {col.title()}", hole=0.35)
            fig = style_fig(fig, height=460)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"No '{col}' column found.")

# --- Full-Width Charts ---
plot_sales_breakdown(fdf, "category")
plot_sales_breakdown(fdf, "region")
plot_sales_breakdown(fdf, "payment_method", chart_type="pie", title_suffix=" Mix")

# --- Heatmap: Weekday x Category ---
st.subheader("Heatmap: Weekday × Category")
needed = {"weekday","category","net_sales"}
if needed.issubset(fdf.columns):
    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    temp = fdf.assign(weekday=lambda d: pd.Categorical(d["weekday"], categories=order, ordered=True))
    piv = temp.pivot_table(index="weekday", columns="category", values="net_sales",
                           aggfunc="sum", fill_value=0)
    fig = px.imshow(piv, aspect="auto", color_continuous_scale="Blues",
                    labels=dict(color="Net Sales"), title="Net Sales by Weekday and Category")
    fig = style_fig(fig, height=420)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info(f"Need columns: {', '.join(needed)}.")

# --- Time Series Charts ---
def plot_time_series(df, y_col, title):
    st.subheader(title)
    if "date" in df.columns and df["date"].notna().any():
        ts = df.groupby("date", as_index=False)[y_col].sum()
        if y_col == "net_sales":
            fig = px.line(ts, x="date", y=y_col, markers=True, title=title)
            fig = style_fig(fig, height=420, ytitle="Net Sales")
        else: # quantity
            fig = px.area(ts, x="date", y=y_col, title=title)
            fig = style_fig(fig, height=360, ytitle="Units")
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"No valid dates to plot for {y_col}.")

plot_time_series(fdf, "net_sales", "Net Sales Over Time")
plot_time_series(fdf, "quantity", "Units Sold Over Time")

# --- Discount vs Net Sales (Bubble Scatter) ---
st.subheader("Discount vs Net Sales (by Category)")
needed = {"discount", "net_sales", "category", "order_id", "quantity"}
if needed.issubset(fdf.columns):
    agg = (
        fdf.groupby("category", as_index=False)
          .agg(avg_discount=("discount", "mean"),
               net_sales=("net_sales", "sum"),
               orders=("order_id", "nunique"),
               units=("quantity", "sum"))
    )
    agg = agg.dropna(subset=["avg_discount", "net_sales"])
    agg = agg[(agg["avg_discount"] >= 0) & (agg["net_sales"] > 0)]

    if not agg.empty:
        # Regression
        x, y = agg["avg_discount"].to_numpy(), agg["net_sales"].to_numpy()
        a, b = np.polyfit(x, y, deg=1)
        r2 = 1 - np.sum((y - (a * x + b)) ** 2) / np.sum((y - np.mean(y)) ** 2)

        log_scale = st.toggle("Log scale (Y)", value=False, help="Good when categories have very different sales sizes.")
        
        # Plot
        fig = px.scatter(
            agg, x="avg_discount", y="net_sales", size="orders", color="category",
            hover_data={"avg_discount":":.2%", "net_sales":":,.0f", "orders":":,", "units":":," , "category":True},
            labels={"avg_discount":"Average Discount", "net_sales":"Net Sales"},
        )
        fig.update_traces(marker=dict(opacity=0.85, line=dict(width=0.5)))

        # Regression line, Quadrant guides, Quadrant shading
        xs = np.linspace(float(x.min()), float(x.max()), 100)
        fig.add_scatter(x=xs, y=a * xs + b, mode="lines", name=f"Trend (R²={r2:.2f})")
        x_med, y_med = float(np.median(x)), float(np.median(y))
        fig.add_vline(x=x_med, line_dash="dot", opacity=0.35)
        fig.add_hline(y=y_med, line_dash="dot", opacity=0.35)
        fig.add_vrect(x0=x.min(), x1=x_med, fillcolor="LightGreen", opacity=0.05, line_width=0)
        fig.add_vrect(x0=x_med, x1=x.max(), fillcolor="LightCoral", opacity=0.05, line_width=0)
        fig.add_hrect(y0=y_med, y1=y.max(), fillcolor="LightSkyBlue", opacity=0.05, line_width=0)

        fig.update_layout(
            title=f"Discount Intensity vs Net Sales — R²={r2:.2f}",
            yaxis_title="Net Sales",
            xaxis_tickformat=".0%",
            height=420,
            legend_title="Category",
            hovermode="closest",
        )
        if log_scale:
            fig.update_yaxes(type="log", tickformat="~s")

        st.plotly_chart(fig, use_container_width=True)

        # Takeaways
        agg["sales_per_discount_point"] = agg["net_sales"] / np.maximum(agg["avg_discount"], 1e-6)
        top3 = agg.sort_values("sales_per_discount_point", ascending=False).head(3)
        bottom3 = agg.sort_values("sales_per_discount_point", ascending=True).head(3)

        col_a, col_b = st.columns(2)
        format_func = lambda d: d["sales_per_discount_point"].map("{:,.0f}".format)
        with col_a:
            st.markdown("**Top efficiency**")
            st.dataframe(top3[["category","sales_per_discount_point"]].assign(sales_per_discount_point=format_func),
                         hide_index=True, use_container_width=True)
        with col_b:
            st.markdown("**Bottom efficiency**")
            st.dataframe(bottom3[["category","sales_per_discount_point"]].assign(sales_per_discount_point=format_func),
                         hide_index=True, use_container_width=True)
    else:
        st.info("Insufficient data for scatter plot after cleaning.")
else:
    st.info(f"Need columns: {', '.join(needed)}.")

# --- Detail Table + Download ---
st.subheader("Detailed Transactions")
show_cols = [
    c for c in [
        "order_id", "order_date", "customer_id", "product_id", "category",
        "region", "payment_method", "quantity", "price", "discount",
        "gross_sales", "net_sales"
    ] if c in fdf.columns
]
st.dataframe(
    fdf[show_cols].sort_values("order_date", na_position="last"),
    use_container_width=True, hide_index=True
)

csv = fdf[show_cols].to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download filtered data (CSV)", data=csv, file_name="filtered_sales.csv", mime="text/csv")

st.caption("Note: Discounts are treated as fractions (e.g., 0.10 = 10%).")
