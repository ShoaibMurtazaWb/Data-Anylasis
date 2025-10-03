# AnalysisFile.py (Short and Simple Final Version)
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# --- Config & Styling ---
st.set_page_config(page_title="Sales Dashboard", page_icon="📈", layout="wide")
st.title("📈 Financial Sales Dashboard")

px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = ["#2563eb", "#16a34a", "#dc2626", "#9333ea", "#ea580c", "#0891b2", "#f59e0b"]

def style_fig(fig, height=380, ytitle=None):
    fig.update_layout(height=height, margin=dict(l=20, r=20, t=60, b=40), font=dict(size=14, color='black'))
    if ytitle: fig.update_layout(yaxis_title=ytitle)
    return fig

# Simple CSS
st.markdown("""
<style>
h3 { padding: .35rem .6rem; background: rgba(37,99,235,.12); border-left: 3px solid #2563eb; border-radius: 6px; }
.stMetric { background: #fff; border: 1px solid rgba(0,0,0,.06); border-radius: 12px; padding: .6rem .8rem; color: black;}
section[data-testid="stSidebar"] { background: #f8fafc; }
.st-emotion-cache-1wivap2{color:black;}
</style>
""", unsafe_allow_html=True)

# --- Data Loading & Prep ---
DATA_PATH = Path(__file__).parent / "data" / "ecommerce_dataset.csv"

@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip().lower() for c in df.columns]
    
    if "order_date" in df.columns:
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

    for c in ("quantity", "price", "discount"):
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")

    if {"quantity", "price"}.issubset(df.columns):
        df["gross_sales"] = df["quantity"] * df["price"]
        df["net_sales"] = df["gross_sales"] * (1 - df.get("discount", 0).clip(0, 0.99))
    
    if "order_date" in df.columns:
        df["date"] = df["order_date"].dt.date
        df["weekday"] = df["order_date"].dt.day_name()
    
    valid_cols = [c for c in ["net_sales", "quantity"] if c in df.columns]
    return df.dropna(subset=valid_cols)

df = load_and_prepare_data()

# --- Sidebar Filters ---
with st.sidebar:
    st.header("🔧 Filters")
    st.caption("Default Dataset.")
    
    start, end = None, None
    if "order_date" in df.columns:
        min_d, max_d = df["order_date"].min(), df["order_date"].max()
        start, end = st.date_input("Date range", value=(min_d.date(), max_d.date()), min_value=min_d.date(), max_value=max_d.date())
    
    filter_cols = ["region", "category", "payment_method"]
    sel_filters = {}
    for col in filter_cols:
        if col in df.columns:
            options = sorted(df[col].dropna().unique().tolist())
            sel_filters[col] = st.multiselect(col.title(), options, default=options)

# Apply filters
fdf = df.copy()
if start and end and "order_date" in fdf.columns: 
    fdf = fdf[(fdf["order_date"].dt.date >= start) & (fdf["order_date"].dt.date <= end)]
for col, selections in sel_filters.items():
    if selections: fdf = fdf[fdf[col].isin(selections)]

if fdf.empty: st.warning("No data after filters. Adjust filters."); st.stop()

# --- KPIs ---
c1, c2, c3, c4 = st.columns(4)
total_net = fdf["net_sales"].sum()
orders = fdf["order_id"].nunique()
c1.metric("Net Sales", f"${total_net:,.0f}")
c2.metric("Gross Sales", f"${fdf['gross_sales'].sum():,.0f}")
c3.metric("Units Sold", f"{int(fdf['quantity'].sum()):,}")
c4.metric("Avg Order Value (AOV)", f"${total_net / orders:,.2f}" if orders else "$0.00")

# --- Chart Helper (Generic, Title Removed) ---
def plot_sales_chart(df, col, chart_type="bar", y_col="net_sales", height=380):
    st.subheader(f"{y_col.replace('_', ' ').title()} by {col.title()}")
    if col in df.columns:
        
        # Aggregation logic: required for time series and breakdowns
        if chart_type in ["line", "area"]:
            agg = df.groupby(col, as_index=False)[y_col].sum()
        else:
            agg = df.groupby(col, as_index=False)[y_col].sum().sort_values(y_col, ascending=False)
        
        # Plotting
        if chart_type == "bar":
            fig = px.bar(agg, x=col, y=y_col, text_auto=True, title=None)
        elif chart_type == "pie":
            fig = px.pie(agg, values=y_col, names=col, title=None, hole=0.35)
        elif chart_type == "line":
            fig = px.line(agg, x=col, y=y_col, markers=True, title=None)
            fig.update_layout(hovermode="x unified")
        elif chart_type == "area":
            fig = px.area(agg, x=col, y=y_col, title=None)
            fig.update_layout(hovermode="x unified")

        st.plotly_chart(style_fig(fig, height=height, ytitle=y_col.replace('_', ' ').title()), use_container_width=True)
    else: st.info(f"No '{col}' column found.")

# --- Full-Width Charts ---
plot_sales_chart(fdf, "category")
plot_sales_chart(fdf, "region")
plot_sales_chart(fdf, "payment_method", chart_type="pie", height=460)

# Heatmap
st.subheader("Heatmap: Weekday × Category")
if {"weekday","category","net_sales"}.issubset(fdf.columns):
    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    piv = fdf.pivot_table(index="weekday", columns="category", values="net_sales", aggfunc="sum", fill_value=0)
    piv = piv.reindex(order, fill_value=0) # Re-index for correct order
    fig = px.imshow(piv, aspect="auto", color_continuous_scale="Blues", labels=dict(color="Net Sales"), title=None)
    st.plotly_chart(style_fig(fig, height=420), use_container_width=True)
else: st.info("Need columns: weekday, category, net_sales.")

# Time Series (using the common helper)
plot_sales_chart(fdf, "date", chart_type="line", height=420, y_col="net_sales")
plot_sales_chart(fdf, "date", chart_type="area", height=360, y_col="quantity")

# --- Discount vs Net Sales (Bubble Scatter) ---
st.subheader("Discount vs Net Sales (by Category)")
needed = {"discount", "net_sales", "category", "order_id", "quantity"}
if needed.issubset(fdf.columns):
    agg = fdf.groupby("category", as_index=False).agg(
        avg_discount=("discount", "mean"), net_sales=("net_sales", "sum"), orders=("order_id", "nunique"), units=("quantity", "sum")
    ).dropna(subset=["avg_discount", "net_sales"])
    agg = agg[(agg["avg_discount"] >= 0) & (agg["net_sales"] > 0)]

    if not agg.empty:
        x, y = agg["avg_discount"].to_numpy(), agg["net_sales"].to_numpy()
        a, b = np.polyfit(x, y, deg=1)
        r2 = 1 - np.sum((y - (a * x + b)) ** 2) / np.sum((y - np.mean(y)) ** 2)

        log_scale = st.toggle("Log scale (Y)", value=False, help="Use for large sales differences.")
        
        fig = px.scatter(agg, x="avg_discount", y="net_sales", size="orders", color="category",
            hover_data={"avg_discount":":.2%", "net_sales":":,.0f", "orders":":,", "units":":," , "category":True}, title=None)
        
        # Trendline and quadrants
        xs = np.linspace(x.min(), x.max(), 100)
        fig.add_scatter(x=xs, y=a * xs + b, mode="lines", name=f"Trend (R²={r2:.2f})")
        x_med, y_med = np.median(x), np.median(y)
        fig.add_vline(x=x_med, line_dash="dot", opacity=0.35); fig.add_hline(y=y_med, line_dash="dot", opacity=0.35)
        
        fig.update_layout(title=None, yaxis_title="Net Sales", xaxis_tickformat=".0%", height=420)
        if log_scale: fig.update_yaxes(type="log", tickformat="~s")
        st.plotly_chart(fig, use_container_width=True)

        # Efficiency Takeaways
        agg["sales_per_discount"] = agg["net_sales"] / np.maximum(agg["avg_discount"], 1e-6)
        top3 = agg.sort_values("sales_per_discount", ascending=False).head(3)
        bottom3 = agg.sort_values("sales_per_discount", ascending=True).head(3)

        col_a, col_b = st.columns(2)
        format_func = lambda x: f"{x:,.0f}"
        
        with col_a:
            st.markdown("**Top Efficiency**")
            top3['sales_per_discount'] = top3['sales_per_discount'].apply(format_func)
            st.dataframe(top3[["category","sales_per_discount"]].rename(columns={'sales_per_discount': 'Sales/Discount'}), hide_index=True, use_container_width=True)
        with col_b:
            st.markdown("**Bottom Efficiency**")
            bottom3['sales_per_discount'] = bottom3['sales_per_discount'].apply(format_func)
            st.dataframe(bottom3[["category","sales_per_discount"]].rename(columns={'sales_per_discount': 'Sales/Discount'}), hide_index=True, use_container_width=True)
    else: st.info("Insufficient data for scatter plot after cleaning.")
else: st.info(f"Need columns: {', '.join(needed)}.")

# --- Detail Table + Download ---
st.subheader("Detailed Transactions")
show_cols = [c for c in ["order_id", "order_date", "category", "region", "quantity", "price", "discount", "gross_sales", "net_sales"] if c in fdf.columns]
st.dataframe(fdf[show_cols].sort_values("order_date", na_position="last"), use_container_width=True, hide_index=True)
st.download_button("⬇️ Download filtered data (CSV)", data=fdf[show_cols].to_csv(index=False).encode("utf-8"), file_name="filtered_sales.csv", mime="text/csv")
st.caption("Discounts are treated as fractions (e.g., 0.10).")
