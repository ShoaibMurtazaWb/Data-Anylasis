# AnalysisFile.py
from pathlib import Path
import io
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Financial Sales Dashboard", page_icon="📈", layout="wide")
st.title("📈 Financial Sales Dashboard")

# ------------------------------------------
# Styling:
# ------------------------------------------
# === THEME & STYLING ===
# Plotly theme (consistent across all charts)
# === THEME & STYLING (fixed) ===
import plotly.express as px

# Global template + discrete colors for categorical series
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

# Subtle CSS polish
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


# ------------------------------------------
# Expected schema & helpers
# ------------------------------------------
EXPECTED_COLS = [
    "order_id", "customer_id", "product_id", "category", "quantity",
    "price", "discount", "order_date", "region", "payment_method"
]

DATA_PATH = Path(__file__).parent / "data" / "ecommerce_dataset.csv"

@st.cache_data
def load_default() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)

def coerce_schema(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: 
        return df
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        st.warning(f"Missing columns: {missing}. Proceeding with available columns.")

    # Parse dates
    if "order_date" in df.columns:
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

    # Numerics
    for c in ("quantity", "price", "discount"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

def add_computed(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if {"quantity", "price"}.issubset(out.columns):
        out["gross_sales"] = out["quantity"] * out["price"]
    else:
        out["gross_sales"] = np.nan

    if "discount" in out.columns:
        # discount is a fraction (0.10 = 10%)
        out["net_sales"] = out["gross_sales"] * (1 - out["discount"].clip(0, 0.99))
    else:
        out["net_sales"] = out["gross_sales"]

    if "order_date" in out.columns and pd.api.types.is_datetime64_any_dtype(out["order_date"]):
        out["date"] = out["order_date"].dt.date
        out["month"] = out["order_date"].dt.to_period("M").dt.to_timestamp()
        out["weekday"] = out["order_date"].dt.day_name()
    else:
        out["date"] = pd.NaT
        out["month"] = pd.NaT
        out["weekday"] = np.nan
    return out

# ------------------------------------------
# Load default dataset (no uploader)
# ------------------------------------------
df = load_default()
df = coerce_schema(df)
df = add_computed(df)
df = df.dropna(subset=[c for c in ["net_sales", "gross_sales", "quantity"] if c in df.columns])

# ------------------------------------------
# Sidebar filters only
# ------------------------------------------
with st.sidebar:
    st.header("🔧 Filters")
    st.caption("Using default dataset: **data/ecommerce_dataset.csv**")
    st.caption("Expected columns: " + ", ".join(EXPECTED_COLS))

    if "order_date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["order_date"]):
        min_d, max_d = df["order_date"].min(), df["order_date"].max()
        start, end = st.date_input(
            "Date range",
            value=(min_d.date(), max_d.date()),
            min_value=min_d.date(),
            max_value=max_d.date(),
        )
    else:
        start, end = None, None

    regions = sorted(df["region"].dropna().unique().tolist()) if "region" in df.columns else []
    cats    = sorted(df["category"].dropna().unique().tolist()) if "category" in df.columns else []
    pmodes  = sorted(df["payment_method"].dropna().unique().tolist()) if "payment_method" in df.columns else []

    sel_regions = st.multiselect("Region", regions, default=regions)
    sel_cats    = st.multiselect("Category", cats, default=cats)
    sel_pmodes  = st.multiselect("Payment Method", pmodes, default=pmodes)

# Apply filters
mask = pd.Series(True, index=df.index)
if start and end and "order_date" in df.columns:
    mask &= (df["order_date"].dt.date >= start) & (df["order_date"].dt.date <= end)
if sel_regions and "region" in df.columns:
    mask &= df["region"].isin(sel_regions)
if sel_cats and "category" in df.columns:
    mask &= df["category"].isin(sel_cats)
if sel_pmodes and "payment_method" in df.columns:
    mask &= df["payment_method"].isin(sel_pmodes)

fdf = df.loc[mask].copy()
if fdf.empty:
    st.warning("No data after filters. Adjust filters.")
    st.stop()

# ------------------------------------------
# KPI row
# ------------------------------------------
col1, col2, col3, col4 = st.columns(4)
total_net   = fdf["net_sales"].sum()
total_gross = fdf["gross_sales"].sum()
units       = fdf["quantity"].sum()
orders      = fdf["order_id"].nunique() if "order_id" in fdf.columns else len(fdf)
aov         = total_net / orders if orders else 0.0

col1.metric("Net Sales", f"${total_net:,.0f}")
col2.metric("Gross Sales", f"${total_gross:,.0f}")
col3.metric("Units Sold", f"{int(units):,}")
col4.metric("Avg Order Value (AOV)", f"${aov:,.2f}")

# ------------------------------------------
# FULL-WIDTH CHARTS (stacked)
# ------------------------------------------

# 1) Bar: Sales by Category
st.subheader("Sales by Category")
if "category" in fdf.columns:
    cat = (
        fdf.groupby("category", as_index=False)["net_sales"]
        .sum()
        .sort_values("net_sales", ascending=False)
    )
    fig = px.bar(cat, x="category", y="net_sales", text_auto=True, title="Net Sales by Category")
    fig.update_layout(yaxis_title="Net Sales", height=380)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No 'category' column found.")

# 2) Bar: Sales by Region
st.subheader("Sales by Region")
if "region" in fdf.columns:
    reg = (
        fdf.groupby("region", as_index=False)["net_sales"]
        .sum()
        .sort_values("net_sales", ascending=False)
    )
    fig = px.bar(reg, x="region", y="net_sales", text_auto=True, title="Net Sales by Region")
    fig.update_layout(yaxis_title="Net Sales", height=360)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No 'region' column found.")

# 3) Pie: Payment Method Mix
st.subheader("Payment Method Mix")
if "payment_method" in fdf.columns:
    pm = fdf.groupby("payment_method", as_index=False)["net_sales"].sum()
    fig = px.pie(pm, values="net_sales", names="payment_method", title="Net Sales by Payment Method", hole=0.35)
    fig.update_layout(height=460)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No 'payment_method' column found.")

# 3) Heatmap: Weekday x Category (Net Sales) 
st.subheader("Heatmap: Weekday × Category")
if {"weekday","category","net_sales"}.issubset(fdf.columns):
    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    temp = fdf.copy()
    temp["weekday"] = pd.Categorical(temp["weekday"], categories=order, ordered=True)
    piv = temp.pivot_table(index="weekday", columns="category", values="net_sales",
                           aggfunc="sum", fill_value=0)
    fig = px.imshow(piv, aspect="auto", color_continuous_scale="Blues",
                    labels=dict(color="Net Sales"), title="Net Sales by Weekday and Category")
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Need columns: weekday, category, net_sales.")


# 5) Full-width line chart: Net Sales Over Time
st.subheader("Net Sales Over Time")
if "date" in fdf.columns and fdf["date"].notna().any():
    ts = fdf.groupby("date", as_index=False)["net_sales"].sum()
    fig = px.line(ts, x="date", y="net_sales", markers=True, title="Daily Net Sales")
    fig.update_layout(hovermode="x unified", yaxis_title="Net Sales", height=420)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No valid dates to plot.")



# === Units Sold Over Time (full width) ===
st.subheader("Units Sold Over Time")
if "date" in fdf.columns and fdf["date"].notna().any():
    uts = fdf.groupby("date", as_index=False)["quantity"].sum()
    fig = px.area(uts, x="date", y="quantity", title="Daily Units Sold")
    fig.update_layout(hovermode="x unified", yaxis_title="Units", height=360)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No valid dates to plot for units.")

# # 5) Scatter: Discount vs Net Sales (by Category) — NO trendline (keeps Py3.13 wheel-only)
# st.subheader("Discount vs Net Sales (by Category)")
# needed = {"discount", "net_sales", "category"}
# if needed.issubset(fdf.columns):
#     agg = (
#         fdf.groupby("category", as_index=False)
#         .agg(avg_discount=("discount", "mean"),
#              net_sales=("net_sales", "sum"))
#     )
#     fig = px.scatter(
#         agg, x="avg_discount", y="net_sales",
#         labels={"avg_discount": "Average Discount", "net_sales": "Net Sales"},
#         title="Discount Intensity vs Net Sales",
#     )
#     fig.update_layout(height=380)
#     st.plotly_chart(fig, use_container_width=True)
# else:
#     st.info("Need columns: discount, net_sales, category.")

# ------------------------------------------
# 6) Improved: Discount vs Net Sales (by Category)
# ------------------------------------------
st.subheader("Discount vs Net Sales (by Category)")

needed = {"discount", "net_sales", "category"}
if needed.issubset(fdf.columns):

    # ---- aggregate category KPIs
    agg = (
        fdf.groupby("category", as_index=False)
          .agg(avg_discount=("discount", "mean"),
               net_sales=("net_sales", "sum"),
               orders=("order_id", "nunique"),
               units=("quantity", "sum"))
    )
    # Guard against zeros/negatives for logs
    agg = agg.replace([np.inf, -np.inf], np.nan).dropna(subset=["avg_discount", "net_sales"])
    agg = agg[(agg["avg_discount"] >= 0) & (agg["net_sales"] > 0)]

    # ---- simple linear regression with NumPy (no SciPy)
    x = agg["avg_discount"].to_numpy()
    y = agg["net_sales"].to_numpy()
    # y = a*x + b
    a, b = np.polyfit(x, y, deg=1)
    y_hat = a * x + b
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # ---- plot (bubble size = orders)
    log_scale = st.toggle("Log scale (Y)", value=False, help="Good when categories have very different sales sizes.")
    fig = px.scatter(
        agg, x="avg_discount", y="net_sales",
        size="orders", color="category",
        hover_data={"avg_discount":":.2%", "net_sales":":,.0f", "orders":":,",
                    "units":":," , "category":True},
        labels={"avg_discount":"Average Discount", "net_sales":"Net Sales"},
    )
    # tidy marker sizing
    fig.update_traces(marker=dict(opacity=0.85, line=dict(width=0.5)))

    # ---- regression line (manual trace)
    xs = np.linspace(float(x.min()), float(x.max()), 100)
    ys = a * xs + b
    fig.add_scatter(x=xs, y=ys, mode="lines", name=f"Trend (R²={r2:.2f})")

    # ---- quadrant guides (median split)
    x_med = float(np.median(x))
    y_med = float(np.median(y))
    fig.add_vline(x=x_med, line_dash="dot", opacity=0.35)
    fig.add_hline(y=y_med, line_dash="dot", opacity=0.35)

    # optional quadrant shading (light tint)
    fig.add_vrect(x0=x.min(), x1=x_med, fillcolor="LightGreen", opacity=0.05, line_width=0)
    fig.add_vrect(x0=x_med, x1=x.max(), fillcolor="LightCoral", opacity=0.05, line_width=0)
    fig.add_hrect(y0=y_med, y1=y.max(), fillcolor="LightSkyBlue", opacity=0.05, line_width=0)
    # (bottom-left remains unshaded)

    # ---- axes & layout
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

    # ---- quick takeaway list (top/bottom 3 categories by sales efficiency)
    # efficiency = sales per discount point (rough proxy)
    agg["sales_per_discount_point"] = agg["net_sales"] / np.maximum(agg["avg_discount"], 1e-6)
    top3 = agg.sort_values("sales_per_discount_point", ascending=False).head(3)[["category","sales_per_discount_point"]]
    bottom3 = agg.sort_values("sales_per_discount_point", ascending=True).head(3)[["category","sales_per_discount_point"]]

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Top efficiency (sales per discount point)**")
        st.dataframe(top3.assign(sales_per_discount_point=lambda d: d["sales_per_discount_point"].map("{:,.0f}".format)),
                     hide_index=True, use_container_width=True)
    with col_b:
        st.markdown("**Bottom efficiency (needs review)**")
        st.dataframe(bottom3.assign(sales_per_discount_point=lambda d: d["sales_per_discount_point"].map("{:,.0f}".format)),
                     hide_index=True, use_container_width=True)

else:
    st.info("Need columns: discount, net_sales, category.")


# ------------------------------------------
# Detail table + download
# ------------------------------------------
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

st.caption("Note: Discounts are treated as fractions (e.g., 0.10 = 10%). If your data stores 10 as 10%, divide by 100 before upload.")


# # streamlit_app.py
# import io
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy import stats
# import streamlit as st

# st.set_page_config(page_title="Quick EDA", page_icon="📊", layout="wide")
# st.title("📊 Quick EDA (CSV/Excel)")

# # ---------- File Upload ----------
# file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])
# if not file:
#     st.info("Upload a file to begin.")
#     st.stop()

# # Read file (auto-detect type)
# @st.cache_data
# def load_df(uploaded):
#     name = getattr(uploaded, "name", "")
#     if name.lower().endswith((".xlsx", ".xls")):
#         return pd.read_excel(uploaded)  # needs openpyxl for .xlsx
#     else:
#         # Try to auto-detect encoding; fall back to default
#         data = uploaded.read()
#         try:
#             return pd.read_csv(io.BytesIO(data))
#         except Exception:
#             uploaded.seek(0)
#             return pd.read_csv(uploaded, encoding_errors="ignore")

# df = load_df(file)
# st.success(f"Loaded shape: {df.shape}")

# st.subheader("Preview")
# st.dataframe(df.head(), use_container_width=True)

# # ---------- Basic Checks ----------
# st.subheader("Missing Values")
# st.write(df.isnull().sum())

# st.subheader("Dtypes")
# st.write(df.dtypes)

# # ---------- Choose columns (optional) ----------
# num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

# with st.expander("Column selection"):
#     st.caption("Auto-detected numeric and categorical columns (you can adjust):")
#     num_cols = st.multiselect("Numeric", num_cols, default=num_cols)
#     cat_cols = st.multiselect("Categorical", cat_cols, default=cat_cols)

# # ---------- Descriptives ----------
# if num_cols:
#     st.subheader("Descriptive Statistics (numeric)")
#     st.write(df[num_cols].describe())

# for c in cat_cols[:3]:  # show a few to keep it simple
#     st.subheader(f"Value counts: {c}")
#     st.write(df[c].value_counts(dropna=False))

# # ---------- Plots ----------
# sns.set_theme()

# def show_plot(fig):
#     st.pyplot(fig, clear_figure=True)

# # Histogram for a numeric column
# if num_cols:
#     st.subheader("Histogram")
#     col = st.selectbox("Select numeric column for histogram:", num_cols)
#     bins = st.slider("Bins", 10, 100, 30)
#     fig, ax = plt.subplots(figsize=(7, 4))
#     sns.histplot(df[col].dropna(), bins=bins, kde=True, ax=ax)
#     ax.set_title(f"Distribution of {col}")
#     show_plot(fig)

# # Correlation heatmap
# if len(num_cols) >= 2:
#     st.subheader("Correlation Heatmap")
#     corr = df[num_cols].corr(numeric_only=True)
#     fig, ax = plt.subplots(figsize=(7, 6))
#     sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
#     ax.set_title("Correlation (numeric)")
#     show_plot(fig)

# # Count plot for a categorical
# if cat_cols:
#     st.subheader("Category Count Plot")
#     cat = st.selectbox("Select categorical column:", cat_cols, key="cat_count")
#     fig, ax = plt.subplots(figsize=(7, 4))
#     sns.countplot(x=cat, data=df, ax=ax)
#     ax.set_title(f"Counts: {cat}")
#     ax.tick_params(axis='x', rotation=45)
#     show_plot(fig)

# # Box plot: numeric vs categorical
# if num_cols and cat_cols:
#     st.subheader("Box Plot (numeric vs categorical)")
#     y = st.selectbox("Numeric (Y)", num_cols, key="box_y")
#     x = st.selectbox("Categorical (X)", cat_cols, key="box_x")
#     fig, ax = plt.subplots(figsize=(8, 4))
#     sns.boxplot(data=df, x=x, y=y, ax=ax)
#     ax.set_title(f"{y} across {x}")
#     ax.tick_params(axis='x', rotation=45)
#     show_plot(fig)

# # ---------- Simple Tests ----------
# st.subheader("Simple Statistical Tests")

# # T-test: pick a numeric and a binary categorical
# if num_cols and cat_cols:
#     t_num = st.selectbox("Numeric for t-test", num_cols, key="ttest_num")
#     t_cat = st.selectbox("Binary categorical for t-test", cat_cols, key="ttest_cat")
#     if df[t_cat].nunique(dropna=True) == 2:
#         levels = [lvl for lvl in df[t_cat].dropna().unique()]
#         a = df.loc[df[t_cat] == levels[0], t_num].dropna()
#         b = df.loc[df[t_cat] == levels[1], t_num].dropna()
#         if len(a) > 1 and len(b) > 1:
#             t_stat, p_val = stats.ttest_ind(a, b, equal_var=False, nan_policy='omit')
#             st.write(f"**T-Test** ({t_cat}: {levels[0]} vs {levels[1]}) on {t_num} → t={t_stat:.3f}, p={p_val:.4f}")
#         else:
#             st.info("Not enough data in both groups for t-test.")
#     else:
#         st.info(f"Selected categorical '{t_cat}' isn’t binary.")

# # One-way ANOVA: numeric vs multi-level categorical
# if num_cols and cat_cols:
#     a_num = st.selectbox("Numeric for ANOVA", num_cols, key="anova_num")
#     a_cat = st.selectbox("Categorical for ANOVA (≥2 levels)", cat_cols, key="anova_cat")
#     groups = [g.dropna().values for _, g in df.groupby(a_cat)[a_num]]
#     if len(groups) >= 2 and all(len(g) > 1 for g in groups):
#         F, p = stats.f_oneway(*groups)
#         st.write(f"**ANOVA** ({a_num} ~ {a_cat}) → F={F:.3f}, p={p:.4f}")
#     else:
#         st.info("Need at least two groups with >1 value each for ANOVA.")

# st.caption("Tip: For Excel files, make sure your sheet has tidy headers. For very large files, consider CSV for faster uploads.")
