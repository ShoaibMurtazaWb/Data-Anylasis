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

# streamlit_app.py
import io
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Financial Sales Dashboard", page_icon="📈", layout="wide")
st.title("📈 Financial Sales Dashboard")

# ------------------------------------------
# Expected schema & helpers
# ------------------------------------------
EXPECTED_COLS = [
    "order_id", "customer_id", "product_id", "category", "quantity",
    "price", "discount", "order_date", "region", "payment_method"
]

@st.cache_data
def load_df(file) -> pd.DataFrame:
    if file is None:
        return pd.DataFrame()
    name = getattr(file, "name", "").lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)  # requires openpyxl for .xlsx
    # CSV
    data = file.read()
    try:
        return pd.read_csv(io.BytesIO(data))
    except Exception:
        file.seek(0)
        return pd.read_csv(file, encoding_errors="ignore")

def coerce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and coerce types."""
    if df.empty:
        return df
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        st.warning(f"Missing columns: {missing}. Proceeding with available columns.")
    # Parse date
    if "order_date" in df.columns:
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    # Numerics
    for c in ("quantity", "price", "discount"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def add_computed(df: pd.DataFrame) -> pd.DataFrame:
    """Compute gross/net sales and time breakdowns."""
    out = df.copy()
    if {"quantity", "price"}.issubset(out.columns):
        out["gross_sales"] = out["quantity"] * out["price"]
    else:
        out["gross_sales"] = np.nan
    if "discount" in out.columns:
        # assume discount is a fraction (0.10 = 10%)
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
# Sidebar: upload & filters
# ------------------------------------------
with st.sidebar:
    st.header("🔧 Controls")
    data_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
    st.caption("Expected columns: " + ", ".join(EXPECTED_COLS))

df = load_df(data_file)
if df.empty:
    st.info("Upload a CSV/Excel file to begin.")
    st.stop()

df = coerce_schema(df)
df = add_computed(df)

# Ensure essential fields exist
essential = ["net_sales", "gross_sales", "quantity"]
df = df.dropna(subset=[c for c in essential if c in df.columns])

# Dynamic filters
with st.sidebar:
    # Date range
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
    sel_regions = st.multiselect("Region", regions, default=regions)

    cats = sorted(df["category"].dropna().unique().tolist()) if "category" in df.columns else []
    sel_cats = st.multiselect("Category", cats, default=cats)

    pmodes = sorted(df["payment_method"].dropna().unique().tolist()) if "payment_method" in df.columns else []
    sel_pmodes = st.multiselect("Payment Method", pmodes, default=pmodes)

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
total_net = fdf["net_sales"].sum()
total_gross = fdf["gross_sales"].sum()
units = fdf["quantity"].sum()
orders = fdf["order_id"].nunique() if "order_id" in fdf.columns else len(fdf)
aov = total_net / orders if orders else 0.0
disc_rate = 1 - (total_net / total_gross) if total_gross > 0 else 0.0

col1.metric("Net Sales", f"${total_net:,.0f}")
col2.metric("Gross Sales", f"${total_gross:,.0f}")
col3.metric("Units Sold", f"{int(units):,}")
col4.metric("Avg Order Value (AOV)", f"${aov:,.2f}")

# ------------------------------------------
# Time series & category performance
# ------------------------------------------
row1_col1, row1_col2 = st.columns((2, 1))

with row1_col1:
    st.subheader("Net Sales Over Time")
    if "date" in fdf.columns and fdf["date"].notna().any():
        ts = fdf.groupby("date", as_index=False)["net_sales"].sum()
        fig = px.line(ts, x="date", y="net_sales", markers=True, title="Daily Net Sales")
        fig.update_layout(hovermode="x unified", yaxis_title="Net Sales")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No valid dates to plot.")

with row1_col2:
    st.subheader("Sales by Category")
    if "category" in fdf.columns:
        cat = (
            fdf.groupby("category", as_index=False)["net_sales"]
            .sum()
            .sort_values("net_sales", ascending=False)
        )
        fig = px.bar(cat, x="category", y="net_sales", text_auto=True, title="Net Sales by Category")
        fig.update_layout(yaxis_title="Net Sales")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No 'category' column found.")

# ------------------------------------------
# Region & payment mix
# ------------------------------------------
row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    st.subheader("Sales by Region")
    if "region" in fdf.columns:
        reg = (
            fdf.groupby("region", as_index=False)["net_sales"]
            .sum()
            .sort_values("net_sales", ascending=False)
        )
        fig = px.bar(reg, x="region", y="net_sales", text_auto=True, title="Net Sales by Region")
        fig.update_layout(yaxis_title="Net Sales")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No 'region' column found.")

with row2_col2:
    st.subheader("Payment Method Mix")
    if "payment_method" in fdf.columns:
        pm = fdf.groupby("payment_method", as_index=False)["net_sales"].sum()
        fig = px.pie(pm, values="net_sales", names="payment_method", title="Net Sales by Payment Method", hole=0.35)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No 'payment_method' column found.")

# ------------------------------------------
# Discount vs revenue & top customers
# ------------------------------------------
row3_col1, row3_col2 = st.columns(2)

with row3_col1:
    st.subheader("Discount vs Net Sales (by Category)")
    needed = {"discount", "net_sales", "category"}
    if needed.issubset(fdf.columns):
        agg = (
            fdf.groupby("category", as_index=False)
            .agg(avg_discount=("discount", "mean"), net_sales=("net_sales", "sum"))
        )
        fig = px.scatter(
            agg, x="avg_discount", y="net_sales",
            trendline="ols",  # requires statsmodels in Plotly env, but optional
            labels={"avg_discount": "Average Discount", "net_sales": "Net Sales"},
            title="Discount Intensity vs Net Sales"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need columns: discount, net_sales, category.")

with row3_col2:
    st.subheader("Top 10 Customers by Net Sales")
    needed = {"customer_id", "net_sales"}
    if needed.issubset(fdf.columns):
        top_cust = (
            fdf.groupby("customer_id", as_index=False)
            .agg(net_sales=("net_sales", "sum"), orders=("order_id", "nunique"))
            .sort_values("net_sales", ascending=False)
            .head(10)
        )
        st.dataframe(top_cust, use_container_width=True, hide_index=True)
    else:
        st.info("Need columns: customer_id, net_sales.")

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
