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



import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Title
st.title("Basic Data Analysis App")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Show dataset preview
    st.subheader("Dataset Preview")
    st.write(df.head())

    # Show basic info
    st.subheader("Summary Statistics")
    st.write(df.describe())

    # Column selection
    st.subheader("Column-wise Analysis")
    column = st.selectbox("Select a column for analysis", df.columns)

    if pd.api.types.is_numeric_dtype(df[column]):
        st.write(f"Summary of {column}:")
        st.write(df[column].describe())

        # Histogram
        fig, ax = plt.subplots()
        df[column].hist(ax=ax, bins=20)
        ax.set_title(f"Histogram of {column}")
        st.pyplot(fig)

    else:
        st.write(f"Value counts of {column}:")
        st.write(df[column].value_counts())
