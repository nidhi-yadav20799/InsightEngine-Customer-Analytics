import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
from pathlib import Path

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="InsightEngine Customer Analytics Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# STYLING
# ============================================================
st.markdown("""
<style>
    .stApp {
        background-color: #f5f6fa;
    }
    section[data-testid="stSidebar"] {
        background-color: #1f2a44;
    }
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p {
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] div[data-baseweb="select"] * {
        color: #1f2a44 !important;
    }
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e3e6ec;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    div[data-testid="stMetricLabel"] { color: #6b7280; font-weight: 600; }
    div[data-testid="stMetricValue"] { color: #1f2a44; }
    h1 { color: #1f2a44; font-weight: 700; }
    h2, h3 { color: #1f2a44; }
    .block-container { padding-top: 2rem; }

    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border-radius: 8px 8px 0 0;
        padding: 10px 18px;
        color: #1f2a44;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f2a44 !important;
        color: #ffffff !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }

    table { width: 100% !important; margin-left: auto; margin-right: auto; }
    table th, table td { text-align: center !important; }
</style>
""", unsafe_allow_html=True)

sns.set_theme(style="whitegrid")
plt.rcParams["axes.edgecolor"] = "#4a4a4a"
plt.rcParams["axes.labelcolor"] = "#2b2b2b"
plt.rcParams["text.color"] = "#2b2b2b"

# ---- Curated color palette (used consistently, not random) ----
NAVY = "#1f2a44"
STEEL_BLUE = "#3b6ea5"
TEAL = "#2a9d8f"
CORAL = "#e76f51"
AMBER = "#e9c46a"
SLATE = "#264653"

ACCENT = STEEL_BLUE  # kept for backward-compatible references below

# ============================================================
# HEADER
# ============================================================
st.title("InsightEngine Customer Analytics Dashboard")
st.caption("Customer Analytics  ·  EDA  ·  Segmentation  ·  Churn  ·  Recommendations")
st.divider()

# ============================================================
# DATA LOADING
# ============================================================
def categorize_product(description):
    """Derive a product category from the description using keyword matching.
    The dataset has no native category column, so this heuristic groups
    products into business-relevant buckets for filtering and analysis."""
    desc = str(description).upper()
    categories = {
        "Christmas": ["CHRISTMAS", "XMAS", "SANTA", "ADVENT"],
        "Bags": ["BAG", "SHOPPER"],
        "Home Decor": ["LIGHT", "CANDLE", "LANTERN", "ORNAMENT", "FRAME", "CLOCK"],
        "Kitchen & Dining": ["MUG", "CUP", "PLATE", "BOWL", "TEAPOT", "KITCHEN", "JAR"],
        "Stationery": ["CARD", "NOTEBOOK", "PEN", "PENCIL", "STICKER"],
        "Toys & Games": ["TOY", "GAME", "PUZZLE", "DOLL"],
        "Garden": ["GARDEN", "PLANT", "FLOWER", "POT"],
        "Bath & Body": ["SOAP", "BATH", "TOWEL"],
    }
    for cat, keywords in categories.items():
        if any(k in desc for k in keywords):
            return cat
    return "Other"

@st.cache_data
def load_data():
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_FILE = BASE_DIR / "data" / "online_retail_II.xlsx"

    df = pd.read_excel(DATA_FILE, sheet_name="Year 2009-2010")
    df = df.dropna(subset=["Customer ID"])
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["TotalAmount"] = df["Quantity"] * df["Price"]
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
    df["Category"] = df["Description"].apply(categorize_product)

    return df

# ============================================================
# SIDEBAR FILTERS
# ============================================================
st.sidebar.header("Filters")

min_date = df["InvoiceDate"].min().date()
max_date = df["InvoiceDate"].max().date()

date_range = st.sidebar.date_input(
    "Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
    key="date_range_filter"
)

country_options = sorted(df["Country"].dropna().unique())
selected_country = st.sidebar.selectbox(
    "Country",
    options=["All"] + country_options,
    key="country_filter"
)

category_options = sorted(df["Category"].dropna().unique())
selected_categories = st.sidebar.multiselect(
    "Category",
    options=category_options,
    default=[],
    key="category_filter",
    placeholder="All categories"
)

if st.sidebar.button("Reset Filters"):
    st.session_state["date_range_filter"] = (min_date, max_date)
    st.session_state["country_filter"] = "All"
    st.session_state["category_filter"] = []
    st.rerun()

st.sidebar.divider()
st.sidebar.caption("InsightEngine v1.0 · Internal Analytics Tool")

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

mask = (df["InvoiceDate"].dt.date >= start_date) & (df["InvoiceDate"].dt.date <= end_date)
if selected_country != "All":
    mask &= df["Country"] == selected_country
if selected_categories:
    mask &= df["Category"].isin(selected_categories)

filtered_df = df[mask]

if filtered_df.empty:
    st.warning("No data available for the selected filters. Please adjust the date range, country, or category.")
    st.stop()

# ============================================================
# KPI SECTION
# ============================================================
st.header("Overview")

total_revenue = filtered_df["TotalAmount"].sum()
total_customers = filtered_df["Customer ID"].nunique()
total_orders = filtered_df["Invoice"].nunique()
avg_order_value = total_revenue / total_orders if total_orders else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Revenue", f"£{total_revenue:,.0f}")
col2.metric("Customers", f"{total_customers:,}")
col3.metric("Orders", f"{total_orders:,}")
col4.metric("Avg Order Value", f"£{avg_order_value:.2f}")

st.caption(f"Showing {len(filtered_df):,} transactions · {start_date} to {end_date}")
st.divider()

# ============================================================
# EDA SECTION
# ============================================================
st.header("Exploratory Data Analysis")

st.subheader("Revenue Over Time")
daily_revenue = filtered_df.groupby(filtered_df["InvoiceDate"].dt.date)["TotalAmount"].sum()
st.line_chart(daily_revenue, color=STEEL_BLUE)

eda_col1, eda_col2 = st.columns(2)

with eda_col1:
    st.subheader("Order Value Distribution")
    order_values = filtered_df.groupby("Invoice")["TotalAmount"].sum()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(order_values, bins=40, color=TEAL, ax=ax)
    ax.set_xlabel("Order Value (£)")
    ax.set_ylabel("Number of Orders")
    st.pyplot(fig)

with eda_col2:
    st.subheader("Top 10 Products by Revenue")
    top_revenue = (
        filtered_df.groupby("Description")["TotalAmount"]
        .sum().sort_values(ascending=False).head(10)
    )
    st.bar_chart(top_revenue, color=CORAL)

st.subheader("Order Quantity Variability — Top 10 Revenue Products")
top_products = top_revenue.index
fig2, ax2 = plt.subplots(figsize=(10, 5))
sns.boxplot(
    data=filtered_df[filtered_df["Description"].isin(top_products)],
    x="Quantity", y="Description", color=STEEL_BLUE, ax=ax2
)
ax2.set_xlabel("Quantity per Order")
ax2.set_ylabel("Product")
st.pyplot(fig2)

eda_col3, eda_col4 = st.columns(2)

with eda_col3:
    st.subheader("Revenue by Country (Top 10)")
    country_rev = (
        filtered_df.groupby("Country")["TotalAmount"]
        .sum().sort_values(ascending=False).head(10)
    )
    st.bar_chart(country_rev, color=AMBER)

with eda_col4:
    st.subheader("Top 10 Customers by Revenue")
    top_customers = (
        filtered_df.groupby("Customer ID")["TotalAmount"]
        .sum().sort_values(ascending=False).head(10)
    )
    st.bar_chart(top_customers, color=SLATE)

st.subheader("Revenue by Category")
category_rev = (
    filtered_df.groupby("Category")["TotalAmount"]
    .sum().sort_values(ascending=False)
)
st.bar_chart(category_rev, color=TEAL)
st.caption("Category derived from product description keywords (dataset has no native category field).")

st.subheader("Correlation Heatmap — Quantity, Price, Revenue")
corr_data = filtered_df[["Quantity", "Price", "TotalAmount"]].corr()
fig_corr, ax_corr = plt.subplots(figsize=(6, 4))
sns.heatmap(corr_data, annot=True, cmap="coolwarm", center=0, fmt=".2f", ax=ax_corr, cbar=True)
st.pyplot(fig_corr)
st.caption("Shows linear relationships between order-level Quantity, Price, and Revenue.")

st.divider()

# ============================================================
# CACHED RFM / SEGMENTATION / SIMILARITY HELPERS
# ============================================================
@st.cache_data
def compute_rfm(data):
    snapshot_date = data["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = data.groupby("Customer ID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("Invoice", "nunique"),
        Monetary=("TotalAmount", "sum")
    )
    return rfm

@st.cache_data
def run_kmeans(rfm, n_clusters=4):
    n_clusters = max(1, min(n_clusters, len(rfm)))

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm = rfm.copy()
    rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

    cluster_order = rfm.groupby("Cluster")["Monetary"].mean().sort_values(ascending=False).index
    all_segment_names = ["High Value Customers", "Loyal Customers", "Regular Customers", "At Risk Customers"]
    segment_names = {cluster_order[i]: all_segment_names[i] for i in range(len(cluster_order))}

    rfm["Segment"] = rfm["Cluster"].map(segment_names)
    return rfm

@st.cache_data
def build_similarity(data):
    user_item_matrix = data.pivot_table(
        index="Customer ID", columns="Description", values="Quantity",
        aggfunc="sum", fill_value=0
    )
    similarity = cosine_similarity(user_item_matrix)
    similarity_df = pd.DataFrame(similarity, index=user_item_matrix.index, columns=user_item_matrix.index)
    return user_item_matrix, similarity_df

def recommend_products(customer_id, user_item_matrix, similarity_df, n_recommendations=5):
    similar_customers = (
        similarity_df.loc[customer_id].sort_values(ascending=False).iloc[1:11].index
    )
    recommendation_scores = (
        user_item_matrix.loc[similar_customers].sum(axis=0).sort_values(ascending=False)
    )
    purchased = set(
        user_item_matrix.loc[customer_id][user_item_matrix.loc[customer_id] > 0].index
    )
    recommendation_scores = recommendation_scores[~recommendation_scores.index.isin(purchased)]
    recommendation_scores = recommendation_scores[recommendation_scores > 0]
    return recommendation_scores.head(n_recommendations)

rfm_base = compute_rfm(filtered_df)

if len(rfm_base) < 4:
    st.warning(
        f"Only {len(rfm_base)} customer(s) found in the selected filters — "
        "not enough data for full 4-segment clustering. Try widening the Date/Country/Category filters."
    )

rfm = run_kmeans(rfm_base)
user_item_matrix, similarity_df = build_similarity(filtered_df)

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs(
    ["Customer Segmentation", "Churn Predictor", "Recommendations", "A/B Testing"]
)

# ---------------- Segmentation ----------------
with tab1:
    st.header("Customer Segmentation")
    st.caption("RFM Analysis · K-Means Clustering")

    seg_col1, seg_col2 = st.columns(2)

    with seg_col1:
        st.subheader("Segment Distribution")
        st.bar_chart(rfm["Segment"].value_counts(), color=STEEL_BLUE)

    with seg_col2:
        st.subheader("Average Revenue by Segment")
        st.bar_chart(rfm.groupby("Segment")["Monetary"].mean(), color=CORAL)

    st.subheader("Segment Profile")
    profile = rfm.groupby("Segment")[["Recency", "Frequency", "Monetary"]].mean().round(2)
    st.table(profile)
    st.caption("Lower Recency = more recent purchase. Higher Frequency/Monetary = more valuable customer.")

    with st.expander("View customer-level segmentation data"):
        st.dataframe(rfm.reset_index(), use_container_width=True)
        st.download_button(
            "Download Segmentation Report (CSV)",
            rfm.reset_index().to_csv(index=False).encode("utf-8"),
            "customer_segments.csv",
            "text/csv"
        )

# ---------------- Churn Predictor ----------------
with tab2:
    st.header("Customer Churn Predictor")
    st.caption("Logistic Regression on RFM features · Train/Test Split Evaluation")

    churn_df = rfm_base.copy()
    churn_df["Churn"] = (churn_df["Recency"] > 90).astype(int)

    if churn_df["Churn"].nunique() < 2 or len(churn_df) < 10:
        st.warning("Not enough data/churn variation in the current filtered data to train and evaluate a model. Try widening the filters.")
    else:
        X = churn_df[["Recency", "Frequency", "Monetary"]]
        y = churn_df["Churn"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba)

        churn_df["Churn Probability"] = model.predict_proba(X)[:, 1]

        st.subheader("Model Performance (on held-out test set)")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{accuracy*100:.1f}%")
        m2.metric("Precision", f"{precision*100:.1f}%")
        m3.metric("Recall", f"{recall*100:.1f}%")
        m4.metric("F1 Score", f"{f1*100:.1f}%")
        m5.metric("ROC-AUC", f"{roc_auc:.3f}")

        perf_col1, perf_col2 = st.columns(2)

        with perf_col1:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm, ax_cm = plt.subplots(figsize=(4.5, 4))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"], ax=ax_cm
            )
            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("Actual")
            st.pyplot(fig_cm)

        with perf_col2:
            st.subheader("ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            fig_roc, ax_roc = plt.subplots(figsize=(4.5, 4))
            ax_roc.plot(fpr, tpr, color=CORAL, linewidth=2, label=f"AUC = {roc_auc:.3f}")
            ax_roc.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random Guess")
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.legend()
            st.pyplot(fig_roc)

        st.subheader("Feature Importance")
        importance = pd.Series(model.coef_[0], index=X.columns).sort_values()
        st.bar_chart(importance, color=SLATE)
        st.caption("Positive coefficient = increases churn risk. Negative = decreases churn risk.")

        st.divider()

        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("Customers at Risk (full dataset)", f"{churn_df['Churn'].mean()*100:.1f}%")
        metric_col2.metric("Total Customers Analyzed", f"{len(churn_df):,}")

        st.subheader("Lookup Individual Customer")
        customer = st.selectbox("Select Customer ID", churn_df.index.astype(int).sort_values().unique())
        row = churn_df.loc[customer]

        c1, c2, c3 = st.columns(3)
        c1.metric("Recency (days)", int(row["Recency"]))
        c2.metric("Frequency (orders)", int(row["Frequency"]))
        c3.metric("Total Spend", f"£{row['Monetary']:.2f}")

        st.progress(min(int(row["Churn Probability"] * 100), 100))
        if row["Churn"] == 1:
            st.error(f"High Churn Risk — {row['Churn Probability']*100:.1f}% probability")
        else:
            st.success(f"Low Churn Risk — {row['Churn Probability']*100:.1f}% probability")

        with st.expander("View all customer churn scores"):
            st.dataframe(
                churn_df.reset_index()[["Customer ID", "Recency", "Frequency", "Monetary", "Churn Probability"]],
                use_container_width=True
            )
            st.download_button(
                "Download Churn Report (CSV)",
                churn_df.reset_index().to_csv(index=False).encode("utf-8"),
                "churn_predictions.csv",
                "text/csv"
            )

# ---------------- Recommendations ----------------
with tab3:
    st.header("Product Recommendation Demo")
    st.caption("Collaborative Filtering · Cosine Similarity")

    customer = st.selectbox(
        "Customer ID",
        user_item_matrix.index.astype(int).sort_values().unique(),
        key="recommend_customer"
    )

    recommendations = recommend_products(customer, user_item_matrix, similarity_df)

    if recommendations.empty:
        st.warning("No recommendations available for this customer.")
    else:
        st.subheader(f"Top 5 Recommended Products for Customer {customer}")
        recommendation_df = recommendations.reset_index()
        recommendation_df.columns = ["Product", "Recommendation Score"]
        st.table(recommendation_df.set_index("Product"))

# ---------------- A/B Testing ----------------
with tab4:
    st.header("A/B Test Simulation")
    st.caption("Statistical Significance Testing · Independent Samples t-test")

    st.markdown(
        "Simulates an A/B test comparing **average order value** between two customer "
        "groups split by country, to demonstrate statistical hypothesis testing on real transaction data."
    )

    ab_country_options = sorted(filtered_df["Country"].dropna().unique())

    if len(ab_country_options) < 2:
        st.warning("Need at least 2 countries in the filtered data to run an A/B test. Try widening the Country filter.")
    else:
        ab_col1, ab_col2 = st.columns(2)
        with ab_col1:
            group_a_country = st.selectbox("Group A (Country)", ab_country_options, index=0, key="ab_group_a")
        with ab_col2:
            default_b_index = 1 if len(ab_country_options) > 1 else 0
            group_b_country = st.selectbox("Group B (Country)", ab_country_options, index=default_b_index, key="ab_group_b")

        if group_a_country == group_b_country:
            st.warning("Please select two different countries to compare.")
        else:
            group_a_orders = filtered_df[filtered_df["Country"] == group_a_country].groupby("Invoice")["TotalAmount"].sum()
            group_b_orders = filtered_df[filtered_df["Country"] == group_b_country].groupby("Invoice")["TotalAmount"].sum()

            if len(group_a_orders) < 2 or len(group_b_orders) < 2:
                st.warning("Not enough orders in one of the selected groups to run a statistical test.")
            else:
                t_stat, p_value = stats.ttest_ind(group_a_orders, group_b_orders, equal_var=False)

                metric_col1, metric_col2, metric_col3 = st.columns(3)
                metric_col1.metric(f"{group_a_country} — Avg Order Value", f"£{group_a_orders.mean():.2f}")
                metric_col2.metric(f"{group_b_country} — Avg Order Value", f"£{group_b_orders.mean():.2f}")
                metric_col3.metric("P-Value", f"{p_value:.4f}")

                st.subheader("Order Value Distribution by Group")
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.histplot(group_a_orders, color=STEEL_BLUE, label=group_a_country, alpha=0.6, ax=ax, kde=True)
                sns.histplot(group_b_orders, color=CORAL, label=group_b_country, alpha=0.6, ax=ax, kde=True)
                ax.set_xlabel("Order Value (£)")
                ax.legend()
                st.pyplot(fig)

                st.subheader("Hypothesis Test Result")
                alpha = 0.05
                if p_value < alpha:
                    st.success(
                        f"**Statistically significant difference** (p = {p_value:.4f} < {alpha}). "
                        f"We reject the null hypothesis — average order values between **{group_a_country}** "
                        f"and **{group_b_country}** are significantly different."
                    )
                else:
                    st.info(
                        f"**No statistically significant difference** (p = {p_value:.4f} ≥ {alpha}). "
                        f"We fail to reject the null hypothesis — there isn't enough evidence that "
                        f"average order values differ between **{group_a_country}** and **{group_b_country}**."
                    )

                with st.expander("View test details"):
                    st.markdown(f"""
                    - **Test used:** Welch's independent samples t-test (unequal variances assumed)
                    - **Null hypothesis (H₀):** Mean order value of Group A = Mean order value of Group B
                    - **Alternative hypothesis (H₁):** Mean order value of Group A ≠ Mean order value of Group B
                    - **Significance level (α):** 0.05
                    - **t-statistic:** {t_stat:.4f}
                    - **p-value:** {p_value:.4f}
                    - **Group A sample size:** {len(group_a_orders)} orders
                    - **Group B sample size:** {len(group_b_orders)} orders
                    """)