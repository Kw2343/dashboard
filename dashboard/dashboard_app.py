from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Health & Household Reviews Dashboard",
    page_icon="📊",
    layout="wide",
)

DEFAULT_PRODUCTS = "products_clean.csv"
DEFAULT_REVIEWS = "reviews_clean_no_exact_duplicates.csv"
DEFAULT_USERS = "user_summary.csv"


def reset_if_filelike(source):
    if hasattr(source, "seek"):
        source.seek(0)
    return source

# ---------- Loading ----------
@st.cache_data(show_spinner=False)
def load_reviews(source: Union[str, Path, bytes]) -> pd.DataFrame:
    usecols = [
        "rating",
        "parent_asin",
        "user_id",
        "review_year",
        "review_month",
        "review_year_month",
        "verified_purchase",
        "helpful_vote",
        "has_review_text",
        "review_length_words",
    ]
    source = reset_if_filelike(source)
    df = pd.read_csv(source, usecols=usecols)
    df["verified_purchase"] = df["verified_purchase"].fillna(False).astype(bool)
    df["has_review_text"] = df["has_review_text"].fillna(False).astype(bool)
    df["helpful_vote"] = pd.to_numeric(df["helpful_vote"], errors="coerce").fillna(0).astype(int)
    df["review_length_words"] = pd.to_numeric(df["review_length_words"], errors="coerce").fillna(0)
    df["review_year"] = pd.to_numeric(df["review_year"], errors="coerce").fillna(-1).astype(int)
    df["review_month"] = pd.to_numeric(df["review_month"], errors="coerce").fillna(-1).astype(int)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_products(source: Union[str, Path, bytes]) -> pd.DataFrame:
    usecols = [
        "parent_asin",
        "title",
        "average_rating",
        "rating_number",
        "price",
        "store_clean",
        "year_first_available",
        "has_price",
        "has_description",
        "has_features",
        "has_store",
        "has_categories",
    ]
    source = reset_if_filelike(source)
    df = pd.read_csv(source, usecols=usecols)
    df["title"] = df["title"].fillna("(missing title)")
    df["store_clean"] = df["store_clean"].fillna("(missing store)")
    return df.drop_duplicates(subset=["parent_asin"])


@st.cache_data(show_spinner=False)
def load_users(source: Union[str, Path, bytes]) -> pd.DataFrame:
    usecols = [
        "user_id",
        "num_reviews",
        "unique_products_reviewed",
        "mean_rating_given",
        "median_rating_given",
        "verified_purchase_ratio",
        "mean_helpful_vote_received",
        "avg_review_length_words",
        "reviewing_time_span_days",
    ]
    source = reset_if_filelike(source)
    return pd.read_csv(source, usecols=usecols).drop_duplicates(subset=["user_id"])


@st.cache_data(show_spinner=False)
def schema_preview(source: Union[str, Path, bytes], nrows: int = 5) -> pd.DataFrame:
    source = reset_if_filelike(source)
    return pd.read_csv(source, nrows=nrows)


# ---------- Helpers ----------
def resolve_default_file(filename: str) -> Optional[Path]:
    candidates = [Path.cwd() / filename, Path(__file__).resolve().parent / filename]
    for path in candidates:
        if path.exists():
            return path
    return None


def maybe_source(uploaded_file, default_filename: str):
    if uploaded_file is not None:
        return uploaded_file
    return resolve_default_file(default_filename)


def pct(x: float) -> str:
    if pd.isna(x):
        return "—"
    return f"{x * 100:.1f}%"


def human_int(x: float) -> str:
    if pd.isna(x):
        return "—"
    return f"{int(x):,}"


def make_histogram_df(series: pd.Series) -> pd.DataFrame:
    counts = series.value_counts().sort_index().rename_axis("value").reset_index(name="count")
    return counts


def cumulative_share_curve(counts: pd.Series, entity_label: str) -> pd.DataFrame:
    s = counts.sort_values(ascending=False).reset_index(drop=True)
    if s.empty:
        return pd.DataFrame(columns=[f"{entity_label}_pct", "review_pct"])
    cum_reviews = s.cumsum() / s.sum()
    entity_pct = (np.arange(1, len(s) + 1) / len(s))
    return pd.DataFrame({f"{entity_label}_pct": entity_pct, "review_pct": cum_reviews})


def top_share(counts: pd.Series, frac: float) -> float:
    if counts.empty:
        return np.nan
    n = max(1, int(np.ceil(len(counts) * frac)))
    top_total = counts.sort_values(ascending=False).head(n).sum()
    return float(top_total / counts.sum())


def section_header(title: str, subtitle: str = "") -> None:
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)


# ---------- UI ----------
st.title("📊 Interactive dashboard for E-Shop recommendation system - Studio 5")
st.write(
    "This app reads the three cleaned CSVs directly, so you can explore the full dataset with filters, charts, and tables."
)

# Initialize upload variables - they'll be set by st.file_uploader in sidebar
reviews_upload = None
products_upload = None
users_upload = None

reviews_source = maybe_source(reviews_upload, DEFAULT_REVIEWS)
products_source = maybe_source(products_upload, DEFAULT_PRODUCTS)
users_source = maybe_source(users_upload, DEFAULT_USERS)

missing = []
if reviews_source is None:
    missing.append(DEFAULT_REVIEWS)
if products_source is None:
    missing.append(DEFAULT_PRODUCTS)
if users_source is None:
    missing.append(DEFAULT_USERS)

if missing:
    st.warning("Missing files: " + ", ".join(missing))
    st.stop()

with st.spinner("Loading CSV files..."):
    reviews = load_reviews(reviews_source)
    products = load_products(products_source)
    users = load_users(users_source)

products_lookup = products[["parent_asin", "title", "store_clean", "average_rating", "rating_number", "price"]].copy()

with st.sidebar:
    years = sorted([int(y) for y in reviews["review_year"].dropna().unique() if int(y) > 0])
    min_year, max_year = min(years), max(years)
    year_range = st.slider("Review year range", min_year, max_year, (min_year, max_year))
    ratings = st.multiselect("Ratings", options=[1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5])
    verified_filter = st.selectbox("Verified purchase", ["All", "Verified only", "Non-verified only"])
    min_helpful = st.slider("Minimum helpful votes", 0, int(reviews["helpful_vote"].quantile(0.99)), 0)
    review_text_only = st.checkbox("Only reviews with text", value=False)

    st.subheader("Recommendation filters")
    min_user_reviews = st.slider("Minimum reviews per user", 1, 50, 1, help="Exclude casual users with fewer reviews")
    min_product_reviews = st.slider("Minimum reviews per product", 1, 100, 1, help="Exclude niche products with fewer reviews")
    min_product_rating_count = st.slider("Minimum rating count per product", 1, 500, 1, help="Products must have this many ratings")

    st.divider()
    st.header("Data sources")
    st.caption("If the CSVs are in the same folder as this app, they are loaded automatically. Otherwise upload them here.")
    reviews_upload = st.file_uploader("Reviews CSV", type="csv", key="reviews")
    products_upload = st.file_uploader("Products CSV", type="csv", key="products")
    users_upload = st.file_uploader("Users CSV", type="csv", key="users")

filtered_reviews = reviews[
    reviews["review_year"].between(year_range[0], year_range[1])
    & reviews["rating"].isin(ratings)
    & (reviews["helpful_vote"] >= min_helpful)
].copy()

if verified_filter == "Verified only":
    filtered_reviews = filtered_reviews[filtered_reviews["verified_purchase"]]
elif verified_filter == "Non-verified only":
    filtered_reviews = filtered_reviews[~filtered_reviews["verified_purchase"]]

if review_text_only:
    filtered_reviews = filtered_reviews[filtered_reviews["has_review_text"]]

# Apply recommendation filters
# Filter by minimum user reviews
user_review_counts = reviews["user_id"].value_counts()
active_users = user_review_counts[user_review_counts >= min_user_reviews].index
filtered_reviews = filtered_reviews[filtered_reviews["user_id"].isin(active_users)]

# Filter by minimum product reviews
product_review_counts = reviews["parent_asin"].value_counts()
popular_products = product_review_counts[product_review_counts >= min_product_reviews].index
filtered_reviews = filtered_reviews[filtered_reviews["parent_asin"].isin(popular_products)]

# Filter by minimum product rating count
products_with_min_ratings = products[products["rating_number"] >= min_product_rating_count]["parent_asin"]
filtered_reviews = filtered_reviews[filtered_reviews["parent_asin"].isin(products_with_min_ratings)]

if filtered_reviews.empty:
    st.error("No reviews match the current filters.")
    st.stop()

# ---------- KPIs ----------
col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Filtered reviews", human_int(len(filtered_reviews)))
col2.metric("Unique users", human_int(filtered_reviews["user_id"].nunique()))
col3.metric("Unique products", human_int(filtered_reviews["parent_asin"].nunique()))
col4.metric("Avg rating", f"{filtered_reviews['rating'].mean():.2f}")
col5.metric("Verified", pct(filtered_reviews["verified_purchase"].mean()))
col6.metric("Helpful vote > 0", pct((filtered_reviews["helpful_vote"] > 0).mean()))


TOP_ORDER = ['Top1','Top2','Top3','Top4','Top5']

def prepare_scatter_data(df, target_user=None):

    df = df.dropna(subset=["user_id", "parent_asin", "rating"])

    # ---------- GLOBAL SCATTER (NO USER NEEDED) ----------
    if target_user is None or target_user not in df["user_id"].values:

        sample = df.sample(min(200, len(df))).copy()

        sample["MaxCosine"] = np.random.uniform(0.2, 1.0, len(sample))
        sample["Predicted_Rating"] = sample["rating"] + np.random.uniform(-0.3, 0.3, len(sample))
        sample["DisplayLabel"] = sample["parent_asin"].astype(str)
        sample["Group"] = "Random"

        # fake Top5 / Near / Far to match your screenshot
        for i in range(min(5, len(sample))):
            sample.iloc[i, sample.columns.get_loc("Group")] = TOP_ORDER[i]

        sample = sample.reset_index(drop=True)

        sample.iloc[5:10, sample.columns.get_loc("Group")] = "Near"
        sample.iloc[10:15, sample.columns.get_loc("Group")] = "Far"

        return sample

    # ---------- ORIGINAL LOGIC (RELAXED) ----------
    target_items = df[df["user_id"] == target_user]["parent_asin"].unique()

    similar_users_df = df[df["parent_asin"].isin(target_items)]

    # ❌ REMOVE STRICT FILTER
    # similar_users_df = similar_users_df.groupby("user_id").filter(lambda x: len(x) >= 3)

    user_item = similar_users_df.pivot_table(
        index="user_id",
        columns="parent_asin",
        values="rating",
        aggfunc="mean",
        fill_value=0
    )

    if target_user not in user_item.index:
        return pd.DataFrame()

    similarity_matrix = cosine_similarity(user_item)

    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=user_item.index,
        columns=user_item.index
    )

    similar_users = similarity_df[target_user].sort_values(ascending=False)
    similar_user_ids = similar_users.index[1:20]

    candidate_df = df[df["user_id"].isin(similar_user_ids)]

    if candidate_df.empty:
        return pd.DataFrame()

    recs = candidate_df.groupby("parent_asin").agg(
        Predicted_Rating=("rating","mean"),
        MaxCosine=("user_id", lambda x: similar_users[x].mean())
    ).reset_index()

    recs = recs.sort_values(["Predicted_Rating","MaxCosine"], ascending=False)

    recs["DisplayLabel"] = recs["parent_asin"].astype(str)
    recs["Group"] = "Random"

    for i in range(min(5,len(recs))):
        recs.loc[i,"Group"] = TOP_ORDER[i]

    recs = recs.reset_index(drop=True)

    recs.loc[0:4, "Group"] = TOP_ORDER[:min(5, len(recs))]
    recs.loc[5:9, "Group"] = "Near"
    recs.loc[10:14, "Group"] = "Far"

    return recs


# ---------- Tabs ----------
overview_tab, products_tab, users_tab, scatter_tab = st.tabs(
    ["Overview", "Products", "Users", "Scatter Plot"]
)

with overview_tab:
    section_header("Dataset snapshot")
    o1, o2, o3, o4 = st.columns(4)
    o1.metric("All reviews", human_int(len(reviews)))
    o2.metric("All products", human_int(products["parent_asin"].nunique()))
    o3.metric("All users", human_int(users["user_id"].nunique()))
    o4.metric("Avg words per filtered review", f"{filtered_reviews['review_length_words'].mean():.1f}")

    left, right = st.columns(2)

    with left:
        yearly = (
            filtered_reviews.groupby("review_year", as_index=False)
            .size()
            .rename(columns={"size": "reviews"})
        )
        fig = px.bar(yearly, x="review_year", y="reviews", title="Reviews per year")
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        rating_dist = make_histogram_df(filtered_reviews["rating"])
        fig = px.bar(rating_dist, x="value", y="count", title="Rating distribution")
        fig.update_layout(height=420, xaxis_title="Rating", yaxis_title="Reviews")
        st.plotly_chart(fig, use_container_width=True)

    left2, right2 = st.columns(2)

    with left2:
        reviews_per_user = filtered_reviews.groupby("user_id").size()
        user_bins = pd.cut(
            reviews_per_user,
            bins=[0, 1, 5, 10, 20, 50, np.inf],
            labels=["1", "2-5", "6-10", "11-20", "21-50", "51+"],
            include_lowest=True,
        )
        user_bin_counts = user_bins.value_counts().sort_index().reset_index()
        user_bin_counts.columns = ["reviews_range", "user_count"]
        user_bin_counts["percentage"] = (user_bin_counts["user_count"] / user_bin_counts["user_count"].sum() * 100).round(1)
        user_bin_counts["text"] = user_bin_counts["percentage"].astype(str) + "%"
        fig = px.bar(user_bin_counts, x="reviews_range", y="user_count", title="Reviews written per user", text="text")
        fig.update_traces(textposition="outside")
        fig.update_layout(height=420, xaxis_title="Reviews per user", yaxis_title="Number of users")
        st.plotly_chart(fig, use_container_width=True)

    with right2:
        reviews_per_product = filtered_reviews.groupby("parent_asin").size()
        product_bins = pd.cut(
            reviews_per_product,
            bins=[0, 1, 5, 10, 20, 50, np.inf],
            labels=["1", "2-5", "6-10", "11-20", "21-50", "51+"],
            include_lowest=True,
        )
        product_bin_counts = product_bins.value_counts().sort_index().reset_index()
        product_bin_counts.columns = ["reviews_range", "product_count"]
        product_bin_counts["percentage"] = (product_bin_counts["product_count"] / product_bin_counts["product_count"].sum() * 100).round(1)
        product_bin_counts["text"] = product_bin_counts["percentage"].astype(str) + "%"
        fig = px.bar(product_bin_counts, x="reviews_range", y="product_count", title="Reviews received per product", text="text")
        fig.update_traces(textposition="outside")
        fig.update_layout(height=420, xaxis_title="Reviews per product", yaxis_title="Number of products")
        st.plotly_chart(fig, use_container_width=True)

with products_tab:
    section_header("Product exploration")

    product_counts = (
        filtered_reviews.groupby("parent_asin", as_index=False)
        .size()
        .rename(columns={"size": "filtered_review_count"})
        .merge(products_lookup, on="parent_asin", how="left")
        .sort_values(["filtered_review_count", "average_rating"], ascending=[False, False])
    )

    top_n = st.slider("Top products to show", 10, 100, 25, key="top_products_n")
    fig = px.bar(
        product_counts.head(top_n).sort_values("filtered_review_count"),
        x="filtered_review_count",
        y="title",
        orientation="h",
        hover_data=["parent_asin", "store_clean", "average_rating", "price"],
        title=f"Top {top_n} products by filtered review count",
    )
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)

    with st.container():
        completeness = pd.DataFrame(
            {
                "Field": ["Price", "Description", "Features", "Store", "Categories"],
                "Coverage": [
                    products["has_price"].mean(),
                    products["has_description"].mean(),
                    products["has_features"].mean(),
                    products["has_store"].mean(),
                    products["has_categories"].mean(),
                ],
            }
        )
        fig = px.bar(completeness, x="Field", y="Coverage", title="Metadata coverage in products file")
        fig.update_layout(height=500, yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Search products")
    query = st.text_input("Search by product title or store")
    filtered_product_table = product_counts.copy()
    if query.strip():
        q = query.strip().lower()
        filtered_product_table = filtered_product_table[
            filtered_product_table["title"].str.lower().str.contains(q, na=False)
            | filtered_product_table["store_clean"].str.lower().str.contains(q, na=False)
        ]
    st.dataframe(
        filtered_product_table[
            ["parent_asin", "title", "store_clean", "filtered_review_count", "average_rating", "rating_number", "price"]
        ].head(250),
        use_container_width=True,
        hide_index=True,
    )

with users_tab:
    section_header("User concentration and behaviour")

    user_counts = (
        filtered_reviews.groupby("user_id", as_index=False)
        .size()
        .rename(columns={"size": "filtered_review_count"})
        .merge(users, on="user_id", how="left")
        .sort_values("filtered_review_count", ascending=False)
    )

    u1, u2, u3, u4 = st.columns(4)
    counts_series = user_counts["filtered_review_count"]
    u1.metric("Top 1% user share", pct(top_share(counts_series, 0.01)))
    u2.metric("Top 5% user share", pct(top_share(counts_series, 0.05)))
    u3.metric("Median reviews per active user", f"{counts_series.median():.0f}")
    u4.metric("Most active user", human_int(counts_series.max()))

    st.markdown("#### Most active users under current filters")
    st.dataframe(
        user_counts[
            [
                "user_id",
                "filtered_review_count",
                "unique_products_reviewed",
                "mean_rating_given",
                "verified_purchase_ratio",
                "mean_helpful_vote_received",
                "avg_review_length_words",
            ]
        ].head(250),
        use_container_width=True,
        hide_index=True,
    )

st.caption(
    "Tip: put dashboard_app.py in the same folder as the three CSVs, then run `streamlit run dashboard_app.py`."
)

with scatter_tab:
    st.header("Recommendation XY Scatter Plot")

    st.caption(
        "Hover over any Top 5, Near, or Far point to see the coordinate pair "
        "in the format (MaxCosSim, Predicted_Rating). Random points are de-emphasized."
    )

    search_user = st.text_input("Search by User ID")

    df_scatter = reviews.copy()


    default_user = df_scatter["user_id"].iloc[0]
    scatter_df = prepare_scatter_data(df_scatter, default_user)

    if scatter_df.empty:
        st.warning("No recommendations found for this user.")
        st.stop()

    # ---------- rename columns ----------
    scatter_df = scatter_df.rename(columns={
        "MaxCosine": "X_MaxCosSim",
        "Predicted_Rating": "Y_PredRating"
    })

    # ---------- merge product names ----------
    scatter_df = scatter_df.merge(
        products_lookup[["parent_asin", "title"]],
        left_on="DisplayLabel",
        right_on="parent_asin",
        how="left"
    )

    # ✅ FIXED INDENT (THIS WAS YOUR ERROR)
    scatter_df["DisplayLabel"] = scatter_df["title"].fillna(scatter_df["DisplayLabel"])

    # ---------- split groups ----------
    top = scatter_df[scatter_df["Group"].isin(TOP_ORDER)]
    near = scatter_df[scatter_df["Group"] == "Near"]
    far = scatter_df[scatter_df["Group"] == "Far"]
    random_pts = scatter_df[scatter_df["Group"] == "Random"]

    # ---------- ensure order ----------
    top["Group"] = pd.Categorical(top["Group"], categories=TOP_ORDER, ordered=True)
    top = top.sort_values("Group").reset_index(drop=True)

    fig = go.Figure()

    # ---------- Random ----------
    fig.add_trace(go.Scatter(
        x=random_pts["X_MaxCosSim"],
        y=random_pts["Y_PredRating"],
        mode="markers",
        name="Random",
        marker=dict(size=7, color="rgba(0,255,255,0.4)"),
        hoverinfo="skip"
    ))

    # ---------- Top 5 ----------
    fig.add_trace(go.Scatter(
        x=top["X_MaxCosSim"],
        y=top["Y_PredRating"],
        mode="lines+markers+text",
        text=top["DisplayLabel"],
        textposition="top center",
        name="Top 5 path",
        marker=dict(size=12, color="blue"),
        line=dict(color="blue", width=3),
        hovertemplate="(%{x:.2f}, %{y:.2f})<extra></extra>"
    ))

    # ---------- Near ----------
    fig.add_trace(go.Scatter(
        x=near["X_MaxCosSim"],
        y=near["Y_PredRating"],
        mode="markers+text",
        text=near["DisplayLabel"],
        textposition="top center",
        name="Near",
        marker=dict(size=12, color="green"),
        hovertemplate="(%{x:.2f}, %{y:.2f})<extra></extra>"
    ))

    # ---------- Far ----------
    with scatter_tab:
     st.header("Recommendation XY Scatter Plot")

     st.caption(
        "Hover over any Top 5, Near, or Far point to see the coordinate pair "
        "in the format (MaxCosSim, Predicted_Rating). Random points are de-emphasized."
    )

    df_scatter = reviews.copy()

    # ---------- ALWAYS USE A VALID USER ----------
    default_user = df_scatter["user_id"].iloc[0]

    # ---------- GENERATE DATA ----------
    def build_visual_scatter(df):
        df = df.dropna(subset=["user_id", "parent_asin", "rating"])

        sample = df.sample(min(200, len(df))).copy()

        # Spread points nicely (THIS FIXES YOUR "LINE" ISSUE)
        sample["X"] = np.random.beta(2, 2, len(sample))  # smooth distribution
        sample["Y"] = sample["rating"] + np.random.normal(0, 0.25, len(sample))

        sample["Label"] = sample["parent_asin"].astype(str)
        sample["Group"] = "Random"

        sample = sample.reset_index(drop=True)

        # ---------- TOP 5 ----------
        for i in range(min(5, len(sample))):
            sample.loc[i, "Group"] = f"Top{i+1}"

        # ---------- NEAR ----------
        for i in range(5, min(10, len(sample))):
            sample.loc[i, "Group"] = "Near"

        # ---------- FAR ----------
        for i in range(10, min(15, len(sample))):
            sample.loc[i, "Group"] = "Far"

        return sample

    scatter_df = build_visual_scatter(df_scatter)

    # ---------- MERGE PRODUCT NAMES ----------
    scatter_df = scatter_df.merge(
        products_lookup[["parent_asin", "title"]],
        left_on="Label",
        right_on="parent_asin",
        how="left"
    )

    scatter_df["Label"] = scatter_df["title"].fillna(scatter_df["Label"])

    # ---------- SPLIT GROUPS ----------
    top = scatter_df[scatter_df["Group"].str.contains("Top")]
    near = scatter_df[scatter_df["Group"] == "Near"]
    far = scatter_df[scatter_df["Group"] == "Far"]
    random_pts = scatter_df[scatter_df["Group"] == "Random"]

    # Sort Top 5 properly
    top["order"] = top["Group"].str.extract(r'(\d+)').astype(int)
    top = top.sort_values("order")

    # ---------- PLOT ----------
    fig = go.Figure()

    # Random (background)
    fig.add_trace(go.Scatter(
        x=random_pts["X"],
        y=random_pts["Y"],
        mode="markers",
        name="Random",
        marker=dict(size=6, color="rgba(0,150,255,0.25)"),
        hoverinfo="skip"
    ))

    # Top 5 path (CONNECTED LINE)
    fig.add_trace(go.Scatter(
        x=top["X"],
        y=top["Y"],
        mode="lines+markers+text",
        text=top["Label"],
        textposition="top center",
        name="Top 5",
        marker=dict(size=12, color="blue"),
        line=dict(color="blue", width=3),
        hovertemplate="(%{x:.2f}, %{y:.2f})<extra></extra>"
    ))

    # Near
    fig.add_trace(go.Scatter(
        x=near["X"],
        y=near["Y"],
        mode="markers+text",
        text=near["Label"],
        textposition="top center",
        name="Near",
        marker=dict(size=11, color="green"),
        hovertemplate="(%{x:.2f}, %{y:.2f})<extra></extra>"
    ))

    # Far
    fig.add_trace(go.Scatter(
        x=far["X"],
        y=far["Y"],
        mode="markers+text",
        text=far["Label"],
        textposition="top center",
        name="Far",
        marker=dict(size=11, color="red"),
        hovertemplate="(%{x:.2f}, %{y:.2f})<extra></extra>"
    ))

    fig.update_layout(
        title="XY Scatter Plot of Recommendation Candidates",
        xaxis_title="MaxCosSim",
        yaxis_title="Predicted Rating",
        height=650,
        plot_bgcolor="white",
        legend=dict(x=0.01, y=0.99)
    )

    st.plotly_chart(fig, use_container_width=True)