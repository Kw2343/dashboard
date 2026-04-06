from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

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
st.title("📊 Interactive dashboard for your uploaded CSV files")
st.write(
    "This app reads the three cleaned CSVs directly, so you can explore the full dataset with filters, charts, and tables."
)

with st.sidebar:
    st.header("Data sources")
    st.caption("If the CSVs are in the same folder as this app, they are loaded automatically. Otherwise upload them here.")
    reviews_upload = st.file_uploader("Reviews CSV", type="csv", key="reviews")
    products_upload = st.file_uploader("Products CSV", type="csv", key="products")
    users_upload = st.file_uploader("Users CSV", type="csv", key="users")

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
    st.header("Review filters")
    years = sorted([int(y) for y in reviews["review_year"].dropna().unique() if int(y) > 0])
    min_year, max_year = min(years), max(years)
    year_range = st.slider("Review year range", min_year, max_year, (min_year, max_year))
    ratings = st.multiselect("Ratings", options=[1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5])
    verified_filter = st.selectbox("Verified purchase", ["All", "Verified only", "Non-verified only"])
    min_helpful = st.slider("Minimum helpful votes", 0, int(reviews["helpful_vote"].quantile(0.99)), 0)
    review_text_only = st.checkbox("Only reviews with text", value=False)

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

# ---------- Tabs ----------
overview_tab, reviews_tab, products_tab, users_tab, dictionary_tab = st.tabs(
    ["Overview", "Reviews", "Products", "Users", "Data dictionary"]
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

    bottom_left, bottom_right = st.columns(2)

    with bottom_left:
        vp = (
            filtered_reviews.groupby(["review_year", "verified_purchase"], as_index=False)
            .size()
            .rename(columns={"size": "reviews"})
        )
        vp["verified_purchase"] = vp["verified_purchase"].map({True: "Verified", False: "Non-verified"})
        fig = px.bar(
            vp,
            x="review_year",
            y="reviews",
            color="verified_purchase",
            barmode="stack",
            title="Verified vs non-verified reviews over time",
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    with bottom_right:
        helpful_bins = pd.cut(
            filtered_reviews["helpful_vote"],
            bins=[-1, 0, 1, 5, 10, 25, 50, 100, np.inf],
            labels=["0", "1", "2-5", "6-10", "11-25", "26-50", "51-100", "100+"],
        )
        helpful_df = helpful_bins.value_counts().sort_index().reset_index()
        helpful_df.columns = ["helpful_votes", "reviews"]
        fig = px.bar(helpful_df, x="helpful_votes", y="reviews", title="Helpful-vote distribution")
        fig.update_layout(height=420, xaxis_title="Helpful votes")
        st.plotly_chart(fig, use_container_width=True)

with reviews_tab:
    section_header("Review-level exploration")
    r1, r2 = st.columns(2)

    with r1:
        by_month = (
            filtered_reviews.groupby("review_year_month", as_index=False)
            .size()
            .rename(columns={"size": "reviews"})
            .sort_values("review_year_month")
        )
        fig = px.line(by_month, x="review_year_month", y="reviews", markers=True, title="Reviews per month")
        fig.update_layout(height=420, xaxis_title="Year-month")
        st.plotly_chart(fig, use_container_width=True)

    with r2:
        rating_by_year = (
            filtered_reviews.groupby("review_year", as_index=False)["rating"]
            .mean()
            .rename(columns={"rating": "avg_rating"})
        )
        fig = px.line(rating_by_year, x="review_year", y="avg_rating", markers=True, title="Average rating by year")
        fig.update_layout(height=420, yaxis_range=[1, 5])
        st.plotly_chart(fig, use_container_width=True)

    r3, r4 = st.columns(2)
    with r3:
        fig = px.histogram(
            filtered_reviews,
            x="review_length_words",
            nbins=50,
            title="Review length distribution (words)",
        )
        fig.update_layout(height=420, xaxis_title="Words")
        st.plotly_chart(fig, use_container_width=True)

    with r4:
        stats = filtered_reviews.groupby("rating", as_index=False).agg(
            avg_words=("review_length_words", "mean"),
            avg_helpful=("helpful_vote", "mean"),
        )
        fig = px.scatter(
            stats,
            x="avg_words",
            y="avg_helpful",
            size=[18] * len(stats),
            text="rating",
            title="By rating: average review length vs helpful votes",
        )
        fig.update_traces(textposition="top center")
        fig.update_layout(height=420, xaxis_title="Avg review length (words)", yaxis_title="Avg helpful votes")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Filtered review summary table")
    review_summary = (
        filtered_reviews.groupby("rating", as_index=False)
        .agg(
            reviews=("rating", "size"),
            avg_helpful_vote=("helpful_vote", "mean"),
            avg_length_words=("review_length_words", "mean"),
            verified_ratio=("verified_purchase", "mean"),
        )
        .sort_values("rating")
    )
    review_summary["verified_ratio"] = review_summary["verified_ratio"].map(lambda x: round(x * 100, 2))
    st.dataframe(review_summary, use_container_width=True, hide_index=True)

with products_tab:
    section_header("Product exploration")

    product_counts = (
        filtered_reviews.groupby("parent_asin", as_index=False)
        .size()
        .rename(columns={"size": "filtered_review_count"})
        .merge(products_lookup, on="parent_asin", how="left")
        .sort_values(["filtered_review_count", "average_rating"], ascending=[False, False])
    )

    p1, p2 = st.columns(2)
    with p1:
        top_n = st.slider("Top products to show", 10, 100, 25, key="top_products_n")
        fig = px.bar(
            product_counts.head(top_n).sort_values("filtered_review_count"),
            x="filtered_review_count",
            y="title",
            orientation="h",
            hover_data=["parent_asin", "store_clean", "average_rating", "price"],
            title=f"Top {top_n} products by filtered review count",
        )
        fig.update_layout(height=700, yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with p2:
        product_scatter = products[(products["rating_number"] > 0) & products["average_rating"].notna()].copy()
        product_scatter["price_bucket"] = np.where(product_scatter["price"].notna(), "Has price", "No price")
        fig = px.scatter(
            product_scatter,
            x="rating_number",
            y="average_rating",
            color="price_bucket",
            hover_data=["title", "store_clean", "price", "parent_asin"],
            log_x=True,
            title="Products: rating count vs average rating",
        )
        fig.update_layout(height=700, xaxis_title="Number of ratings (log scale)", yaxis_range=[1, 5])
        st.plotly_chart(fig, use_container_width=True)

    p3, p4 = st.columns(2)
    with p3:
        store_counts = (
            products.groupby("store_clean", as_index=False)
            .agg(products=("parent_asin", "nunique"), avg_rating=("average_rating", "mean"))
            .query("store_clean != '(missing store)'")
            .sort_values("products", ascending=False)
            .head(20)
        )
        fig = px.bar(store_counts.sort_values("products"), x="products", y="store_clean", orientation="h", title="Top stores by number of products")
        fig.update_layout(height=500, yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with p4:
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

    year_counts = (
        products.dropna(subset=["year_first_available"])
        .assign(year_first_available=lambda d: d["year_first_available"].astype(int))
        .groupby("year_first_available", as_index=False)
        .size()
        .rename(columns={"size": "products"})
    )
    fig = px.bar(year_counts, x="year_first_available", y="products", title="Products by first-available year")
    fig.update_layout(height=420, xaxis_title="First available year")
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

    uleft, uright = st.columns(2)

    with uleft:
        curve = cumulative_share_curve(counts_series, "user")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=curve["user_pct"], y=curve["review_pct"], mode="lines", name="Actual"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect equality"))
        fig.update_layout(
            title="Cumulative share of reviews by cumulative share of users",
            height=450,
            xaxis_title="Share of users",
            yaxis_title="Share of reviews",
        )
        st.plotly_chart(fig, use_container_width=True)

    with uright:
        review_bins = pd.cut(
            counts_series,
            bins=[0, 1, 2, 3, 5, 10, 20, 50, np.inf],
            labels=["1", "2", "3", "4-5", "6-10", "11-20", "21-50", "51+"],
            include_lowest=True,
        )
        bin_df = review_bins.value_counts().sort_index().reset_index()
        bin_df.columns = ["reviews_written", "users"]
        fig = px.bar(bin_df, x="reviews_written", y="users", title="Active users by number of filtered reviews")
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(
        user_counts.head(2000),
        x="filtered_review_count",
        y="mean_rating_given",
        size="avg_review_length_words" if "avg_review_length_words" in user_counts.columns else None,
        hover_data=["user_id", "unique_products_reviewed", "verified_purchase_ratio", "mean_helpful_vote_received"],
        title="Top active users: review count vs mean rating",
    )
    fig.update_layout(height=500, xaxis_title="Filtered review count", yaxis_range=[1, 5])
    st.plotly_chart(fig, use_container_width=True)

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

with dictionary_tab:
    section_header("Preview and schema")
    d1, d2, d3 = st.columns(3)

    with d1:
        st.markdown("#### Reviews file")
        st.dataframe(schema_preview(reviews_source), use_container_width=True, hide_index=True)
        st.write(f"Columns loaded in dashboard: {', '.join(reviews.columns)}")

    with d2:
        st.markdown("#### Products file")
        st.dataframe(schema_preview(products_source), use_container_width=True, hide_index=True)
        st.write(f"Columns loaded in dashboard: {', '.join(products.columns)}")

    with d3:
        st.markdown("#### Users file")
        st.dataframe(schema_preview(users_source), use_container_width=True, hide_index=True)
        st.write(f"Columns loaded in dashboard: {', '.join(users.columns)}")

st.caption(
    "Tip: put dashboard_app.py in the same folder as the three CSVs, then run `streamlit run dashboard_app.py`."
)
