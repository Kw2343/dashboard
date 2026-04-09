from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

TOP_ORDER = ['Top1','Top2','Top3','Top4','Top5']

def load_scatter_data(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        st.error("Scatter Excel file not found.")
        st.stop()

    df = pd.read_excel(file_path)

    # Standardize column names
    df = df.rename(columns={
        "X_MaxCosSim": "MaxCosine",
        "Y_PredRating": "Predicted_Rating"
    })

    required = ["DisplayLabel", "Group", "MaxCosine", "Predicted_Rating"]
    for col in required:
        if col not in df.columns:
            st.error(f"Missing column: {col}")
            st.stop()

    return df.dropna()


def create_scatter_plot(df: pd.DataFrame):

    top = df[df["Group"].isin(TOP_ORDER)].copy()
    near = df[df["Group"] == "Near"]
    far = df[df["Group"] == "Far"]
    random_pts = df[df["Group"] == "Random"]

    # Sort Top 5
    top["order"] = top["Group"].map({g: i for i, g in enumerate(TOP_ORDER)})
    top = top.sort_values("order")

    fig = go.Figure()

    # Random
    fig.add_trace(go.Scatter(
        x=random_pts["MaxCosine"],
        y=random_pts["Predicted_Rating"],
        mode="markers",
        name="Random",
        marker=dict(size=6, color="rgba(0,255,255,0.4)"),
        hoverinfo="skip"
    ))

    # Top 5
    fig.add_trace(go.Scatter(
        x=top["MaxCosine"],
        y=top["Predicted_Rating"],
        mode="lines+markers+text",
        text=top["DisplayLabel"],
        textposition="top center",
        name="Top 5",
        marker=dict(size=12, color="blue"),
        line=dict(width=3),
        hovertemplate="(%{x:.2f}, %{y:.2f})<extra></extra>"
    ))

    # Near
    fig.add_trace(go.Scatter(
        x=near["MaxCosine"],
        y=near["Predicted_Rating"],
        mode="markers",
        text=near["DisplayLabel"],
        name="Near",
        marker=dict(size=12, color="green"),
        hovertemplate="(%{text}<br>(%{x:.2f}, %{y:.2f}))"
    ))

    # Far
    fig.add_trace(go.Scatter(
        x=far["MaxCosine"],
        y=far["Predicted_Rating"],
        mode="markers",
        text=far["DisplayLabel"],
        name="Far",
        marker=dict(size=12, color="red"),
        hovertemplate="(%{text}<br>(%{x:.2f}, %{y:.2f}))"
    ))

    fig.update_layout(
        title="Recommendation Scatter Plot",
        xaxis_title="Cosine Similarity",
        yaxis_title="Predicted Rating",
        height=650
    )

    return fig