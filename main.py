import os
import glob
from typing import List, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from collections import defaultdict

RESULT_COLS = [
    "Open P&L",
    "Open P&L %",
    "Net profit: All",
    "Net profit %: All",
    "Net profit: Long",
    "Net profit %: Long",
    "Net profit: Short",
    "Net profit %: Short",
    "Gross profit: All",
    "Gross profit %: All",
    "Gross profit: Long",
    "Gross profit %: Long",
    "Gross profit: Short",
    "Gross profit %: Short",
    "Gross loss: All",
    "Gross loss %: All",
    "Gross loss: Long",
    "Gross loss %: Long",
    "Gross loss: Short",
    "Gross loss %: Short",
    "Commission paid: All",
    "Commission paid: Long",
    "Commission paid: Short",
    "Buy & hold return",
    "Buy & hold return %",
    "Max equity run-up",
    "Max equity run-up %",
    "Max equity drawdown",
    "Max equity drawdown %",
    "Max contracts held: All",
    "Max contracts held: Long",
    "Max contracts held: Short",
    "Total trades: All",
    "Total trades: Long",
    "Total trades: Short",
    "Total open trades: All",
    "Total open trades: Long",
    "Total open trades: Short",
    "Winning trades: All",
    "Winning trades: Long",
    "Winning trades: Short",
    "Losing trades: All",
    "Losing trades: Long",
    "Losing trades: Short",
    "Percent profitable: All",
    "Percent profitable: Long",
    "Percent profitable: Short",
    "Avg P&L: All",
    "Avg P&L %: All",
    "Avg P&L: Long",
    "Avg P&L %: Long",
    "Avg P&L: Short",
    "Avg winning trade: All",
    "Avg winning trade %: All",
    "Avg winning trade: Long",
    "Avg winning trade %: Long",
    "Avg winning trade: Short",
    "Avg losing trade: All",
    "Avg losing trade %: All",
    "Avg losing trade: Long",
    "Avg losing trade %: Long",
    "Avg losing trade: Short",
    "Ratio avg win / avg loss: All",
    "Ratio avg win / avg loss: Long",
    "Ratio avg win / avg loss: Short",
    "Largest winning trade: All",
    "Largest winning trade: Long",
    "Largest winning trade: Short",
    "Largest winning trade percent: All",
    "Largest winning trade percent: Long",
    "Largest winning trade percent: Short",
    "Largest losing trade: All",
    "Largest losing trade: Long",
    "Largest losing trade: Short",
    "Largest losing trade percent: All",
    "Largest losing trade percent: Long",
    "Largest losing trade percent: Short",
    "Avg # bars in trades: All",
    "Avg # bars in trades: Long",
    "Avg # bars in trades: Short",
    "Avg # bars in winning trades: All",
    "Avg # bars in winning trades: Long",
    "Avg # bars in winning trades: Short",
    "Avg # bars in losing trades: All",
    "Avg # bars in losing trades: Long",
    "Avg # bars in losing trades: Short",
    "Sharpe ratio",
    "Sortino ratio",
    "Profit factor: All",
    "Profit factor: Long",
    "Profit factor: Short",
    "Margin calls: All",
    "Margin calls: Long",
    "Margin calls: Short"
]

def find_default_csv_file(base_dir: str) -> str | None:
    backtest_dir = os.path.join(base_dir, "BackTestResults")
    pattern = os.path.join(backtest_dir, "*.csv")
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

@st.cache_data(show_spinner=False)
def load_results(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        na_values=["∅", "NA", "NaN", "nan", "None", "null", ""],
        keep_default_na=True,
        low_memory=False,
    )
    for col in df.columns:
        if df[col].dtype == object:
            coerced = pd.to_numeric(df[col], errors="coerce")
            if coerced.notna().sum() > 0:
                df[col] = coerced
    return df

def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

def compute_pareto_front(
    df: pd.DataFrame,
    obj_cols: List[str],
    maximize_flags: List[bool],
) -> pd.Series:
    """Return a boolean Series indicating whether each row is on the N-dimensional Pareto front.

    Dominance rule:
    - A point A dominates B if A is >= (or <=) in all objectives according to direction
      and strictly better in at least one objective.
    """
    if df.empty or not obj_cols:
        return pd.Series([], dtype=bool)

    # Extract objective values and normalize directions (always maximize)
    points = []
    for col, maximize in zip(obj_cols, maximize_flags):
        values = df[col].to_numpy()
        points.append(values if maximize else -values)
    points = np.column_stack(points)

    n_points = points.shape[0]
    is_pareto = np.ones(n_points, dtype=bool)

    # Vectorized Pareto dominance check
    for i in range(n_points):
        if not is_pareto[i]:
            continue
        # Compare point i with all others
        diff = points - points[i]
        # Dominated if another point is >= in all objectives and > in at least one
        dominated = np.all(diff >= 0, axis=1) & np.any(diff > 0, axis=1)
        dominated[i] = False  # A point does not dominate itself
        if np.any(dominated):
            is_pareto[i] = False

    return pd.Series(is_pareto, index=df.index)

def sorted_pareto_df(
    df: pd.DataFrame,
    obj_cols: List[str],
    maximize_flags: List[bool],
    pareto_mask: pd.Series,
) -> pd.DataFrame:
    p = df.loc[pareto_mask].copy()
    # Sort by all objectives based on directions
    p.sort_values(
        by=obj_cols,
        ascending=[not m for m in maximize_flags],
        inplace=True,
    )
    return p

def build_scatter_chart(
    df: pd.DataFrame,
    obj_cols: List[str],
    pareto_mask: pd.Series,
) -> go.Figure:
    plot_df = df.copy()
    plot_df["Pareto"] = np.where(pareto_mask, "Pareto front", "Interior")

    # Prepare hover text with all numeric columns
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    # hover_template = "<br>".join([f"{col}: %{{customdata[{i}]}}" for i, col in enumerate(numeric_cols)])
    # customdata = plot_df[numeric_cols].values
    hover_template = "<br>".join([f"{col}: %{{customdata[{i}]}}" for i, col in enumerate(obj_cols)])
    customdata = plot_df[obj_cols].values

    if len(obj_cols) == 2:
        # 2D Scatter Plot
        fig = go.Figure()
        # Interior points
        interior = plot_df[plot_df["Pareto"] == "Interior"]
        fig.add_trace(
            go.Scatter(
                x=interior[obj_cols[0]],
                y=interior[obj_cols[1]],
                mode="markers",
                marker=dict(size=8, color="#457b9d", opacity=0.6),
                name="Interior",
                customdata=customdata[plot_df["Pareto"] == "Interior"],
                hovertemplate=hover_template,
            )
        )
        # Pareto front points
        pareto = plot_df[plot_df["Pareto"] == "Pareto front"]
        fig.add_trace(
            go.Scatter(
                x=pareto[obj_cols[0]],
                y=pareto[obj_cols[1]],
                mode="markers",
                marker=dict(size=10, color="#e63946", opacity=0.9),
                name="Pareto front",
                customdata=customdata[plot_df["Pareto"] == "Pareto front"],
                hovertemplate=hover_template,
            )
        )
        fig.update_layout(
            xaxis_title=obj_cols[0],
            yaxis_title=obj_cols[1],
            showlegend=True,
            height=600,
        )
    elif len(obj_cols) == 3:
        # 3D Scatter Plot
        fig = go.Figure()
        # Interior points
        interior = plot_df[plot_df["Pareto"] == "Interior"]
        fig.add_trace(
            go.Scatter3d(
                x=interior[obj_cols[0]],
                y=interior[obj_cols[1]],
                z=interior[obj_cols[2]],
                mode="markers",
                marker=dict(size=5, color="#457b9d", opacity=0.6),
                name="Interior",
                customdata=customdata[plot_df["Pareto"] == "Interior"],
                hovertemplate=hover_template,
            )
        )
        # Pareto front points
        pareto = plot_df[plot_df["Pareto"] == "Pareto front"]
        fig.add_trace(
            go.Scatter3d(
                x=pareto[obj_cols[0]],
                y=pareto[obj_cols[1]],
                z=pareto[obj_cols[2]],
                mode="markers",
                marker=dict(size=7, color="#e63946", opacity=0.9),
                name="Pareto front",
                customdata=customdata[plot_df["Pareto"] == "Pareto front"],
                hovertemplate=hover_template,
            )
        )
        fig.update_layout(
            scene=dict(
                xaxis_title=obj_cols[0],
                yaxis_title=obj_cols[1],
                zaxis_title=obj_cols[2],
            ),
            showlegend=True,
            height=600,
        )
    else:
        # Table for 4 or 5 objectives
        st.dataframe(plot_df)
        return None

    return fig

def main() -> None:
    st.set_page_config(
        page_title="MO",
        page_icon="📊",
        layout="wide"
    )
    st.title("Multi-objective Results Explorer (Pareto Front)")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_dir = os.path.join(base_dir, "BackTestResults")
    default_csv = find_default_csv_file(base_dir)

    csv_pattern = os.path.join(csv_dir, "*.csv")
    csv_files = glob.glob(csv_pattern)
    csv_files.sort(key=os.path.getmtime, reverse=True)
    if not csv_files:
        st.error(f"No CSV files found in {csv_dir}. Please add results.")
        st.stop()

    with st.sidebar:
        st.header("Data")
        labels = [os.path.basename(p) for p in csv_files]
        selected_idx = 0
        if default_csv in csv_files:
            selected_idx = csv_files.index(default_csv)
        selected_label = st.selectbox("Select CSV in BackTestResults/", labels, index=selected_idx)
        selected_path = csv_files[labels.index(selected_label)]

        st.header("Objectives")
        num_objectives = st.number_input(
            "Number of objectives",
            min_value=2,
            max_value=5,
            value=2,
            step=1
        )

    csv_source = selected_path
    if isinstance(csv_source, str) and not os.path.exists(csv_source):
        st.error("The specified CSV path does not exist.")
        st.stop()

    df = load_results(csv_source)
    numeric_cols = get_numeric_columns(df)
    numeric_cols = [col for col in numeric_cols if col in RESULT_COLS]
    if len(numeric_cols) < 2:
        st.error("Need at least two numeric columns to plot objectives.")
        st.stop()

    with st.sidebar:
        st.subheader("Select Objectives")
        selected_cols = []
        maximize_flags = []
        for i in range(num_objectives):
            # Filter out already selected columns
            available_cols = [col for col in numeric_cols if col not in selected_cols]
            default_idx = min(i, len(available_cols) - 1)
            col = st.selectbox(
                f"Objective {i+1}",
                available_cols,
                index=default_idx,
                key=f"obj_{i}"
            )
            selected_cols.append(col)
            direction = st.radio(
                f"Direction for {col}",
                ["Maximize", "Minimize"],
                horizontal=True,
                index=0,
                key=f"dir_{i}"
            )
            maximize_flags.append(direction == "Maximize")
        st.caption("Tip: examples include 'Net profit: All', 'Margin calls: Short'.")

        show_labels = True
        download_enabled = True

    # Drop rows with NaN in any selected objectives
    work_df = df[selected_cols].dropna()
    mask_valid = df[selected_cols].notna().all(axis=1)
    valid_df = df.loc[mask_valid].copy()

    pareto_mask_valid = compute_pareto_front(valid_df, selected_cols, maximize_flags)
    pareto_mask_full = pd.Series(False, index=df.index)
    pareto_mask_full.loc[valid_df.index] = pareto_mask_valid

    left, right = st.columns([2, 1])
    with left:
        st.subheader("Visualization: All points with Pareto front highlighted")
        fig = build_scatter_chart(valid_df, selected_cols, pareto_mask_valid)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Pareto front size")
        st.metric("Points on front", int(pareto_mask_valid.sum()))
        st.metric("Total valid points", int(len(valid_df)))

    st.divider()
    st.subheader("Pareto front details")
    pareto_df = sorted_pareto_df(valid_df, selected_cols, maximize_flags, pareto_mask_valid)

    # if show_labels:
    pareto_dict = {}
    unique_keys = defaultdict(int)
    for idx, row in pareto_df.iterrows():
        key = ", ".join([f"{col}:{row[col]}" for col in selected_cols])
        row_dict = row.to_dict()
        row_dict = {k: v for k, v in row_dict.items() if k not in RESULT_COLS}
        pareto_dict[key] = row_dict
        unique_keys[key] += 1

    st.markdown("**Duplicate Results with same parameters:**")
    for key, count in unique_keys.items():
        if count > 1:
            st.markdown(f"  - {key} (x{count} paremeter sets)")

    selected_key = st.selectbox("Select a Pareto point", pareto_dict.keys())
    selected_row = pareto_dict[selected_key]
    st.write(selected_row)
    
    # Convert selected_row to DataFrame for display
    selected_row_df = pd.DataFrame([selected_row])

    # if download_enabled and not selected_row_df.empty:
    if not selected_row_df.empty:
        csv_bytes = selected_row_df.to_csv(index=False).encode("utf-8")
        filename = f"{Path(selected_path).stem}_pareto_front.csv"
        st.download_button(
            label="Download Pareto CSV",
            data=csv_bytes,
            file_name=filename,
            mime="text/csv",
        )

if __name__ == "__main__":
    main()