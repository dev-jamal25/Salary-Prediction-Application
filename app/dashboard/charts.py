"""
Chart rendering for Streamlit dashboard.

Renders charts from stored chart_spec_json using prediction data.
"""

import logging
from typing import Dict, Any, Optional
import pandas as pd
import streamlit as st
import altair as alt

logger = logging.getLogger(__name__)


def render_chart(
    chart_spec: Optional[Dict[str, Any]],
    predictions_df: pd.DataFrame
) -> None:
    """
    Render a chart using stored chart spec and prediction data.
    
    Args:
        chart_spec: Chart specification dict with keys:
                    - chart_type: str (e.g., "bar", "line")
                    - x: str (column name)
                    - y: str (column name)
                    - title: str (optional)
                    - aggregation: str (optional, e.g., "mean", "median")
        predictions_df: DataFrame with prediction data
    """
    if chart_spec is None or predictions_df.empty:
        st.info("No chart data available for this run.")
        return
    
    try:
        chart_type = chart_spec.get("chart_type", "bar")
        x_col = chart_spec.get("x")
        y_col = chart_spec.get("y")
        title = chart_spec.get("title", "Salary Analysis")
        aggregation = chart_spec.get("aggregation", "mean")
        
        if not x_col or not y_col:
            st.warning("Chart specification incomplete (missing x or y column).")
            return
        
        # Check if columns exist
        if x_col not in predictions_df.columns or y_col not in predictions_df.columns:
            st.warning(f"Chart columns not found in data. Expected: {x_col}, {y_col}")
            return
        
        # Aggregate data for chart
        if aggregation in ["mean", "median", "min", "max", "sum"]:
            agg_func = aggregation if aggregation != "median" else "median"
            chart_data = predictions_df.groupby(x_col)[y_col].agg(agg_func).reset_index()
        else:
            chart_data = predictions_df.groupby(x_col).size().reset_index(name=y_col)
        
        # Render chart based on type
        if chart_type == "bar":
            chart = alt.Chart(chart_data).mark_bar().encode(
                x=alt.X(x_col, title=x_col.replace("_", " ").title()),
                y=alt.Y(y_col, title=y_col.replace("_", " ").title()),
                tooltip=[x_col, y_col]
            ).properties(
                title=title,
                width=700,
                height=400
            )
        elif chart_type == "line":
            chart = alt.Chart(chart_data).mark_line().encode(
                x=alt.X(x_col, title=x_col.replace("_", " ").title()),
                y=alt.Y(y_col, title=y_col.replace("_", " ").title()),
                tooltip=[x_col, y_col]
            ).properties(
                title=title,
                width=700,
                height=400
            )
        elif chart_type == "scatter":
            chart = alt.Chart(predictions_df).mark_point().encode(
                x=alt.X(x_col, title=x_col.replace("_", " ").title()),
                y=alt.Y(y_col, title=y_col.replace("_", " ").title()),
                tooltip=[x_col, y_col]
            ).properties(
                title=title,
                width=700,
                height=400
            )
        else:
            st.warning(f"Unsupported chart type: {chart_type}")
            return
        
        st.altair_chart(chart, use_container_width=True)
        logger.info(f"Rendered {chart_type} chart: {x_col} vs {y_col}")
    
    except Exception as e:
        logger.error(f"Failed to render chart: {str(e)}")
        st.error(f"Failed to render chart: {str(e)}")
