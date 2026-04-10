"""
Chart rendering for Streamlit dashboard.

Renders charts from stored chart_spec_json using prediction data.
Applies friendly labels to categorical axes at the display layer only.
Supports: bar, line, scatter, boxplot, histogram.
"""

import logging
from typing import Dict, Any, Optional
import pandas as pd
import streamlit as st
import altair as alt

logger = logging.getLogger(__name__)

# Friendly label mappings for categorical fields (display only, no DB changes)
CATEGORICAL_LABELS = {
    "experience_level": {
        "EN": "Junior",
        "MI": "Intermediate",
        "SE": "Senior",
        "EX": "Executive",
    },
    "employment_type": {
        "FT": "Full-time",
        "PT": "Part-time",
        "CT": "Contract",
        "FL": "Freelance",
    },
    "company_size": {
        "S": "Small",
        "M": "Medium",
        "L": "Large",
    },
    "remote_ratio": {
        0: "On-site",
        50: "Hybrid",
        100: "Remote",
    },
}


def render_chart(
    chart_spec: Optional[Dict[str, Any]],
    predictions_df: pd.DataFrame
) -> None:
    """
    Render a chart using stored chart spec and prediction data.
    
    Applies friendly labels to categorical x-axis fields (display only).
    Supports: bar, line, scatter, boxplot, histogram.
    
    Args:
        chart_spec: Chart specification dict with keys:
                    - chart_type: str (e.g., "bar", "line", "scatter", "boxplot", "histogram")
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
        chart_type = chart_spec.get("chart_type", "bar").lower()
        x_col = chart_spec.get("x")
        y_col = chart_spec.get("y")
        title = chart_spec.get("title", "Salary Analysis")
        aggregation = chart_spec.get("aggregation", "mean")
        
        if not x_col or not y_col:
            st.warning("⚠️ Chart specification incomplete (missing x or y column).")
            return
        
        # Check if columns exist
        if x_col not in predictions_df.columns or y_col not in predictions_df.columns:
            st.warning(
                f"⚠️ Chart columns not found in predictions. Expected: {x_col}, {y_col}"
            )
            return
        
        # Validate numeric y column for aggregation-based charts
        if chart_type in ["bar", "line", "boxplot"] and not pd.api.types.is_numeric_dtype(predictions_df[y_col]):
            st.warning(f"⚠️ Chart type '{chart_type}' requires numeric y column. '{y_col}' is not numeric.")
            return
        
        # Prepare chart data
        if chart_type == "bar":
            # Aggregate data for bar chart
            if aggregation in ["mean", "median", "min", "max", "sum"]:
                chart_data = predictions_df.groupby(x_col)[y_col].agg(aggregation).reset_index()
            else:
                chart_data = predictions_df.groupby(x_col).size().reset_index(name=y_col)
            
            # Apply friendly labels to x-axis if categorical
            if x_col in CATEGORICAL_LABELS:
                label_map = CATEGORICAL_LABELS[x_col]
                chart_data[x_col] = chart_data[x_col].map(label_map).fillna(chart_data[x_col])
            
            x_title = x_col.replace("_", " ").title()
            y_title = f"{aggregation.capitalize()} {y_col.replace('_', ' ').title()}"
            
            chart = alt.Chart(chart_data).mark_bar().encode(
                x=alt.X(x_col, title=x_title),
                y=alt.Y(y_col, title=y_title),
                tooltip=[x_col, y_col]
            ).properties(
                title=title,
                width=700,
                height=400
            ).interactive()
        
        elif chart_type == "line":
            # Aggregate data for line chart
            if aggregation in ["mean", "median", "min", "max", "sum"]:
                chart_data = predictions_df.groupby(x_col)[y_col].agg(aggregation).reset_index()
            else:
                chart_data = predictions_df.groupby(x_col).size().reset_index(name=y_col)
            
            # Apply friendly labels
            if x_col in CATEGORICAL_LABELS:
                label_map = CATEGORICAL_LABELS[x_col]
                chart_data[x_col] = chart_data[x_col].map(label_map).fillna(chart_data[x_col])
            
            x_title = x_col.replace("_", " ").title()
            y_title = f"{aggregation.capitalize()} {y_col.replace('_', ' ').title()}"
            
            chart = alt.Chart(chart_data).mark_line(point=True).encode(
                x=alt.X(x_col, title=x_title),
                y=alt.Y(y_col, title=y_title),
                tooltip=[x_col, y_col]
            ).properties(
                title=title,
                width=700,
                height=400
            ).interactive()
        
        elif chart_type == "scatter":
            # Scatter plot using raw data
            x_title = x_col.replace("_", " ").title()
            y_title = y_col.replace("_", " ").title()
            
            chart = alt.Chart(predictions_df).mark_point(opacity=0.6).encode(
                x=alt.X(x_col, title=x_title),
                y=alt.Y(y_col, title=y_title),
                tooltip=[x_col, y_col],
                color=alt.value("steelblue")
            ).properties(
                title=title,
                width=700,
                height=400
            ).interactive()
        
        elif chart_type == "boxplot":
            # Box plot for distribution by category
            if x_col in CATEGORICAL_LABELS:
                # Make a copy and apply labels for display
                plot_data = predictions_df.copy()
                label_map = CATEGORICAL_LABELS[x_col]
                plot_data[f"{x_col}_labeled"] = plot_data[x_col].map(label_map).fillna(plot_data[x_col])
                x_col_display = f"{x_col}_labeled"
            else:
                plot_data = predictions_df
                x_col_display = x_col
            
            x_title = x_col.replace("_", " ").title()
            y_title = y_col.replace("_", " ").title()
            
            chart = alt.Chart(plot_data).mark_boxplot().encode(
                x=alt.X(x_col_display, title=x_title),
                y=alt.Y(y_col, title=y_title),
                tooltip=[x_col_display, y_col]
            ).properties(
                title=title,
                width=700,
                height=400
            ).interactive()
        
        elif chart_type == "histogram":
            # Histogram for distribution
            y_title = f"Count of {y_col.replace('_', ' ').title()}"
            
            chart = alt.Chart(predictions_df).mark_bar().encode(
                x=alt.X(y_col, bin=alt.Bin(maxbins=30), title=y_col.replace("_", " ").title()),
                y=alt.Y("count()", title=y_title),
                tooltip=[y_col, "count()"]
            ).properties(
                title=title,
                width=700,
                height=400
            ).interactive()
        
        else:
            st.warning(
                f"⚠️ Unsupported chart type: '{chart_type}'. "
                f"Supported types: bar, line, scatter, boxplot, histogram"
            )
            return
        
        st.altair_chart(chart, use_container_width=True)
        logger.info(f"Rendered {chart_type} chart: {x_col} vs {y_col}")
    
    except Exception as e:
        logger.error(f"Failed to render chart: {str(e)}")
        st.error(f"❌ Failed to render chart: {str(e)}")