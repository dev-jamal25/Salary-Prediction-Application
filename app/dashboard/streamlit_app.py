"""
Streamlit dashboard that reads from Supabase only.

Displays:
- Prediction history / runs
- Prediction results
- LLM narrative summary
- Key insights
- Reconstructed chart from stored spec
- Graceful empty states
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
import streamlit as st
import pandas as pd
import json
import logging
from datetime import datetime
from dotenv import load_dotenv

from app.dashboard.data_access import (
    get_all_runs,
    get_run_by_id,
    get_predictions_for_run,
    get_analysis_for_run,
)
from app.dashboard.charts import render_chart

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Salary Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Friendly label mappings (display only)
EXPERIENCE_LABELS = {
    "EN": "Junior",
    "MI": "Intermediate",
    "SE": "Senior",
    "EX": "Executive",
}

EMPLOYMENT_LABELS = {
    "FT": "Full-time",
    "PT": "Part-time",
    "CT": "Contract",
    "FL": "Freelance",
}

COMPANY_SIZE_LABELS = {
    "S": "Small",
    "M": "Medium",
    "L": "Large",
}

REMOTE_RATIO_LABELS = {
    0: "On-site",
    50: "Hybrid",
    100: "Remote",
}


def get_friendly_label(raw_value: str, label_map: dict) -> str:
    """Convert raw value to friendly label, fallback to raw value if not found."""
    return label_map.get(raw_value, str(raw_value))


def format_predictions_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Add friendly label columns for display."""
    df_display = df.copy()
    
    # Add friendly labels for categorical fields
    if "experience_level" in df_display.columns:
        df_display["Experience Level"] = df_display["experience_level"].apply(
            lambda x: get_friendly_label(x, EXPERIENCE_LABELS)
        )
    
    if "employment_type" in df_display.columns:
        df_display["Employment Type"] = df_display["employment_type"].apply(
            lambda x: get_friendly_label(x, EMPLOYMENT_LABELS)
        )
    
    if "company_size" in df_display.columns:
        df_display["Company Size"] = df_display["company_size"].apply(
            lambda x: get_friendly_label(x, COMPANY_SIZE_LABELS)
        )
    
    if "remote_ratio" in df_display.columns:
        df_display["Remote Work"] = df_display["remote_ratio"].apply(
            lambda x: get_friendly_label(x, REMOTE_RATIO_LABELS)
        )
    
    # Format salary with commas
    if "predicted_salary_usd" in df_display.columns:
        df_display["Predicted Salary"] = df_display["predicted_salary_usd"].apply(
            lambda x: f"${x:,.0f}"
        )
    
    return df_display


def main():
    """Main dashboard application."""
    st.title("📊 Salary Prediction Dashboard")
    st.markdown("View prediction results and analysis from completed pipeline runs.")
    
    # Sidebar for run selection
    st.sidebar.header("🔍 Run Selection")
    
    # Fetch all runs
    runs_df = get_all_runs()
    
    if runs_df.empty:
        st.error("❌ No prediction runs found in Supabase. Run the pipeline first.")
        return
    
    # Create run selector with latest run selected by default
    runs_df["display_name"] = runs_df.apply(
        lambda row: f"{row['run_name']} ({row['created_at'].strftime('%Y-%m-%d %H:%M')})",
        axis=1
    )
    
    selected_idx = 0  # Default to latest (first row after ordering)
    run_display_names = runs_df["display_name"].tolist()
    selected_display = st.sidebar.selectbox(
        "Select a run:",
        options=run_display_names,
        index=selected_idx,
        help="Choose a completed pipeline run to view results"
    )
    
    # Get selected run ID
    selected_run_idx = run_display_names.index(selected_display)
    selected_run_id = runs_df.iloc[selected_run_idx]["id"]
    selected_run_data = get_run_by_id(selected_run_id)
    
    if not selected_run_data:
        st.error("❌ Could not load selected run data.")
        return
    
    # Add refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()
    
    # ========================================================================
    # Section 1: Run Metadata
    # ========================================================================
    st.header("📋 Run Metadata")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Run ID", selected_run_id)
        st.metric("Status", selected_run_data.get("status", "N/A").upper())
    with col2:
        st.metric("Model", selected_run_data.get("model_name", "N/A"))
        st.metric("Version", selected_run_data.get("model_version", "N/A"))
    with col3:
        st.metric("Scenarios", selected_run_data.get("scenario_count", 0))
        created_at = selected_run_data.get("created_at", "N/A")
        if created_at != "N/A":
            try:
                created_dt = pd.to_datetime(created_at)
                st.metric("Created", created_dt.strftime("%Y-%m-%d %H:%M"))
            except:
                st.metric("Created", created_at)
    
    # Show notes if present
    notes = selected_run_data.get("notes")
    if notes:
        st.info(f"📝 Notes: {notes}")
    
    # ========================================================================
    # Section 2: LLM Analysis
    # ========================================================================
    st.header("📖 LLM Analysis")
    
    analysis = get_analysis_for_run(selected_run_id)
    
    if analysis:
        # Title and summary
        st.subheader(analysis.get("title", "Analysis"))
        st.write(analysis.get("summary", "No summary available."))
        
        # Key insights
        key_insights = analysis.get("key_insights_json", [])
        if key_insights:
            st.markdown("**Key Insights:**")
            for insight in key_insights:
                st.markdown(f"- {insight}")
        
        # Show analysis metadata
        with st.expander("ℹ️ Analysis Details"):
            analysis_created = analysis.get("created_at", "N/A")
            st.write(f"**Analysis created:** {analysis_created}")
    else:
        st.info("No LLM analysis available for this run yet.")
    
    # ========================================================================
    # Section 3: Predictions Table
    # ========================================================================
    st.header("📊 Predictions")
    
    predictions_df = get_predictions_for_run(selected_run_id)
    
    if predictions_df.empty:
        st.warning("No predictions found for this run.")
    else:
        st.markdown(f"**Total predictions:** {len(predictions_df)}")
        
        # Filters
        st.subheader("Filters")
        col1, col2, col3 = st.columns(3)
        
        filters = {}
        
        if "experience_level" in predictions_df.columns:
            with col1:
                unique_exp = sorted(predictions_df["experience_level"].unique())
                selected_exp = st.multiselect(
                    "Experience Level",
                    options=unique_exp,
                    default=unique_exp,
                    format_func=lambda x: get_friendly_label(x, EXPERIENCE_LABELS)
                )
                filters["experience_level"] = selected_exp
        
        if "employment_type" in predictions_df.columns:
            with col2:
                unique_emp = sorted(predictions_df["employment_type"].unique())
                selected_emp = st.multiselect(
                    "Employment Type",
                    options=unique_emp,
                    default=unique_emp,
                    format_func=lambda x: get_friendly_label(x, EMPLOYMENT_LABELS)
                )
                filters["employment_type"] = selected_emp
        
        if "remote_ratio" in predictions_df.columns:
            with col3:
                unique_remote = sorted(predictions_df["remote_ratio"].unique())
                selected_remote = st.multiselect(
                    "Remote Work",
                    options=unique_remote,
                    default=unique_remote,
                    format_func=lambda x: get_friendly_label(x, REMOTE_RATIO_LABELS)
                )
                filters["remote_ratio"] = selected_remote
        
        if "job_title" in predictions_df.columns:
            unique_titles = sorted(predictions_df["job_title"].unique())
            selected_titles = st.multiselect(
                "Job Title",
                options=unique_titles,
                default=unique_titles,
                help="Leave empty to show all titles"
            )
            if selected_titles:
                filters["job_title"] = selected_titles
        
        if "company_location" in predictions_df.columns:
            unique_locations = sorted(predictions_df["company_location"].unique())
            selected_locations = st.multiselect(
                "Company Location",
                options=unique_locations,
                default=unique_locations,
                help="Leave empty to show all locations"
            )
            if selected_locations:
                filters["company_location"] = selected_locations
        
        if "company_size" in predictions_df.columns:
            unique_sizes = sorted(predictions_df["company_size"].unique())
            selected_sizes = st.multiselect(
                "Company Size",
                options=unique_sizes,
                default=unique_sizes,
                format_func=lambda x: get_friendly_label(x, COMPANY_SIZE_LABELS)
            )
            filters["company_size"] = selected_sizes
        
        # Apply filters
        filtered_df = predictions_df.copy()
        for col, values in filters.items():
            if col in filtered_df.columns and values:
                filtered_df = filtered_df[filtered_df[col].isin(values)]
        
        # Display table with friendly labels
        display_df = format_predictions_for_display(filtered_df)
        
        # Select columns to display
        display_cols = [
            "work_year", "Experience Level", "Employment Type", "job_title",
            "employee_residence", "Remote Work", "company_location", "Company Size",
            "Predicted Salary", "api_status"
        ]
        display_cols = [c for c in display_cols if c in display_df.columns]
        
        st.dataframe(
            display_df[display_cols],
            use_container_width=True,
            hide_index=True
        )
        
        # Summary statistics
        st.markdown("**Summary Statistics**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            avg_salary = filtered_df["predicted_salary_usd"].mean()
            st.metric("Average Salary", f"${avg_salary:,.0f}")
        with col2:
            min_salary = filtered_df["predicted_salary_usd"].min()
            st.metric("Min Salary", f"${min_salary:,.0f}")
        with col3:
            max_salary = filtered_df["predicted_salary_usd"].max()
            st.metric("Max Salary", f"${max_salary:,.0f}")
        with col4:
            st.metric("Filtered Count", len(filtered_df))
    
    # ========================================================================
    # Section 4: Chart
    # ========================================================================
    if analysis:
        st.header("📈 Visualization")
        
        chart_spec = analysis.get("chart_spec_json")
        if chart_spec:
            render_chart(chart_spec, predictions_df)
        else:
            st.info("No chart specification available for this run.")
    
    # ========================================================================
    # Footer
    # ========================================================================
    st.divider()
    st.caption("💡 This dashboard reads from Supabase. Data is refreshed on page reload or when you click the Refresh button.")


if __name__ == "__main__":
    main()
