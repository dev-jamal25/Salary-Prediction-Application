"""
Data access layer for Streamlit dashboard.

Provides simple Supabase queries for:
- Latest and historical runs
- Run predictions
- LLM analysis
"""

import logging
from typing import List, Dict, Any, Optional
import pandas as pd

from app.storage.supabase_client import get_client

logger = logging.getLogger(__name__)


def get_all_runs() -> pd.DataFrame:
    """
    Fetch all prediction runs, ordered by most recent first.
    
    Returns:
        DataFrame with columns: id, run_name, model_name, model_version, 
                                status, scenario_count, created_at, notes
    """
    try:
        client = get_client()
        response = client.table("prediction_runs").select(
            "id, run_name, model_name, model_version, status, scenario_count, created_at, notes"
        ).order("created_at", desc=True).execute()
        
        if response.data:
            df = pd.DataFrame(response.data)
            df['created_at'] = pd.to_datetime(df['created_at'])
            logger.info(f"Fetched {len(df)} runs from Supabase")
            return df
        else:
            logger.info("No runs found in Supabase")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Failed to fetch runs: {str(e)}")
        return pd.DataFrame()


def get_run_by_id(run_id: int) -> Optional[Dict[str, Any]]:
    """
    Fetch a specific run by ID.
    
    Args:
        run_id: The run ID
    
    Returns:
        Run record dict or None if not found
    """
    try:
        client = get_client()
        response = client.table("prediction_runs").select("*").eq("id", run_id).execute()
        
        if response.data:
            return response.data[0]
        return None
    except Exception as e:
        logger.error(f"Failed to fetch run {run_id}: {str(e)}")
        return None


def get_predictions_for_run(run_id: int) -> pd.DataFrame:
    """
    Fetch all predictions for a run.
    
    Returns:
        DataFrame with all prediction columns
    """
    try:
        client = get_client()
        response = client.table("predictions").select("*").eq("run_id", run_id).execute()
        
        if response.data:
            df = pd.DataFrame(response.data)
            df['created_at'] = pd.to_datetime(df['created_at'])
            logger.info(f"Fetched {len(df)} predictions for run {run_id}")
            return df
        else:
            logger.info(f"No predictions found for run {run_id}")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Failed to fetch predictions for run {run_id}: {str(e)}")
        return pd.DataFrame()


def get_analysis_for_run(run_id: int) -> Optional[Dict[str, Any]]:
    """
    Fetch the LLM analysis for a run.
    
    Returns:
        Analysis record dict or None if not found
    """
    try:
        client = get_client()
        response = client.table("llm_analyses").select("*").eq("run_id", run_id).execute()
        
        if response.data:
            return response.data[0]
        return None
    except Exception as e:
        logger.error(f"Failed to fetch analysis for run {run_id}: {str(e)}")
        return None
