"""
Repository pattern for Supabase CRUD operations.

Handles persistence of:
1. prediction_runs - batch execution metadata
2. predictions - individual row-level predictions
3. llm_analyses - LLM narratives and chart specs

All functions use the configured Supabase client.
"""

import logging
from typing import List, Optional, Dict, Any
import json
from datetime import datetime

from app.storage.supabase_client import get_client

logger = logging.getLogger(__name__)


# ============================================================================
# Prediction Runs (batch execution metadata)
# ============================================================================

def create_run(
    model_name: str,
    model_version: str,
    scenario_count: int,
    run_name: Optional[str] = None,
    notes: Optional[str] = None,
    status: str = "pending"
) -> int:
    """
    Create a prediction run record.
    
    Args:
        model_name: Name of the model (e.g., 'DecisionTreeRegressor')
        model_version: Model version (e.g., 'v1')
        scenario_count: Number of prediction scenarios in this run
        run_name: Optional human-readable name
        notes: Optional notes about the run
        status: Initial status (default: 'pending')
    
    Returns:
        run_id (integer) - the ID of the created run
    
    Raises:
        Exception: If insert fails
    """
    client = get_client()
    
    payload = {
        "model_name": model_name,
        "model_version": model_version,
        "scenario_count": scenario_count,
        "run_name": run_name,
        "notes": notes,
        "status": status,
    }
    
    try:
        response = client.table("prediction_runs").insert(payload).execute()
        run_id = response.data[0]["id"]
        logger.info(f"Created prediction run: {run_id}")
        return run_id
    except Exception as e:
        logger.error(f"Failed to create prediction run: {e}")
        raise


def update_run_status(run_id: int, status: str) -> bool:
    """
    Update the status of a prediction run.
    
    Args:
        run_id: The run ID to update
        status: New status ('pending', 'completed', or 'failed')
    
    Returns:
        True if successful
    
    Raises:
        Exception: If update fails
    """
    client = get_client()
    
    try:
        response = client.table("prediction_runs").update(
            {"status": status}
        ).eq("id", run_id).execute()
        logger.info(f"Updated run {run_id} to status: {status}")
        return True
    except Exception as e:
        logger.error(f"Failed to update run {run_id} status: {e}")
        raise


def get_run(run_id: int) -> Optional[Dict[str, Any]]:
    """
    Fetch a prediction run by ID.
    
    Args:
        run_id: The run ID
    
    Returns:
        Run record dict or None if not found
    """
    client = get_client()
    
    try:
        response = client.table("prediction_runs").select("*").eq("id", run_id).execute()
        if response.data:
            return response.data[0]
        return None
    except Exception as e:
        logger.error(f"Failed to fetch run {run_id}: {e}")
        raise


def get_latest_run() -> Optional[Dict[str, Any]]:
    """
    Fetch the most recent prediction run.
    
    Returns:
        Latest run record dict or None if no runs exist
    """
    client = get_client()
    
    try:
        response = client.table("prediction_runs").select("*").order(
            "created_at", desc=True
        ).limit(1).execute()
        if response.data:
            return response.data[0]
        return None
    except Exception as e:
        logger.error(f"Failed to fetch latest run: {e}")
        raise


# ============================================================================
# Predictions (individual row-level predictions)
# ============================================================================

def insert_predictions(
    run_id: int,
    predictions: List[Dict[str, Any]]
) -> int:
    """
    Insert multiple predictions for a run.
    
    Args:
        run_id: The prediction run ID
        predictions: List of prediction dicts with keys:
            - work_year, experience_level, employment_type, job_title,
            - employee_residence, remote_ratio, company_location, company_size,
            - predicted_salary_usd
            - (optional) api_status
    
    Returns:
        Number of predictions inserted
    
    Raises:
        Exception: If insert fails
    """
    if not predictions:
        logger.warning("No predictions to insert")
        return 0
    
    client = get_client()
    
    # Add run_id to each prediction
    payloads = [
        {
            **pred,
            "run_id": run_id,
            "api_status": pred.get("api_status", "success"),
        }
        for pred in predictions
    ]
    
    try:
        response = client.table("predictions").insert(payloads).execute()
        count = len(response.data)
        logger.info(f"Inserted {count} predictions for run {run_id}")
        return count
    except Exception as e:
        logger.error(f"Failed to insert predictions for run {run_id}: {e}")
        raise


def get_predictions_for_run(
    run_id: int,
    limit: int = 1000
) -> List[Dict[str, Any]]:
    """
    Fetch all predictions for a run.
    
    Args:
        run_id: The prediction run ID
        limit: Maximum number of predictions to fetch
    
    Returns:
        List of prediction records
    """
    client = get_client()
    
    try:
        response = client.table("predictions").select("*").eq(
            "run_id", run_id
        ).limit(limit).execute()
        logger.info(f"Fetched {len(response.data)} predictions for run {run_id}")
        return response.data
    except Exception as e:
        logger.error(f"Failed to fetch predictions for run {run_id}: {e}")
        raise


# ============================================================================
# LLM Analyses (narratives and chart recommendations)
# ============================================================================

def insert_analysis(
    run_id: int,
    title: Optional[str],
    summary: Optional[str],
    key_insights: Optional[List[str]],
    chart_spec: Optional[Dict[str, Any]],
    raw_response: Optional[Dict[str, Any]] = None
) -> int:
    """
    Insert an LLM analysis for a run.
    
    Args:
        run_id: The prediction run ID
        title: Analysis title
        summary: Natural language summary
        key_insights: List of insight strings
        chart_spec: Chart specification dict
        raw_response: Full LLM response (optional, for debugging)
    
    Returns:
        analysis_id (integer)
    
    Raises:
        Exception: If insert fails
    """
    client = get_client()
    
    payload = {
        "run_id": run_id,
        "title": title,
        "summary": summary,
        "key_insights_json": key_insights,  # Supabase auto-converts list to JSONB
        "chart_spec_json": chart_spec,
        "raw_llm_response_json": raw_response,
    }
    
    try:
        response = client.table("llm_analyses").insert(payload).execute()
        analysis_id = response.data[0]["id"]
        logger.info(f"Created LLM analysis {analysis_id} for run {run_id}")
        return analysis_id
    except Exception as e:
        logger.error(f"Failed to insert analysis for run {run_id}: {e}")
        raise


def get_analysis_for_run(run_id: int) -> Optional[Dict[str, Any]]:
    """
    Fetch the LLM analysis for a run.
    
    For v1, assumes one analysis per run. Future versions may support multiple.
    
    Args:
        run_id: The prediction run ID
    
    Returns:
        Analysis record dict or None if not found
    """
    client = get_client()
    
    try:
        response = client.table("llm_analyses").select("*").eq(
            "run_id", run_id
        ).limit(1).execute()
        if response.data:
            return response.data[0]
        return None
    except Exception as e:
        logger.error(f"Failed to fetch analysis for run {run_id}: {e}")
        raise


# ============================================================================
# Utility functions
# ============================================================================

def create_full_run_with_predictions_and_analysis(
    model_name: str,
    model_version: str,
    predictions: List[Dict[str, Any]],
    analysis_title: Optional[str],
    analysis_summary: Optional[str],
    analysis_insights: Optional[List[str]],
    analysis_chart: Optional[Dict[str, Any]],
    run_name: Optional[str] = None,
    notes: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience function: create a complete run with predictions and analysis in one call.
    
    Returns a dict with:
        - run_id
        - prediction_count
        - analysis_id
    
    Raises:
        Exception: If any step fails
    """
    # Create run
    run_id = create_run(
        model_name=model_name,
        model_version=model_version,
        scenario_count=len(predictions),
        run_name=run_name,
        notes=notes,
        status="pending"
    )
    
    try:
        # Insert predictions
        prediction_count = insert_predictions(run_id, predictions)
        
        # Insert analysis
        analysis_id = insert_analysis(
            run_id=run_id,
            title=analysis_title,
            summary=analysis_summary,
            key_insights=analysis_insights,
            chart_spec=analysis_chart,
        )
        
        # Mark run as completed
        update_run_status(run_id, "completed")
        
        logger.info(f"Complete run {run_id}: {prediction_count} predictions, analysis {analysis_id}")
        
        return {
            "run_id": run_id,
            "prediction_count": prediction_count,
            "analysis_id": analysis_id,
        }
    except Exception as e:
        # Mark run as failed if any step fails
        try:
            update_run_status(run_id, "failed")
        except:
            pass
        raise
