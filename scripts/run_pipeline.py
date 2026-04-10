#!/usr/bin/env python3
"""
End-to-end pipeline orchestrator.

Orchestrates the complete workflow:
1. Generate prediction scenarios
2. Call FastAPI /predict endpoint for each scenario
3. Collect and flatten predictions
4. Analyze with local Ollama
5. Persist results to Supabase
6. Save local debug artifacts

Usage:
    python scripts/run_pipeline.py [--run-name "Optional Run Name"]

Environment variables required:
    API_BASE_URL: FastAPI server URL (default: http://localhost:8000)
    SUPABASE_URL, SUPABASE_KEY: Supabase credentials
    OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT: Ollama config
"""

import sys
import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import requests
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
load_dotenv()

from app.pipeline.scenarios import generate_scenarios
from app.llm.analyzer import analyze_with_ollama
from app.storage.repository import (
    create_run,
    insert_predictions,
    insert_analysis,
    update_run_status,
)

# ============================================================================
# Configuration
# ============================================================================

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
PREDICT_ENDPOINT = f"{API_BASE_URL}/predict"
PREDICT_BATCH_ENDPOINT = f"{API_BASE_URL}/predict-batch"
CHUNK_SIZE = 50  # Default chunk size for batch predictions
API_TIMEOUT = 10  # seconds
ARTIFACTS_DIR = project_root / "artifacts"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Helper Functions
# ============================================================================

def call_predict_api(scenario: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Call FastAPI /predict endpoint with a scenario.
    
    Args:
        scenario: Dict with keys: work_year, experience_level, employment_type, job_title,
                  employee_residence, remote_ratio, company_location, company_size
    
    Returns:
        Tuple of (api_response_dict or None, status_string)
        status_string is either 'success' or error message
    """
    try:
        response = requests.get(
            PREDICT_ENDPOINT,
            params=scenario,
            timeout=API_TIMEOUT
        )
        response.raise_for_status()
        return response.json(), "success"
    except requests.exceptions.ConnectionError as e:
        error_msg = f"Connection error: {str(e)}"
        logger.warning(error_msg)
        return None, error_msg
    except requests.exceptions.Timeout as e:
        error_msg = f"Timeout: {str(e)}"
        logger.warning(error_msg)
        return None, error_msg
    except requests.exceptions.HTTPError as e:
        error_msg = f"HTTP {response.status_code}: {response.text[:100]}"
        logger.warning(error_msg)
        return None, error_msg
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        logger.warning(error_msg)
        return None, error_msg


def call_predict_batch_api(scenarios: List[Dict[str, Any]]) -> Tuple[Optional[List[Dict[str, Any]]], str]:
    """
    Call FastAPI /predict-batch endpoint with a batch of scenarios.
    
    Args:
        scenarios: List of dicts with keys: work_year, experience_level, employment_type, job_title,
                  employee_residence, remote_ratio, company_location, company_size
    
    Returns:
        Tuple of (list_of_predictions or None, status_string)
        Each prediction has: all 8 input fields + predicted_salary_usd + api_status
        status_string is either 'success' or error message
    """
    if not scenarios:
        return [], "success"
    
    try:
        payload = {"scenarios": scenarios}
        response = requests.post(
            PREDICT_BATCH_ENDPOINT,
            json=payload,
            timeout=API_TIMEOUT
        )
        response.raise_for_status()
        batch_response = response.json()
        predictions = batch_response.get("predictions", [])
        return predictions, "success"
    except requests.exceptions.ConnectionError as e:
        error_msg = f"Connection error: {str(e)}"
        logger.warning(error_msg)
        return None, error_msg
    except requests.exceptions.Timeout as e:
        error_msg = f"Timeout: {str(e)}"
        logger.warning(error_msg)
        return None, error_msg
    except requests.exceptions.HTTPError as e:
        error_msg = f"HTTP {response.status_code}: {response.text[:100]}"
        logger.warning(error_msg)
        return None, error_msg
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        logger.warning(error_msg)
        return None, error_msg


def extract_value_from_response_field(field_value: Any) -> Any:
    """
    Extract raw value from API response field.
    
    Fields can be either:
    - Primitive values (int, str, etc.)
    - ValueLabel dicts with 'value' and 'label' keys
    
    Args:
        field_value: The field value from API response
    
    Returns:
        The raw value (unwrapped if it was a ValueLabel)
    """
    if isinstance(field_value, dict) and "value" in field_value:
        return field_value["value"]
    return field_value


def transform_api_response_to_prediction(
    scenario: Dict[str, Any],
    api_response: Dict[str, Any],
    api_status: str
) -> Dict[str, Any]:
    """
    Transform API response into flat prediction dict for storage and LLM.
    
    Args:
        scenario: Original input scenario
        api_response: Response from /predict endpoint
        api_status: Status string ('success' or error message)
    
    Returns:
        Dict with keys: all 8 input features + predicted_salary_usd + api_status
    """
    # Extract inputs and flatten ValueLabel objects
    inputs_with_labels = api_response.get("inputs", {})
    
    prediction = {
        "work_year": extract_value_from_response_field(inputs_with_labels.get("work_year", scenario["work_year"])),
        "experience_level": extract_value_from_response_field(inputs_with_labels.get("experience_level", scenario["experience_level"])),
        "employment_type": extract_value_from_response_field(inputs_with_labels.get("employment_type", scenario["employment_type"])),
        "job_title": extract_value_from_response_field(inputs_with_labels.get("job_title", scenario["job_title"])),
        "employee_residence": extract_value_from_response_field(inputs_with_labels.get("employee_residence", scenario["employee_residence"])),
        "remote_ratio": extract_value_from_response_field(inputs_with_labels.get("remote_ratio", scenario["remote_ratio"])),
        "company_location": extract_value_from_response_field(inputs_with_labels.get("company_location", scenario["company_location"])),
        "company_size": extract_value_from_response_field(inputs_with_labels.get("company_size", scenario["company_size"])),
        "predicted_salary_usd": api_response.get("predicted_salary_usd", 0),
        "api_status": api_status,
    }
    return prediction


def run_llm_analysis(predictions: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Pass collected predictions to LLM analyzer.
    
    Args:
        predictions: List of prediction dicts (raw values, no labels)
    
    Returns:
        Dict with keys: title, summary, key_insights_json, chart_spec_json, raw_llm_response_json
        or None if analysis completely fails
    """
    logger.info(f"Running LLM analysis on {len(predictions)} predictions")
    
    try:
        analysis_result = analyze_with_ollama(predictions)
        logger.info("LLM analysis completed successfully")
        return analysis_result
    except Exception as e:
        logger.error(f"LLM analysis failed: {str(e)}")
        return None


def persist_run(
    run_id: int,
    scenario_count: int,
    predictions: List[Dict[str, Any]],
    analysis: Optional[Dict[str, Any]],
    run_name: Optional[str] = None,
) -> bool:
    """
    Persist predictions and analysis to Supabase for an existing run.
    
    The run should already be created with status='pending' before calling this.
    On success, updates run status to 'completed'. On failure, updates to 'failed'.
    
    Args:
        run_id: The existing run_id (must already be created)
        scenario_count: Number of scenarios for the run
        predictions: List of prediction dicts
        analysis: Analysis dict or None
        run_name: Optional run_name (for logging only)
    
    Returns:
        True if successful, False if failed
    """
    try:
        # Insert predictions
        logger.info(f"Inserting {len(predictions)} predictions")
        count = insert_predictions(run_id, predictions)
        logger.info(f"Inserted {count} predictions")
        
        # Insert analysis if available
        if analysis:
            logger.info("Inserting LLM analysis")
            analysis_id = insert_analysis(
                run_id=run_id,
                title=analysis.get("title", "Salary Analysis"),
                summary=analysis.get("summary", ""),
                key_insights=analysis.get("key_insights", []),
                chart_spec=analysis.get("chart", {}),  # Fixed: use "chart" key, not "chart_spec"
                raw_response=analysis,  # Store full analysis dict as raw LLM response
            )
            logger.info(f"Inserted analysis_id: {analysis_id}")
        else:
            logger.warning("No analysis to insert (analysis was None)")
        
        # Update run status to completed
        logger.info("Marking run as completed")
        update_run_status(run_id, "completed")
        
        logger.info(f"Successfully persisted run {run_id}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to persist to run {run_id}: {str(e)}")
        # Try to mark run as failed
        try:
            update_run_status(run_id, "failed")
        except Exception as e2:
            logger.error(f"Failed to mark run as failed: {str(e2)}")
        return False


def save_debug_artifacts(
    predictions: List[Dict[str, Any]],
    analysis: Optional[Dict[str, Any]],
) -> None:
    """
    Save local debug artifacts for inspection.
    
    Saves:
    - artifacts/pipeline_predictions.json
    - artifacts/pipeline_analysis.json
    
    Args:
        predictions: List of prediction dicts
        analysis: Analysis dict or None
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save predictions
    predictions_file = ARTIFACTS_DIR / "pipeline_predictions.json"
    try:
        with open(predictions_file, "w") as f:
            json.dump(predictions, f, indent=2)
        logger.info(f"Saved predictions to {predictions_file}")
    except Exception as e:
        logger.error(f"Failed to save predictions: {str(e)}")
    
    # Save analysis
    if analysis:
        analysis_file = ARTIFACTS_DIR / "pipeline_analysis.json"
        try:
            with open(analysis_file, "w") as f:
                json.dump(analysis, f, indent=2)
            logger.info(f"Saved analysis to {analysis_file}")
        except Exception as e:
            logger.error(f"Failed to save analysis: {str(e)}")


def print_summary(
    scenario_count: int,
    predictions: List[Dict[str, Any]],
    analysis: Optional[Dict[str, Any]],
    run_id: Optional[int],
    success: bool,
) -> None:
    """
    Print a concise execution summary.
    
    Args:
        scenario_count: Number of scenarios generated
        predictions: List of predictions collected
        analysis: Analysis dict or None
        run_id: Created run_id or None
        success: Whether pipeline completed successfully
    """
    successful_predictions = sum(1 for p in predictions if p.get("api_status") == "success")
    failed_predictions = len(predictions) - successful_predictions
    ollama_used = analysis is not None
    
    print("\n" + "="*70)
    print("PIPELINE EXECUTION SUMMARY")
    print("="*70)
    print(f"Status: {'✓ SUCCESS' if success else '✗ FAILED'}")
    print(f"Run ID: {run_id if run_id else 'N/A'}")
    print(f"Scenarios Generated: {scenario_count}")
    print(f"Successful Predictions: {successful_predictions}")
    print(f"Failed Predictions: {failed_predictions}")
    print(f"LLM Analysis: {'✓ Completed' if ollama_used else '✗ Not completed'}")
    
    if analysis:
        print(f"Analysis Title: {analysis.get('title', 'N/A')}")
    
    print("="*70 + "\n")


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    """Execute the full pipeline orchestration."""
    parser = argparse.ArgumentParser(description="Run the complete salary prediction pipeline")
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional human-readable run name (auto-generated if not provided)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help=f"Batch size for API predictions (default: {CHUNK_SIZE})"
    )
    args = parser.parse_args()
    
    logger.info("Starting pipeline orchestration")
    
    run_id = None  # Initialize for exception handling
    
    try:
        # Step 0: Generate run_name and create run record early
        if args.run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"pipeline_{timestamp}"
        else:
            run_name = args.run_name
        
        # Step 1: Generate scenarios
        logger.info("Step 1: Generating scenarios")
        scenarios = generate_scenarios()
        scenario_count = len(scenarios)
        logger.info(f"Generated {scenario_count} scenarios (10 anchor triples × 432 combinations = ~4,320 total)")
        
        if scenario_count == 0:
            logger.error("No scenarios generated!")
            return 1
        
        # Create prediction run early with pending status
        logger.info(f"Creating prediction run: {run_name}")
        try:
            run_id = create_run(
                model_name="DecisionTreeRegressor",
                model_version="v1",
                scenario_count=scenario_count,
                run_name=run_name,
                status="pending",
            )
            logger.info(f"Created run_id: {run_id} with status=pending")
        except Exception as e:
            logger.error(f"Failed to create run: {str(e)}")
            print_summary(
                scenario_count=scenario_count,
                predictions=[],
                analysis=None,
                run_id=None,
                success=False,
            )
            return 1
        
        # Step 2: Call API for predictions using batch chunks
        logger.info(f"Step 2: Calling API for predictions (chunk size: {args.chunk_size})")
        predictions = []
        total_chunks = (scenario_count + args.chunk_size - 1) // args.chunk_size
        
        for chunk_idx in range(total_chunks):
            start_idx = chunk_idx * args.chunk_size
            end_idx = min(start_idx + args.chunk_size, scenario_count)
            chunk = scenarios[start_idx:end_idx]
            
            logger.info(f"Processing chunk {chunk_idx + 1}/{total_chunks} (scenarios {start_idx + 1}-{end_idx})")
            
            # Call batch API
            batch_predictions, batch_status = call_predict_batch_api(chunk)
            
            if batch_predictions and batch_status == "success":
                # Successfully retrieved batch predictions
                for pred_item in batch_predictions:
                    # Convert batch response item to prediction dict
                    prediction = {
                        "work_year": pred_item["work_year"],
                        "experience_level": pred_item["experience_level"],
                        "employment_type": pred_item["employment_type"],
                        "job_title": pred_item["job_title"],
                        "employee_residence": pred_item["employee_residence"],
                        "remote_ratio": pred_item["remote_ratio"],
                        "company_location": pred_item["company_location"],
                        "company_size": pred_item["company_size"],
                        "predicted_salary_usd": pred_item["predicted_salary_usd"],
                        "api_status": pred_item.get("api_status", "success"),
                    }
                    predictions.append(prediction)
            else:
                # Batch call failed, mark all scenarios in chunk with error
                logger.warning(f"Batch {chunk_idx + 1} failed: {batch_status}")
                for scenario in chunk:
                    prediction = {
                        "work_year": scenario["work_year"],
                        "experience_level": scenario["experience_level"],
                        "employment_type": scenario["employment_type"],
                        "job_title": scenario["job_title"],
                        "employee_residence": scenario["employee_residence"],
                        "remote_ratio": scenario["remote_ratio"],
                        "company_location": scenario["company_location"],
                        "company_size": scenario["company_size"],
                        "predicted_salary_usd": 0,
                        "api_status": batch_status,
                    }
                    predictions.append(prediction)
        
        logger.info(f"Collected {len(predictions)} predictions")
        
        # Step 3: Run LLM analysis
        logger.info("Step 3: Running LLM analysis")
        analysis = run_llm_analysis(predictions)
        
        # Step 4: Save debug artifacts
        logger.info("Step 4: Saving debug artifacts")
        save_debug_artifacts(predictions, analysis)
        
        # Step 5: Persist to Supabase
        logger.info("Step 5: Persisting to Supabase")
        success = persist_run(
            run_id=run_id,
            scenario_count=scenario_count,
            predictions=predictions,
            analysis=analysis,
            run_name=run_name,
        )
        
        # Step 6: Print summary
        print_summary(
            scenario_count=scenario_count,
            predictions=predictions,
            analysis=analysis,
            run_id=run_id,
            success=success,
        )
        
        logger.info("Pipeline orchestration completed" if success else "Pipeline failed")
        return 0 if success else 1
    
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
        # Mark run as failed if it was created
        if run_id:
            try:
                update_run_status(run_id, "failed")
                logger.info(f"Marked run {run_id} as failed")
            except Exception as e2:
                logger.error(f"Failed to mark run as failed: {str(e2)}")
        print_summary(
            scenario_count=0,
            predictions=[],
            analysis=None,
            run_id=None,
            success=False,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
