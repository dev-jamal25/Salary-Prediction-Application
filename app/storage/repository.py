"""
Repository pattern for Supabase CRUD operations.

Three tables:
1. prediction_runs - pipeline execution metadata
2. predictions - individual prediction records
3. llm_analyses - narrative + chart output
"""

from typing import List, Optional, Dict, Any
import pandas as pd


# TODO: Implement CRUD methods
# - create_run(run_name, model_version, dataset_version) -> run_id
# - insert_predictions(run_id, predictions_df)
# - insert_analysis(run_id, llm_analysis_json)
# - get_latest_run() -> run metadata
# - get_predictions_for_run(run_id) -> DataFrame
# - get_analysis_for_run(run_id) -> analysis dict
