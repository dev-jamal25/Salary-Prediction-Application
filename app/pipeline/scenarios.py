"""
Scenario Generation for Salary Prediction.

Generates representative prediction scenarios by combining:
1. Real observed anchor triples (job_title, employee_residence, company_location)
2. All low-cardinality feature values (work_year, experience_level, employment_type, company_size, remote_ratio)

Produces ~4,320 medium-batch scenarios suitable for batch prediction.

Usage:
    from app.pipeline.scenarios import generate_scenarios
    scenarios = generate_scenarios()
    
    # Or run directly:
    python -m app.pipeline.scenarios
"""

import pandas as pd
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from itertools import product

from app.utils.data_contract import (
    EXPERIENCE_LEVEL_VALUES,
    EMPLOYMENT_TYPE_VALUES,
    COMPANY_SIZE_VALUES,
    WORK_YEAR_VALUES,
    REMOTE_RATIO_VALUES,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
CLEANED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "ds_salaries_clean.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


# ============================================================================
# Core Functions
# ============================================================================

def load_cleaned_dataset() -> pd.DataFrame:
    """
    Load the cleaned dataset.
    
    Returns:
        DataFrame with columns: work_year, experience_level, employment_type, job_title,
                               salary, salary_currency, salary_in_usd, employee_residence,
                               remote_ratio, company_location, company_size
    
    Raises:
        FileNotFoundError: If cleaned data file not found
    """
    if not CLEANED_DATA_PATH.exists():
        raise FileNotFoundError(f"Cleaned data not found at {CLEANED_DATA_PATH}")
    
    df = pd.read_csv(CLEANED_DATA_PATH)
    logger.info(f"Loaded {len(df)} records from {CLEANED_DATA_PATH}")
    return df


def extract_anchor_triples(
    df: pd.DataFrame,
    max_triples: int = 30
) -> List[Tuple[str, str, str]]:
    """
    Extract representative anchor triples from the dataset.
    
    An anchor triple is a combination of (job_title, employee_residence, company_location)
    that appears in the real data. We select the most frequent triples to ensure
    representativeness.
    
    Strategy:
    - Find all observed combinations of (job_title, employee_residence, company_location)
    - Rank by frequency
    - Select top N to create a controlled, representative subset
    
    Args:
        df: Cleaned dataset
        max_triples: Maximum number of anchor triples to select (default: 30)
    
    Returns:
        List of (job_title, employee_residence, company_location) tuples
    """
    # Group by the three anchor columns and count occurrences
    anchor_counts = df.groupby(
        ["job_title", "employee_residence", "company_location"]
    ).size().reset_index(name="count")
    
    # Sort by frequency descending
    anchor_counts = anchor_counts.sort_values("count", ascending=False)
    
    # Select top N triples
    selected_triples = anchor_counts.head(max_triples)
    
    # Convert to list of tuples
    triples = [
        (row["job_title"], row["employee_residence"], row["company_location"])
        for _, row in selected_triples.iterrows()
    ]
    
    logger.info(f"Extracted {len(triples)} anchor triples from dataset")
    logger.info(f"Total unique triples available: {len(anchor_counts)}")
    
    return triples


def generate_scenarios(max_anchor_triples: int = 30) -> List[Dict[str, Any]]:
    """
    Generate prediction scenarios.
    
    Creates a combination of:
    - Representative anchor triples (job_title, employee_residence, company_location)
    - All values of low-cardinality features (work_year, experience_level, employment_type, company_size, remote_ratio)
    
    Default batch size: 10 anchor triples × 432 low-cardinality combinations = ~4,320 scenarios
    (Practical medium-sized batch for testing and batch prediction)
    
    Args:
        max_anchor_triples: Maximum number of anchor triples to select (default: 10 for ~236 total scenarios)
    
    Returns:
        List of scenario dicts, each with keys:
        - work_year
        - experience_level
        - employment_type
        - job_title
        - employee_residence
        - remote_ratio
        - company_location
        - company_size
    """
    # Load data
    df = load_cleaned_dataset()
    
    # Extract anchor triples
    anchor_triples = extract_anchor_triples(df, max_triples=max_anchor_triples)
    
    # Low-cardinality feature values
    low_card_features = {
        "work_year": WORK_YEAR_VALUES,
        "experience_level": EXPERIENCE_LEVEL_VALUES,
        "employment_type": EMPLOYMENT_TYPE_VALUES,
        "company_size": COMPANY_SIZE_VALUES,
        "remote_ratio": REMOTE_RATIO_VALUES,
    }
    
    # Generate all combinations
    scenarios = []
    
    for (job_title, employee_residence, company_location) in anchor_triples:
        # For each anchor triple, generate all combinations of low-cardinality features
        for work_year, exp_level, emp_type, comp_size, remote_ratio in product(
            low_card_features["work_year"],
            low_card_features["experience_level"],
            low_card_features["employment_type"],
            low_card_features["company_size"],
            low_card_features["remote_ratio"],
        ):
            scenario = {
                "work_year": work_year,
                "experience_level": exp_level,
                "employment_type": emp_type,
                "job_title": job_title,
                "employee_residence": employee_residence,
                "remote_ratio": remote_ratio,
                "company_location": company_location,
                "company_size": comp_size,
            }
            scenarios.append(scenario)
    
    logger.info(f"Generated {len(scenarios)} scenarios")
    logger.info(f"Breakdown: {len(anchor_triples)} anchor triples × {len(list(product(*low_card_features.values())))} low-cardinality combinations")
    
    return scenarios


def save_scenarios_to_json(scenarios: List[Dict[str, Any]]) -> Path:
    """
    Save scenarios to JSON artifact.
    
    Args:
        scenarios: List of scenario dicts
    
    Returns:
        Path to saved JSON file
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = ARTIFACTS_DIR / "generated_scenarios.json"
    
    try:
        with open(output_file, "w") as f:
            json.dump(scenarios, f, indent=2)
        logger.info(f"Saved {len(scenarios)} scenarios to {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"Failed to save scenarios: {str(e)}")
        raise


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    """
    Runnable script for testing and inspection.
    
    Usage:
        python -m app.pipeline.scenarios
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Generate scenarios (uses new practical default of 10 anchor triples)
    scenarios = generate_scenarios()
    
    # Save to JSON
    output_file = save_scenarios_to_json(scenarios)
    
    # Print summary
    print("\n" + "="*70)
    print("SCENARIO GENERATION SUMMARY")
    print("="*70)
    print(f"Total scenarios generated: {len(scenarios)}")
    print(f"Saved to: {output_file}")
    print("="*70)
    
    # Show example scenarios
    print("\nExample scenarios (first 5):")
    for i, scenario in enumerate(scenarios[:5], 1):
        print(f"\nScenario {i}:")
        for key, value in scenario.items():
            print(f"  {key}: {value}")
    
    print("\n" + "="*70 + "\n")
