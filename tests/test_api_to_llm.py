#!/usr/bin/env python3
"""
Smoke test: API prediction -> LLM analysis integration.

Verifies that FastAPI /predict outputs can be successfully transformed and passed
to the LLM analysis layer (app/llm/analyzer.py).

Run with: python scripts/test_api_to_llm.py

Requirements:
- FastAPI server must be running on http://localhost:8000
- Ollama (local LLM) is optional - test shows fallback if unavailable
"""

import sys
import json
from pathlib import Path
import requests
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.llm.analyzer import analyze_with_ollama

# Configuration
API_BASE_URL = "http://localhost:8000"
PREDICT_ENDPOINT = f"{API_BASE_URL}/predict"
API_TIMEOUT = 10  # seconds

# Sample scenarios for testing (3-5 representative cases)
SAMPLE_SCENARIOS = [
    {
        "work_year": 2022,
        "experience_level": "EN",
        "employment_type": "FT",
        "job_title": "Data Scientist",
        "employee_residence": "US",
        "remote_ratio": 100,
        "company_location": "US",
        "company_size": "L",
    },
    {
        "work_year": 2022,
        "experience_level": "MI",
        "employment_type": "FT",
        "job_title": "Data Engineer",
        "employee_residence": "GB",
        "remote_ratio": 50,
        "company_location": "GB",
        "company_size": "M",
    },
    {
        "work_year": 2022,
        "experience_level": "SE",
        "employment_type": "FT",
        "job_title": "ML Engineer",
        "employee_residence": "CA",
        "remote_ratio": 0,
        "company_location": "US",
        "company_size": "L",
    },
]


def call_api_predict(scenario: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call the FastAPI /predict endpoint with a scenario.
    
    Returns the full API response dict.
    Raises requests.RequestException if call fails.
    """
    try:
        response = requests.get(
            PREDICT_ENDPOINT,
            params=scenario,
            timeout=API_TIMEOUT
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            f"Could not connect to API at {API_BASE_URL}. "
            "Is FastAPI running? Start with: uvicorn app.api.main:app --reload"
        )
    except requests.exceptions.HTTPError as e:
        raise ValueError(f"API error: {response.status_code} - {response.text}")
    except Exception as e:
        raise


def transform_api_response_for_llm(
    api_response: Dict[str, Any],
    original_scenario: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Transform API response into the structure expected by analyzer.py.
    
    The API returns a nested structure with ValueLabel objects for most fields,
    but some fields (like work_year) are raw values. This function flattens them.
    
    Args:
        api_response: Full response from API /predict endpoint
        original_scenario: Original request scenario (for reference)
    
    Returns:
        Flat dict suitable for app/llm/analyzer.py
        
    Structure:
        {
            "work_year": int,
            "experience_level": str,
            "employment_type": str,
            "job_title": str,
            "employee_residence": str,
            "remote_ratio": int,
            "company_location": str,
            "company_size": str,
            "predicted_salary_usd": float
        }
    """
    # Extract inputs with labels from API response
    inputs = api_response.get("inputs", {})
    
    # Helper function to extract value from ValueLabel object or raw value
    def get_value(field):
        val = inputs.get(field)
        # If it's a dict with 'value' key, extract the value; otherwise use as-is
        if isinstance(val, dict) and "value" in val:
            return val["value"]
        return val
    
    # Flatten the nested ValueLabel objects to raw values
    transformed = {
        "work_year": get_value("work_year"),
        "experience_level": get_value("experience_level"),
        "employment_type": get_value("employment_type"),
        "job_title": get_value("job_title"),
        "employee_residence": get_value("employee_residence"),
        "remote_ratio": get_value("remote_ratio"),
        "company_location": get_value("company_location"),
        "company_size": get_value("company_size"),
        "predicted_salary_usd": api_response.get("predicted_salary_usd"),
    }
    
    return transformed


def main():
    """Run the smoke test."""
    print("\n" + "=" * 80)
    print("SMOKE TEST: FastAPI -> LLM Analysis Integration")
    print("=" * 80)
    
    # Step 1: Call API for sample scenarios
    print("\n" + "-" * 80)
    print("STEP 1: Calling FastAPI /predict endpoint")
    print("-" * 80)
    
    api_responses = []
    transformed_predictions = []
    
    for i, scenario in enumerate(SAMPLE_SCENARIOS, 1):
        print(f"\nScenario {i}: {scenario['job_title']} (Level: {scenario['experience_level']})")
        
        try:
            response = call_api_predict(scenario)
            api_responses.append(response)
            
            # Transform for LLM Input
            transformed = transform_api_response_for_llm(response, scenario)
            transformed_predictions.append(transformed)
            
            print(f"  ✓ API call succeeded")
            print(f"    Predicted salary: ${response['predicted_salary_usd']:,.2f}")
            
        except ConnectionError as e:
            print(f"\n❌ API Connection Error:")
            print(f"   {e}")
            return False
        except Exception as e:
            print(f"\n❌ API Error: {e}")
            return False
    
    # Step 2: Display raw API responses
    print("\n" + "-" * 80)
    print("STEP 2: Raw API Prediction Outputs")
    print("-" * 80)
    
    for i, response in enumerate(api_responses, 1):
        print(f"\nResponse {i}:")
        print(json.dumps(response, indent=2, default=str))
    
    # Step 3: Display transformed structure for LLM
    print("\n" + "-" * 80)
    print("STEP 3: Transformed Predictions for LLM Layer")
    print("-" * 80)
    
    print(f"\n{len(transformed_predictions)} predictions transformed for LLM input:")
    for i, pred in enumerate(transformed_predictions, 1):
        print(f"\nPrediction {i}:")
        print(json.dumps(pred, indent=2))
    
    # Step 4: Run LLM analysis
    print("\n" + "-" * 80)
    print("STEP 4: Running LLM Analysis")
    print("-" * 80)
    
    print(f"\nCalling analyze_with_ollama() with {len(transformed_predictions)} predictions...")
    
    analysis = analyze_with_ollama(transformed_predictions, fallback=True)
    
    # Step 5: Display final LLM analysis result
    print("\n" + "-" * 80)
    print("STEP 5: Final LLM Analysis Result")
    print("-" * 80)
    
    print(json.dumps(analysis, indent=2))
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ SMOKE TEST COMPLETE")
    print("=" * 80)
    print(f"\nSummary:")
    print(f"  • API scenarios tested: {len(SAMPLE_SCENARIOS)}")
    print(f"  • API responses collected: {len(api_responses)}")
    print(f"  • Predictions transformed: {len(transformed_predictions)}")
    print(f"  • LLM analysis generated: {'Yes' if analysis else 'No'}")
    print(f"  • Ollama available: {'Yes' if 'unavailable' not in analysis.get('summary', '').lower() else 'No'}")
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
