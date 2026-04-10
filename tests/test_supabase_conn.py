"""
Integration test for Supabase persistence layer.

Tests complete workflow:
- Create prediction run
- Insert predictions
- Insert LLM analysis
- Retrieve and verify data
"""

import pytest
import sys
from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
load_dotenv(project_root / ".env")

from app.storage.repository import (
    create_run, 
    insert_predictions, 
    insert_analysis,
    get_run,
    get_predictions_for_run,
    get_analysis_for_run
)


@pytest.fixture
def test_run_id():
    """Create a test run that will be cleaned up after the test."""
    run_id = create_run(
        model_name="DecisionTreeRegressor",
        model_version="v1_test",
        scenario_count=1,
        run_name="pytest_integration_test"
    )
    yield run_id
    # Note: Manual cleanup would require delete operations; 
    # for now, relying on test database or manual cleanup


def test_create_run():
    """Test creating a prediction run."""
    run_id = create_run(
        model_name="DecisionTreeRegressor",
        model_version="v1",
        scenario_count=3
    )
    assert isinstance(run_id, int), "run_id should be an integer"
    assert run_id > 0, "run_id should be positive"
    print(f"✓ Created run: {run_id}")


def test_insert_and_retrieve_predictions(test_run_id):
    """Test inserting predictions and retrieving them."""
    predictions = [
        {
            "work_year": 2022,
            "experience_level": "EN",
            "employment_type": "FT",
            "job_title": "Data Scientist",
            "employee_residence": "US",
            "remote_ratio": 100,
            "company_location": "US",
            "company_size": "L",
            "predicted_salary_usd": 79333.33,
        },
        {
            "work_year": 2023,
            "experience_level": "MI",
            "employment_type": "FT",
            "job_title": "ML Engineer",
            "employee_residence": "CA",
            "remote_ratio": 50,
            "company_location": "CA",
            "company_size": "M",
            "predicted_salary_usd": 125000.00,
        }
    ]
    
    count = insert_predictions(test_run_id, predictions)
    assert count == 2, f"Expected 2 predictions inserted, got {count}"
    print(f"✓ Inserted {count} predictions")
    
    # Verify retrieval
    retrieved = get_predictions_for_run(test_run_id)
    assert len(retrieved) == 2, f"Expected 2 predictions retrieved, got {len(retrieved)}"
    assert retrieved[0]["predicted_salary_usd"] == 79333.33
    print(f"✓ Retrieved {len(retrieved)} predictions")


def test_insert_and_retrieve_analysis(test_run_id):
    """Test inserting LLM analysis and retrieving it."""
    analysis_id = insert_analysis(
        run_id=test_run_id,
        title="Strong market for senior roles",
        summary="Senior positions command 35% premium over junior roles.",
        key_insights=["Remote work stable", "US locations pay 15% more", "Experience matters most"],
        chart_spec={"chart_type": "bar", "x": "experience_level", "y": "predicted_salary_usd"}
    )
    
    assert isinstance(analysis_id, int), "analysis_id should be an integer"
    assert analysis_id > 0, "analysis_id should be positive"
    print(f"✓ Created analysis: {analysis_id}")
    
    # Verify retrieval
    retrieved = get_analysis_for_run(test_run_id)
    assert retrieved is not None, "Analysis should be retrievable"
    assert retrieved["title"] == "Strong market for senior roles"
    assert len(retrieved["key_insights_json"]) == 3
    print(f"✓ Retrieved analysis with title: {retrieved['title']}")


def test_get_run_status(test_run_id):
    """Test retrieving run and checking status."""
    run = get_run(test_run_id)
    
    assert run is not None, "Run should exist"
    assert "status" in run, "Run should have status field"
    assert run["status"] in ["pending", "completed", "failed"], "Status should be valid"
    print(f"✓ Run status: {run['status']}")


def test_complete_workflow():
    """Test the complete workflow: create run → insert predictions → insert analysis → verify."""
    # Create run
    run_id = create_run(
        model_name="DecisionTreeRegressor",
        model_version="v1_workflow",
        scenario_count=1,
        run_name="complete_workflow_test"
    )
    assert run_id > 0
    
    # Insert predictions
    predictions = [
        {
            "work_year": 2022,
            "experience_level": "EN",
            "employment_type": "FT",
            "job_title": "Data Scientist",
            "employee_residence": "US",
            "remote_ratio": 100,
            "company_location": "US",
            "company_size": "L",
            "predicted_salary_usd": 79333.33,
        }
    ]
    count = insert_predictions(run_id, predictions)
    assert count == 1
    
    # Insert analysis
    analysis_id = insert_analysis(
        run_id=run_id,
        title="Workflow test analysis",
        summary="This is a test.",
        key_insights=["Insight 1", "Insight 2"],
        chart_spec={"chart_type": "bar", "x": "experience_level", "y": "predicted_salary_usd"}
    )
    assert analysis_id > 0
    
    # Verify all data is retrievable
    run = get_run(run_id)
    preds = get_predictions_for_run(run_id)
    analysis = get_analysis_for_run(run_id)
    
    assert run is not None
    assert len(preds) == 1
    assert analysis is not None
    
    print(f"✓ Complete workflow test passed (run_id: {run_id})")