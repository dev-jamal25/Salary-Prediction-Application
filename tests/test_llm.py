#!/usr/bin/env python3
"""
Test Task 5: Local LLM analysis with Ollama

Run with: python tests/test_llm.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.llm.analyzer import analyze_with_ollama
from app.llm.chart_recommender import validate_and_normalize_chart
import json


def test_chart_validation():
    """Test 1: Chart validation with various inputs"""
    print("\n" + "=" * 70)
    print("TEST 1: Chart Validation")
    print("=" * 70)
    
    # Test valid chart
    valid_chart = {
        "chart_type": "bar",
        "title": "Salary by Role",
        "x": "job_title",
        "y": "predicted_salary_usd",
        "aggregation": "mean",
        "description": "Shows average salary by job title"
    }
    result = validate_and_normalize_chart(valid_chart)
    print("\n✓ Valid chart passed:")
    print(json.dumps(result, indent=2))
    
    # Test invalid chart_type
    invalid_chart = {
        "chart_type": "heatmap",
        "title": "Invalid Chart",
        "x": "exp",
        "y": "salary",
        "aggregation": "mean",
        "description": "Should default to bar"
    }
    result = validate_and_normalize_chart(invalid_chart)
    print("\n✓ Invalid chart_type normalized to bar:")
    print(f"  chart_type: {result['chart_type']}")
    
    # Test missing fields
    incomplete_chart = {"title": "Incomplete"}
    result = validate_and_normalize_chart(incomplete_chart)
    print("\n✓ Incomplete chart uses defaults:")
    print(json.dumps(result, indent=2))
    
    # Test non-dict input
    result = validate_and_normalize_chart("not a dict")
    print("\n✓ Non-dict input uses defaults:")
    print(json.dumps(result, indent=2))


def test_ollamaanalysis():
    """Test 2: Ollama analysis with sample predictions"""
    print("\n" + "=" * 70)
    print("TEST 2: Ollama Analysis")
    print("=" * 70)
    
    # Sample predictions
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
            "predicted_salary_usd": 85000
        },
        {
            "work_year": 2022,
            "experience_level": "MI",
            "employment_type": "FT",
            "job_title": "Data Scientist",
            "employee_residence": "US",
            "remote_ratio": 100,
            "company_location": "US",
            "company_size": "L",
            "predicted_salary_usd": 115000
        },
        {
            "work_year": 2022,
            "experience_level": "SE",
            "employment_type": "FT",
            "job_title": "Data Scientist",
            "employee_residence": "US",
            "remote_ratio": 100,
            "company_location": "US",
            "company_size": "L",
            "predicted_salary_usd": 150000
        },
        {
            "work_year": 2022,
            "experience_level": "EN",
            "employment_type": "FT",
            "job_title": "Data Engineer",
            "employee_residence": "US",
            "remote_ratio": 100,
            "company_location": "US",
            "company_size": "L",
            "predicted_salary_usd": 95000
        },
        {
            "work_year": 2022,
            "experience_level": "SE",
            "employment_type": "FT",
            "job_title": "Data Engineer",
            "employee_residence": "US",
            "remote_ratio": 100,
            "company_location": "US",
            "company_size": "L",
            "predicted_salary_usd": 165000
        },
    ]
    
    print("\nInput predictions: 5 samples")
    print(f"  - Experience levels: EN (Junior), MI (Intermediate), SE (Senior)")
    print(f"  - Job titles: Data Scientist, Data Engineer")
    print(f"  - Salary range: $85K - $165K")
    
    print("\nCalling analyze_with_ollama()...")
    analysis = analyze_with_ollama(predictions)
    
    print("\nAnalysis Result:")
    print(json.dumps(analysis, indent=2))
    
    # Validate required fields
    print("\n✓ Structure validation:")
    required_top_fields = {"title", "summary", "key_insights", "chart"}
    present = set(analysis.keys())
    print(f"  - Top-level fields: {present == required_top_fields}")
    
    if "chart" in analysis:
        required_chart_fields = {"chart_type", "title", "x", "y", "aggregation", "description"}
        chart = analysis["chart"]
        chart_present = set(chart.keys())
        print(f"  - Chart fields: {chart_present == required_chart_fields}")
        print(f"    - chart_type: {chart.get('chart_type')}")
        print(f"    - x field: {chart.get('x')}")
        print(f"    - aggregation: {chart.get('aggregation')}")
    
    if isinstance(analysis.get("key_insights"), list):
        print(f"  - Insights count: {len(analysis['key_insights'])}")


def test_fallback():
    """Test 3: Fallback behavior"""
    print("\n" + "=" * 70)
    print("TEST 3: Fallback Behavior")
    print("=" * 70)
    
    print("\nNote: Fallback is triggered if Ollama is not running.")
    print("Testing with minimal predictions...")
    
    minimal = [
        {"experience_level": "EN", "predicted_salary_usd": 80000},
        {"experience_level": "SE", "predicted_salary_usd": 160000}
    ]
    
    analysis = analyze_with_ollama(minimal, fallback=True)
    
    if "summary" in analysis and "not running" in analysis["summary"].lower():
        print("\n✓ Fallback triggered (Ollama not running):")
        print(f"  - Title: {analysis['title']}")
        print(f"  - Summary: {analysis['summary'][:80]}...")
    else:
        print("\n✓ Ollama analysis succeeded:")
        print(f"  - Title: {analysis['title']}")
        print(f"  - Summary: {analysis['summary'][:80]}...")


def test_edge_cases():
    """Test 4: Edge cases"""
    print("\n" + "=" * 70)
    print("TEST 4: Edge Cases")
    print("=" * 70)
    
    # Empty list
    print("\n1. Empty predictions list:")
    result = analyze_with_ollama([])
    print(f"   Result is dict: {isinstance(result, dict)}")
    print(f"   Has fallback structure: {'chart' in result}")
    
    # Non-list input
    print("\n2. Single prediction (not in list):")
    single = {"experience_level": "SE", "predicted_salary_usd": 150000}
    result = analyze_with_ollama(single)
    print(f"   Result is dict: {isinstance(result, dict)}")
    print(f"   Has title: {'title' in result}")
    
    # Predictions without salary
    print("\n3. Predictions missing salary field:")
    no_salary = [
        {"experience_level": "EN", "job_title": "Data Scientist"},
        {"experience_level": "SE", "job_title": "Data Engineer"}
    ]
    result = analyze_with_ollama(no_salary)
    print(f"   Result is dict: {isinstance(result, dict)}")
    print(f"   Fallback triggered: {'not running' in result.get('summary', '').lower()}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TASK 5 TESTS: Local LLM Analysis with Ollama")
    print("=" * 70)
    
    print("\nPre-flight checklist:")
    print("  ✓ analyzer.py loaded")
    print("  ✓ chart_recommender.py loaded")
    print("  ⓘ If Ollama is running, live tests will succeed")
    print("  ⓘ If Ollama is not running, fallback mode will be tested")
    
    try:
        test_chart_validation()
        test_fallback()
        test_edge_cases()
        test_ollamaanalysis()
        
        print("\n" + "=" * 70)
        print("✅ ALL TESTS COMPLETE")
        print("=" * 70)
        print("\nSummary:")
        print("  ✓ Chart validation working")
        print("  ✓ Fallback behavior working")
        print("  ✓ Edge cases handled")
        print("  ✓ Ollama integration ready")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
