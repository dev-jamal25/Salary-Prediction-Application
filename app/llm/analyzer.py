"""
Ollama integration for narrative analysis.

Calls local LLM (llama3.2:1b) to generate data insights in structured JSON format.
Handles predictions summary, generates insights, and recommends visualization.
Validates and parses response with graceful fallback on failures.
"""

import requests
import json
import os
from typing import Dict, Any, Optional, List, Union
import logging
from dotenv import load_dotenv

from app.llm.chart_recommender import validate_and_normalize_chart

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Ollama configuration from environment variables with sensible defaults
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "30"))  # seconds



def _aggregate_prediction_stats(
    predictions: Union[List[Dict[str, Any]], Any]
) -> Dict[str, Any]:
    """
    Extract key statistics from predictions for LLM context.
    
    Handles predictions as list of dicts or similar structure.
    Returns aggregated summary for compact prompt.
    """
    try:
        if not isinstance(predictions, list):
            predictions = list(predictions) if hasattr(predictions, "__iter__") else [predictions]
        
        if not predictions:
            return {"count": 0, "summary": "No predictions provided"}
        
        # Extract salary data
        salaries = []
        experience_levels = {}
        job_titles = {}
        
        for pred in predictions:
            if isinstance(pred, dict):
                if "predicted_salary_usd" in pred:
                    salaries.append(pred["predicted_salary_usd"])
                    
                    # Aggregate by experience level
                    exp_level = pred.get("experience_level", "Unknown")
                    if exp_level not in experience_levels:
                        experience_levels[exp_level] = []
                    experience_levels[exp_level].append(pred["predicted_salary_usd"])
                    
                    # Aggregate by job title
                    job_title = pred.get("job_title", "Unknown")
                    if job_title not in job_titles:
                        job_titles[job_title] = []
                    job_titles[job_title].append(pred["predicted_salary_usd"])
        
        if not salaries:
            return {"count": len(predictions), "summary": "No salary predictions found"}
        
        # Calculate statistics
        stats = {
            "count": len(salaries),
            "avg_salary": round(sum(salaries) / len(salaries), 2),
            "min_salary": min(salaries),
            "max_salary": max(salaries),
            "by_experience": {
                level: {
                    "count": len(sals),
                    "avg": round(sum(sals) / len(sals), 2)
                }
                for level, sals in experience_levels.items()
            },
            "top_titles_by_salary": [
                {
                    "title": title,
                    "count": len(sals),
                    "avg": round(sum(sals) / len(sals), 2)
                }
                for title, sals in sorted(
                    job_titles.items(),
                    key=lambda x: sum(x[1]) / len(x[1]),
                    reverse=True
                )[:5]
            ]
        }
        
        return stats
    except Exception as e:
        logger.warning(f"Failed to aggregate stats: {e}")
        return {"count": 0, "summary": "Failed to aggregate statistics"}


def _build_prompt(stats: Dict[str, Any]) -> str:
    """
    Build a compact prompt for llama3.2:1b.
    
    Includes prediction statistics and requested output format.
    Kept concise for small model performance.
    """
    prompt = f"""You are a data analyst reviewing salary predictions for data science roles.

Statistics from {stats.get('count', 0)} predictions:
- Average salary: ${stats.get('avg_salary', 0):,.0f}
- Range: ${stats.get('min_salary', 0):,.0f} - ${stats.get('max_salary', 0):,.0f}
- By experience level: {json.dumps(stats.get('by_experience', {}), indent=2)}
- Top paying job titles: {json.dumps(stats.get('top_titles_by_salary', [])[:3], indent=2)}

Generate a JSON response with exactly this structure (no markdown, no extra text):
{{
  "title": "One-sentence analysis title",
  "summary": "1-2 sentences grounding insights in the data above. Be specific about numbers and patterns.",
  "key_insights": ["insight 1", "insight 2", "insight 3"],
  "chart": {{
    "chart_type": "bar",
    "title": "Chart title",
    "x": "experience_level or job_title",
    "y": "predicted_salary_usd",
    "color": null,
    "aggregation": "mean",
    "description": "Why this chart matters"
  }}
}}

Return ONLY valid JSON, nothing else."""
    
    return prompt


def _parse_json_response(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract and parse JSON from LLM response.
    
    Attempts to find valid JSON even if response contains extra text.
    Returns None if parsing fails.
    """
    try:
        # Try direct parse first
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON in the response
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            json_str = text[start:end]
            return json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        pass
    
    logger.warning("Could not parse JSON from LLM response")
    return None


def _call_ollama(prompt: str) -> Optional[str]:
    """
    Call Ollama API with the prompt.
    
    Returns raw response text or None on failure.
    Logs errors but doesn't crash.
    """
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.3,  # Low temperature for consistency
            },
            timeout=OLLAMA_TIMEOUT
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "")
        else:
            logger.warning(f"Ollama returned status {response.status_code}")
            return None
            
    except requests.exceptions.ConnectionError:
        logger.warning(
            f"Could not connect to Ollama at {OLLAMA_BASE_URL}. "
            "Is Ollama running? (ollama serve)"
        )
        return None
    except requests.exceptions.Timeout:
        logger.warning(f"Ollama request timed out after {OLLAMA_TIMEOUT}s")
        return None
    except Exception as e:
        logger.warning(f"Error calling Ollama: {e}")
        return None


def _get_fallback_analysis() -> Dict[str, Any]:
    """
    Safe fallback structure when Ollama unavailable or fails.
    
    Provides minimal but valid analysis structure.
    """
    return {
        "title": "Salary Analysis",
        "summary": "LLM analysis unavailable. Local Ollama instance may not be running.",
        "key_insights": [
            "Analysis requires Ollama to be running locally.",
            "Start Ollama with: ollama serve",
            "Model used: llama3.2:1b (ensure it is pulled)"
        ],
        "chart": {
            "chart_type": "bar",
            "title": "Average Salary by Experience Level",
            "x": "experience_level",
            "y": "predicted_salary_usd",
            "color": None,
            "aggregation": "mean",
            "description": "Default chart: average salary by seniority"
        }
    }


def analyze_with_ollama(
    predictions: Union[List[Dict[str, Any]], Any],
    fallback: bool = True
) -> Dict[str, Any]:
    """
    Main entry point: Analyze predictions with local LLM.
    
    Args:
        predictions: List of prediction dicts with 'predicted_salary_usd' and other fields
        fallback: If True, return safe structure on Ollama failure. If False, return None.
    
    Returns:
        Structured analysis dict or fallback/None on failure
    
    Output structure:
        {
            "title": str,
            "summary": str,
            "key_insights": [str, str, str],
            "chart": {
                "chart_type": str,
                "title": str,
                "x": str,
                "y": str,
                "color": str or None,
                "aggregation": str,
                "description": str
            }
        }
    """
    try:
        # Aggregate stats from predictions
        stats = _aggregate_prediction_stats(predictions)
        
        # Build prompt
        prompt = _build_prompt(stats)
        
        # Call Ollama
        response_text = _call_ollama(prompt)
        
        if response_text is None:
            logger.warning("Ollama call failed, using fallback")
            return _get_fallback_analysis() if fallback else None
        
        # Parse JSON response
        parsed = _parse_json_response(response_text)
        
        if parsed is None:
            logger.warning("Could not parse JSON from response, using fallback")
            return _get_fallback_analysis() if fallback else None
        
        # Validate chart structure
        if "chart" not in parsed:
            logger.warning("No chart in response, using fallback")
            return _get_fallback_analysis() if fallback else None
        
        parsed["chart"] = validate_and_normalize_chart(parsed["chart"])
        
        return parsed
        
    except Exception as e:
        logger.error(f"Unexpected error in analyze_with_ollama: {e}")
        return _get_fallback_analysis() if fallback else None
