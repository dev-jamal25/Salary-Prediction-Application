"""
Chart validation and normalization for LLM-recommended visualizations.

Ensures chart objects are safe for Streamlit rendering.
Enforces structure and provides sensible defaults.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Allowed chart types for Streamlit rendering
ALLOWED_CHART_TYPES = {"bar", "line", "scatter", "area"}

# Required chart fields
REQUIRED_CHART_FIELDS = {"chart_type", "title", "x", "y", "aggregation"}

# Allowed aggregation methods
ALLOWED_AGGREGATIONS = {"mean", "median", "sum", "count", "min", "max"}


def validate_and_normalize_chart(chart_obj: Any) -> Dict[str, Any]:
    """
    Validate and normalize a chart recommendation object.
    
    Ensures:
    - Object is a dictionary
    - Has required fields (chart_type, title, x, y, aggregation)
    - chart_type is whitelisted
    - aggregation is valid
    - Returns exactly one chart (not a list)
    
    Args:
        chart_obj: Chart recommendation from LLM (dict or other type)
    
    Returns:
        Normalized, safe chart dict for Streamlit rendering
    
    Chart structure:
        {
            "chart_type": str (bar|line|scatter|area),
            "title": str,
            "x": str (field name),
            "y": str (field name),
            "color": str or None (optional grouping field),
            "aggregation": str (mean|median|sum|count|min|max),
            "description": str (why this chart matters)
        }
    """
    
    # Ensure we have a dict
    if not isinstance(chart_obj, dict):
        logger.warning(f"Chart is not a dict: {type(chart_obj)}, using defaults")
        return _get_default_chart()
    
    # Check for required fields
    missing = REQUIRED_CHART_FIELDS - set(chart_obj.keys())
    if missing:
        logger.warning(f"Chart missing required fields: {missing}, using defaults")
        return _get_default_chart()
    
    normalized = {}
    
    # Validate and normalize chart_type
    chart_type = str(chart_obj.get("chart_type", "")).lower().strip()
    if chart_type not in ALLOWED_CHART_TYPES:
        logger.warning(
            f"Invalid chart_type '{chart_type}', allowed: {ALLOWED_CHART_TYPES}. Defaulting to 'bar'"
        )
        chart_type = "bar"
    normalized["chart_type"] = chart_type
    
    # Validate and normalize title
    title = str(chart_obj.get("title", "")).strip()
    if not title:
        logger.warning("Chart title is empty, using default")
        title = "Salary Analysis"
    normalized["title"] = title
    
    # Validate and normalize x (field name)
    x = str(chart_obj.get("x", "")).strip()
    if not x:
        logger.warning("Chart x field is empty, defaulting to 'experience_level'")
        x = "experience_level"
    normalized["x"] = x
    
    # Validate and normalize y (metric)
    y = str(chart_obj.get("y", "")).strip()
    if not y:
        logger.warning("Chart y field is empty, defaulting to 'predicted_salary_usd'")
        y = "predicted_salary_usd"
    elif y != "predicted_salary_usd":
        # Warn if y is something other than salary
        logger.info(f"Chart y field is '{y}' (not salary)")
    normalized["y"] = y
    
    # Optional: color field (grouping dimension)
    color = chart_obj.get("color")
    if color is not None:
        color = str(color).strip()
        if not color:
            color = None
    normalized["color"] = color
    
    # Validate and normalize aggregation
    agg = str(chart_obj.get("aggregation", "")).lower().strip()
    if agg not in ALLOWED_AGGREGATIONS:
        logger.warning(
            f"Invalid aggregation '{agg}', allowed: {ALLOWED_AGGREGATIONS}. Defaulting to 'mean'"
        )
        agg = "mean"
    normalized["aggregation"] = agg
    
    # Optional: description
    description = str(chart_obj.get("description", "")).strip()
    if not description:
        description = f"Visualization of {y} by {x} with {agg} aggregation"
    normalized["description"] = description
    
    return normalized


def _get_default_chart() -> Dict[str, Any]:
    """
    Return a safe, sensible default chart when LLM recommendation is invalid.
    """
    return {
        "chart_type": "bar",
        "title": "Average Salary by Experience Level",
        "x": "experience_level",
        "y": "predicted_salary_usd",
        "color": None,
        "aggregation": "mean",
        "description": "Average predicted salary grouped by experience level (EN=Junior, MI=Intermediate, SE=Senior, EX=Executive)"
    }
