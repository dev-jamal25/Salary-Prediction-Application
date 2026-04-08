"""
Pydantic models for API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any


class PredictionRequest(BaseModel):
    """Validated prediction request."""
    work_year: int
    experience_level: str
    employment_type: str
    job_title: str
    employee_residence: str
    remote_ratio: int
    company_location: str
    company_size: str


class ModelInfo(BaseModel):
    """Model metadata."""
    model_name: str = "decision_tree_regressor"
    model_version: str = "v1"


class PredictionResponse(BaseModel):
    """Prediction response with validated inputs and prediction."""
    inputs: Dict[str, Any]
    predicted_salary_usd: float
    model_info: ModelInfo
