"""
Pydantic models for API request/response validation.

Validates inputs against discovered values from data_contract.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Dict, Any, Optional
from app.utils.data_contract import CATEGORICAL_FEATURES


class PredictionRequest(BaseModel):
    """
    Validated prediction request.
    
    All categorical inputs are validated against full discovered values
    (not scenario subsets).
    """
    work_year: int = Field(
        ..., 
        description="Work year (2020, 2021, 2022)"
    )
    experience_level: str = Field(
        ..., 
        description="Experience level (Junior, Intermediate, Expert, Director)"
    )
    employment_type: str = Field(
        ..., 
        description="Employment type (CT, FL, FT, PT)"
    )
    job_title: str = Field(
        ..., 
        description="Job title (must be one of discovered titles)"
    )
    employee_residence: str = Field(
        ..., 
        description="Employee residence (2-letter country code)"
    )
    remote_ratio: int = Field(
        ..., 
        description="Remote work ratio (0, 50, or 100)"
    )
    company_location: str = Field(
        ..., 
        description="Company location (2-letter country code)"
    )
    company_size: str = Field(
        ..., 
        description="Company size (S, M, L)"
    )
    
    @field_validator("work_year")
    @classmethod
    def validate_work_year(cls, v):
        allowed = CATEGORICAL_FEATURES["work_year"]
        if v not in allowed:
            raise ValueError(
                f"work_year must be one of {allowed}, got {v}"
            )
        return v
    
    @field_validator("experience_level")
    @classmethod
    def validate_experience_level(cls, v):
        allowed = CATEGORICAL_FEATURES["experience_level"]
        if v not in allowed:
            raise ValueError(
                f"experience_level must be one of {allowed}, got {v}"
            )
        return v
    
    @field_validator("employment_type")
    @classmethod
    def validate_employment_type(cls, v):
        allowed = CATEGORICAL_FEATURES["employment_type"]
        if v not in allowed:
            raise ValueError(
                f"employment_type must be one of {allowed}, got {v}"
            )
        return v
    
    @field_validator("job_title")
    @classmethod
    def validate_job_title(cls, v):
        allowed = CATEGORICAL_FEATURES["job_title"]
        if v not in allowed:
            raise ValueError(
                f"job_title '{v}' not found in discovered job titles. "
                f"Valid titles: {allowed}"
            )
        return v
    
    @field_validator("employee_residence")
    @classmethod
    def validate_employee_residence(cls, v):
        allowed = CATEGORICAL_FEATURES["employee_residence"]
        if v not in allowed:
            raise ValueError(
                f"employee_residence must be one of {allowed}, got {v}"
            )
        return v
    
    @field_validator("remote_ratio")
    @classmethod
    def validate_remote_ratio(cls, v):
        allowed = CATEGORICAL_FEATURES["remote_ratio"]
        if v not in allowed:
            raise ValueError(
                f"remote_ratio must be one of {allowed}, got {v}"
            )
        return v
    
    @field_validator("company_location")
    @classmethod
    def validate_company_location(cls, v):
        allowed = CATEGORICAL_FEATURES["company_location"]
        if v not in allowed:
            raise ValueError(
                f"company_location must be one of {allowed}, got {v}"
            )
        return v
    
    @field_validator("company_size")
    @classmethod
    def validate_company_size(cls, v):
        allowed = CATEGORICAL_FEATURES["company_size"]
        if v not in allowed:
            raise ValueError(
                f"company_size must be one of {allowed}, got {v}"
            )
        return v


class ModelInfo(BaseModel):
    """Model metadata for debugging."""
    model_name: str = "decision_tree_regressor"
    model_version: str = "v1"


class PredictionResponse(BaseModel):
    """
    Prediction response with validated inputs and prediction.
    
    Includes echoed inputs, predicted salary, and model metadata.
    """
    inputs: PredictionRequest
    predicted_salary_usd: float = Field(
        ..., 
        description="Predicted salary in USD"
    )
    model_info: ModelInfo
