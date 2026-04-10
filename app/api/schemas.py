"""
Pydantic models for request/response with display labels.

Validates inputs against full discovered values from data_contract.
Response includes both raw values (for model) and display labels (for UX).
"""

from pydantic import BaseModel, Field, field_validator
from typing import Dict, Any, Optional
from app.utils.data_contract import CATEGORICAL_FEATURES


# Display label mappings
EXPERIENCE_LEVEL_LABELS = {
    "EN": "Junior",
    "MI": "Intermediate", 
    "SE": "Senior",
    "EX": "Executive",
}

EMPLOYMENT_TYPE_LABELS = {
    "FT": "Full-time",
    "PT": "Part-time",
    "CT": "Contract",
    "FL": "Freelance",
}

COMPANY_SIZE_LABELS = {
    "S": "Small",
    "M": "Medium",
    "L": "Large",
}

REMOTE_RATIO_LABELS = {
    0: "On-site",
    50: "Hybrid",
    100: "Remote",
}


class ValueLabel(BaseModel):
    """A value with its human-readable label."""
    value: Any
    label: str


class PredictionRequest(BaseModel):
    """Validate prediction request against discovered values."""
    
    work_year: int = Field(..., description="Work year")
    experience_level: str = Field(..., description="EN/MI/SE/EX")
    employment_type: str = Field(..., description="FT/PT/CT/FL")
    job_title: str = Field(..., description="Job title")
    employee_residence: str = Field(..., description="2-letter country code")
    remote_ratio: int = Field(..., description="0/50/100")
    company_location: str = Field(..., description="2-letter country code")
    company_size: str = Field(..., description="S/M/L")
    
    @field_validator("work_year")
    @classmethod
    def validate_work_year(cls, v):
        allowed = CATEGORICAL_FEATURES["work_year"]
        if v not in allowed:
            raise ValueError(f"work_year must be one of {allowed}, got {v}")
        return v
    
    @field_validator("experience_level")
    @classmethod
    def validate_experience_level(cls, v):
        allowed = CATEGORICAL_FEATURES["experience_level"]
        if v not in allowed:
            raise ValueError(f"experience_level must be one of {allowed}, got '{v}'")
        return v
    
    @field_validator("employment_type")
    @classmethod
    def validate_employment_type(cls, v):
        allowed = CATEGORICAL_FEATURES["employment_type"]
        if v not in allowed:
            raise ValueError(f"employment_type must be one of {allowed}, got '{v}'")
        return v
    
    @field_validator("job_title")
    @classmethod
    def validate_job_title(cls, v):
        allowed = CATEGORICAL_FEATURES["job_title"]
        if v not in allowed:
            raise ValueError(f"job_title '{v}' not found in training data. Must exactly match one of the discovered titles.")
        return v
    
    @field_validator("employee_residence")
    @classmethod
    def validate_employee_residence(cls, v):
        allowed = CATEGORICAL_FEATURES["employee_residence"]
        if v not in allowed:
            # Helpful error hint
            if len(v) > 2 or v.isupper() and v not in allowed:
                hint = " (Did you mean a 2-letter country code like US, GB, DE, IN?)"
            else:
                hint = ""
            raise ValueError(f"employee_residence '{v}' not in training data.{hint} Valid codes: {sorted(allowed)}")
        return v
    
    @field_validator("remote_ratio")
    @classmethod
    def validate_remote_ratio(cls, v):
        allowed = CATEGORICAL_FEATURES["remote_ratio"]
        if v not in allowed:
            raise ValueError(f"remote_ratio must be one of {allowed} (0=On-site, 50=Hybrid, 100=Remote), got {v}")
        return v
    
    @field_validator("company_location")
    @classmethod
    def validate_company_location(cls, v):
        allowed = CATEGORICAL_FEATURES["company_location"]
        if v not in allowed:
            # Helpful error hint
            if len(v) > 2 or v.isupper() and v not in allowed:
                hint = " (Did you mean a 2-letter country code like US, GB, DE, IN?)"
            else:
                hint = ""
            raise ValueError(f"company_location '{v}' not in training data.{hint} Valid codes: {sorted(allowed)}")
        return v
    
    @field_validator("company_size")
    @classmethod
    def validate_company_size(cls, v):
        allowed = CATEGORICAL_FEATURES["company_size"]
        if v not in allowed:
            raise ValueError(f"company_size must be one of {allowed} (S=Small, M=Medium, L=Large), got '{v}'")
        return v


class InputsWithLabels(BaseModel):
    """Echoed inputs with display labels."""
    work_year: int
    experience_level: ValueLabel
    employment_type: ValueLabel
    job_title: ValueLabel
    employee_residence: ValueLabel
    remote_ratio: ValueLabel
    company_location: ValueLabel
    company_size: ValueLabel


class ModelInfo(BaseModel):
    """Model metadata."""
    model_name: str = "decision_tree_regressor"
    model_version: str = "v1"


class PredictionResponse(BaseModel):
    """Response with raw values + display labels."""
    inputs: InputsWithLabels
    predicted_salary_usd: float
    model_info: ModelInfo


class BatchPredictionRequest(BaseModel):
    """Batch prediction request with list of scenarios."""
    scenarios: list[dict[str, Any]] = Field(..., description="List of prediction scenarios")


class BatchPredictionItemResponse(BaseModel):
    """Single prediction result in batch response."""
    work_year: int
    experience_level: str
    employment_type: str
    job_title: str
    employee_residence: str
    remote_ratio: int
    company_location: str
    company_size: str
    predicted_salary_usd: float
    api_status: str = "success"


class BatchPredictionResponse(BaseModel):
    """Batch prediction response with results."""
    predictions: list[BatchPredictionItemResponse]
    total: int
    successful: int
    failed: int

