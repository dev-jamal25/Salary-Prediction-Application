"""
FastAPI application for salary predictions.

GET /predict endpoint with validated inputs.
"""

from fastapi import FastAPI, HTTPException
from .schemas import PredictionRequest, PredictionResponse

app = FastAPI(
    title="Salary Prediction API",
    description="Decision Tree-based salary prediction service",
    version="v1"
)


@app.get("/predict", response_model=PredictionResponse)
async def predict(
    work_year: int,
    experience_level: str,
    employment_type: str,
    job_title: str,
    employee_residence: str,
    remote_ratio: int,
    company_location: str,
    company_size: str,
):
    """
    Predict salary in USD for given job parameters.
    
    Query parameters are validated against discovered categorical values.
    """
    # TODO: Implement prediction logic
    # - Load model and features from artifact_manager
    # - Validate inputs against data_contract
    # - Return validated prediction response
    raise HTTPException(status_code=501, detail="Prediction endpoint not yet implemented")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "model_version": "v1"}
