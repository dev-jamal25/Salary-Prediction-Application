"""FastAPI prediction API.

Two endpoints:
- GET /health: Status check
- GET /predict: Salary prediction

Model loads once at startup and is cached for all requests.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from pydantic import ValidationError
import pandas as pd

from . import loader
from .schemas import (
    PredictionRequest, PredictionResponse, InputsWithLabels, ValueLabel,
    EXPERIENCE_LEVEL_LABELS, EMPLOYMENT_TYPE_LABELS, COMPANY_SIZE_LABELS,
    REMOTE_RATIO_LABELS, ModelInfo
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup."""
    try:
        loader.verify_artifacts()
        print("Model artifacts loaded successfully")
    except Exception as e:
        print(f"Failed to load artifacts: {e}")
        raise
    yield


app = FastAPI(
    title="Salary Prediction API",
    description="Decision tree salary prediction",
    version="v1",
    lifespan=lifespan
)


@app.get("/health")
async def health():
    """Health check."""
    try:
        metadata = loader.load_metadata()
        return {
            "status": "ok",
            "model_version": metadata.get("model_version"),
            "model_type": metadata.get("model_type"),
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/predict", response_model=PredictionResponse)
async def predict(
    work_year: int = Query(...),
    experience_level: str = Query(...),
    employment_type: str = Query(...),
    job_title: str = Query(...),
    employee_residence: str = Query(...),
    remote_ratio: int = Query(...),
    company_location: str = Query(...),
    company_size: str = Query(...),
):
    """Predict salary from job parameters."""
    try:
        request = PredictionRequest(
            work_year=work_year,
            experience_level=experience_level,
            employment_type=employment_type,
            job_title=job_title,
            employee_residence=employee_residence,
            remote_ratio=remote_ratio,
            company_location=company_location,
            company_size=company_size,
        )
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    
    try:
        model = loader.load_model()
        metadata = loader.load_metadata()
        
        input_df = pd.DataFrame({
            "work_year": [request.work_year],
            "experience_level": [request.experience_level],
            "employment_type": [request.employment_type],
            "job_title": [request.job_title],
            "employee_residence": [request.employee_residence],
            "remote_ratio": [request.remote_ratio],
            "company_location": [request.company_location],
            "company_size": [request.company_size],
        })
        
        prediction = float(model.predict(input_df)[0])
        
        inputs_with_labels = InputsWithLabels(
            work_year=request.work_year,
            experience_level=ValueLabel(
                value=request.experience_level,
                label=EXPERIENCE_LEVEL_LABELS.get(request.experience_level, request.experience_level)
            ),
            employment_type=ValueLabel(
                value=request.employment_type,
                label=EMPLOYMENT_TYPE_LABELS.get(request.employment_type, request.employment_type)
            ),
            job_title=ValueLabel(value=request.job_title, label=request.job_title),
            employee_residence=ValueLabel(value=request.employee_residence, label=request.employee_residence),
            remote_ratio=ValueLabel(
                value=request.remote_ratio,
                label=REMOTE_RATIO_LABELS.get(request.remote_ratio, str(request.remote_ratio))
            ),
            company_location=ValueLabel(value=request.company_location, label=request.company_location),
            company_size=ValueLabel(
                value=request.company_size,
                label=COMPANY_SIZE_LABELS.get(request.company_size, request.company_size)
            ),
        )
        
        model_info = ModelInfo(
            model_name=metadata.get("model_type", "decision_tree_regressor"),
            model_version=metadata.get("model_version", "v1"),
        )
        
        return PredictionResponse(
            inputs=inputs_with_labels,
            predicted_salary_usd=prediction,
            model_info=model_info,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
