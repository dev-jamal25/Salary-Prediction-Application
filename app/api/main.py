"""FastAPI prediction API.

Three endpoints:
- GET /health: Status check
- GET /predict: Single scenario prediction
- POST /predict-batch: Batch predictions for multiple scenarios

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
    REMOTE_RATIO_LABELS, ModelInfo,
    BatchPredictionRequest, BatchPredictionResponse, BatchPredictionItemResponse
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


@app.post("/predict-batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Predict salary for multiple scenarios in batch.
    
    Validates each scenario and returns predictions with raw values.
    Includes api_status for each prediction (success or error message).
    """
    if not request.scenarios:
        raise HTTPException(status_code=422, detail="Scenarios list cannot be empty")
    
    try:
        model = loader.load_model()
        metadata = loader.load_metadata()
        
        predictions_list = []
        successful_count = 0
        failed_count = 0
        
        # Process each scenario
        for scenario in request.scenarios:
            try:
                # Validate using PredictionRequest schema
                validated_input = PredictionRequest(**scenario)
                
                # Create DataFrame for prediction
                input_df = pd.DataFrame({
                    "work_year": [validated_input.work_year],
                    "experience_level": [validated_input.experience_level],
                    "employment_type": [validated_input.employment_type],
                    "job_title": [validated_input.job_title],
                    "employee_residence": [validated_input.employee_residence],
                    "remote_ratio": [validated_input.remote_ratio],
                    "company_location": [validated_input.company_location],
                    "company_size": [validated_input.company_size],
                })
                
                # Make prediction
                prediction_value = float(model.predict(input_df)[0])
                
                # Build response item with raw values only
                item = BatchPredictionItemResponse(
                    work_year=validated_input.work_year,
                    experience_level=validated_input.experience_level,
                    employment_type=validated_input.employment_type,
                    job_title=validated_input.job_title,
                    employee_residence=validated_input.employee_residence,
                    remote_ratio=validated_input.remote_ratio,
                    company_location=validated_input.company_location,
                    company_size=validated_input.company_size,
                    predicted_salary_usd=prediction_value,
                    api_status="success"
                )
                predictions_list.append(item)
                successful_count += 1
                
            except ValidationError as e:
                # Include failed scenario with error status
                item = BatchPredictionItemResponse(
                    work_year=scenario.get("work_year", 0),
                    experience_level=scenario.get("experience_level", ""),
                    employment_type=scenario.get("employment_type", ""),
                    job_title=scenario.get("job_title", ""),
                    employee_residence=scenario.get("employee_residence", ""),
                    remote_ratio=scenario.get("remote_ratio", 0),
                    company_location=scenario.get("company_location", ""),
                    company_size=scenario.get("company_size", ""),
                    predicted_salary_usd=0,
                    api_status=f"Validation error: {str(e)[:100]}"
                )
                predictions_list.append(item)
                failed_count += 1
                
            except Exception as e:
                # Include failed scenario with error status
                item = BatchPredictionItemResponse(
                    work_year=scenario.get("work_year", 0),
                    experience_level=scenario.get("experience_level", ""),
                    employment_type=scenario.get("employment_type", ""),
                    job_title=scenario.get("job_title", ""),
                    employee_residence=scenario.get("employee_residence", ""),
                    remote_ratio=scenario.get("remote_ratio", 0),
                    company_location=scenario.get("company_location", ""),
                    company_size=scenario.get("company_size", ""),
                    predicted_salary_usd=0,
                    api_status=f"Error: {str(e)[:100]}"
                )
                predictions_list.append(item)
                failed_count += 1
        
        return BatchPredictionResponse(
            predictions=predictions_list,
            total=len(predictions_list),
            successful=successful_count,
            failed=failed_count
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

