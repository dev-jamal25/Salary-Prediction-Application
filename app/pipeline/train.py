"""
Model Training Pipeline for Salary Prediction.

This script trains a DecisionTreeRegressor with scikit-learn preprocessing.
Produces:
- artifacts/model.joblib (trained model)
- artifacts/model_metadata.json (metadata for inference)

Usage:
    python app/pipeline/train.py
"""

import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# ============================================================================
# Configuration and Paths
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
CLEANED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "ds_salaries_clean.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Model features (must match data_contract.py)
MODEL_INPUT_FEATURES = [
    "work_year",
    "experience_level",
    "employment_type",
    "job_title",
    "employee_residence",
    "remote_ratio",
    "company_location",
    "company_size",
]

TARGET_FEATURE = "salary_in_usd"

# Feature types
NUMERIC_FEATURES = ["work_year", "remote_ratio"]
CATEGORICAL_FEATURES = [
    "experience_level",
    "employment_type",
    "job_title",
    "employee_residence",
    "company_location",
    "company_size",
]

# Training hyperparameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_DEPTH = 10  # Reasonable depth for a decision tree
MIN_SAMPLES_SPLIT = 5  # Prevent overfitting
MIN_SAMPLES_LEAF = 2
CCP_ALPHA = 0.001  # Complexity parameter for pruning (if needed)

def load_data():
    """Load the cleaned dataset."""
    if not CLEANED_DATA_PATH.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {CLEANED_DATA_PATH}")
    
    df = pd.read_csv(CLEANED_DATA_PATH)
    print(f"✓ Loaded cleaned dataset: {len(df)} rows, {len(df.columns)} columns")
    
    return df


def prepare_data(df):
    """Prepare features and target for training."""
    # Extract features and target
    X = df[MODEL_INPUT_FEATURES].copy()
    y = df[TARGET_FEATURE].copy()
    
    print(f"✓ Features shape: {X.shape}")
    print(f"✓ Target shape: {y.shape}")
    print(f"✓ Target range: ${y.min():,.0f} - ${y.max():,.0f}")
    print(f"✓ Target mean: ${y.mean():,.0f}")
    
    return X, y


def split_data(X, y):
    """Split data into train and test sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    print(f"\n✓ Train/Test split (80/20):")
    print(f"  - Train: {len(X_train)} samples")
    print(f"  - Test: {len(X_test)} samples")
    print(f"  - Train target mean: ${y_train.mean():,.0f}")
    print(f"  - Test target mean: ${y_test.mean():,.0f}")
    
    return X_train, X_test, y_train, y_test


def build_pipeline():
    """Build the preprocessing and model pipeline."""
    
    # Create column transformer for categorical features
    # OneHotEncoder with handle_unknown="ignore" to handle unseen categories at inference
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("num", "passthrough", NUMERIC_FEATURES),
        ]
    )
    
    # Build the full pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                DecisionTreeRegressor(
                    max_depth=MAX_DEPTH,
                    min_samples_split=MIN_SAMPLES_SPLIT,
                    min_samples_leaf=MIN_SAMPLES_LEAF,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    
    print("\n✓ Pipeline built:")
    print("  - Preprocessor: ColumnTransformer")
    print("    * Categorical: OneHotEncoder(handle_unknown='ignore')")
    print("    * Numeric: PassThrough")
    print("  - Model: DecisionTreeRegressor")
    
    return pipeline


def train_model(pipeline, X_train, y_train):
    """Train the model."""
    print("\n### Training Model")
    pipeline.fit(X_train, y_train)
    print("✓ Model trained successfully")
    
    return pipeline


def evaluate_model(pipeline, X_train, X_test, y_train, y_test):
    """Evaluate the model on train and test sets."""
    
    # Predictions
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)
    
    # Metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print("\n### Evaluation Metrics\n")
    print("Training Set:")
    print(f"  - MSE: ${train_mse:,.2f}")
    print(f"  - RMSE: ${train_rmse:,.2f}")
    print(f"  - MAE: ${train_mae:,.2f}")
    print(f"  - R²: {train_r2:.4f}")
    
    print("\nTest Set:")
    print(f"  - MSE: ${test_mse:,.2f}")
    print(f"  - RMSE: ${test_rmse:,.2f}")
    print(f"  - MAE: ${test_mae:,.2f}")
    print(f"  - R²: {test_r2:.4f}")
    
    metrics = {
        "train": {
            "mse": train_mse,
            "rmse": train_rmse,
            "mae": train_mae,
            "r2": train_r2,
        },
        "test": {
            "mse": float(test_mse),
            "rmse": float(test_rmse),
            "mae": float(test_mae),
            "r2": float(test_r2),
        }
    }
    
    return metrics


def save_artifacts(pipeline, metrics, total_samples):
    """Save model and metadata."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = ARTIFACTS_DIR / "model.joblib"
    joblib.dump(pipeline, model_path)
    print(f"\n✓ Model saved to {model_path}")
    
    # Prepare metadata
    # Get feature names after preprocessing
    preprocessor = pipeline.named_steps["preprocessor"]
    
    # Get categorical feature names after one-hot encoding
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_feature_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    
    # All feature names (one-hot encoded categoricals + numeric)
    all_feature_names = cat_feature_names + NUMERIC_FEATURES
    
    metadata = {
        "model_type": "DecisionTreeRegressor",
        "model_version": "v1",
        "target_feature": TARGET_FEATURE,
        "input_features": MODEL_INPUT_FEATURES,
        "feature_types": {
            "categorical": CATEGORICAL_FEATURES,
            "numeric": NUMERIC_FEATURES,
        },
        "preprocessor": {
            "type": "ColumnTransformer",
            "categorical_encoder": "OneHotEncoder(handle_unknown='ignore')",
            "numeric_handling": "passthrough",
        },
        "model_params": {
            "max_depth": MAX_DEPTH,
            "min_samples_split": MIN_SAMPLES_SPLIT,
            "min_samples_leaf": MIN_SAMPLES_LEAF,
            "random_state": RANDOM_STATE,
        },
        "training": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "total_samples": total_samples,
        },
        "metrics": metrics,
        "feature_names_after_preprocessing": all_feature_names,
    }
    
    # Save metadata
    metadata_path = ARTIFACTS_DIR / "model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved to {metadata_path}")
    
    return model_path, metadata_path


def verify_artifacts(model_path, metadata_path):
    """Verify that artifacts can be loaded successfully."""
    print("\n### Verifying Artifacts\n")
    
    # Load model
    try:
        loaded_model = joblib.load(model_path)
        print(f"✓ Model loaded successfully from {model_path}")
        print(f"  - Pipeline steps: {list(loaded_model.named_steps.keys())}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False
    
    # Load metadata
    try:
        with open(metadata_path, "r") as f:
            loaded_metadata = json.load(f)
        print(f"✓ Metadata loaded successfully from {metadata_path}")
        print(f"  - Model type: {loaded_metadata['model_type']}")
        print(f"  - Input features: {len(loaded_metadata['input_features'])}")
        print(f"  - Test R²: {loaded_metadata['metrics']['test']['r2']:.4f}")
    except Exception as e:
        print(f"✗ Failed to load metadata: {e}")
        return False
    
    # Test prediction with sample data
    try:
        sample_row = pd.DataFrame({
            "work_year": [2022],
            "experience_level": ["EN"],
            "employment_type": ["FT"],
            "job_title": ["Data Scientist"],
            "employee_residence": ["US"],
            "remote_ratio": [50],
            "company_location": ["US"],
            "company_size": ["M"],
        })
        prediction = loaded_model.predict(sample_row)
        print(f"✓ Sample prediction successful: ${prediction[0]:,.2f}")
    except Exception as e:
        print(f"✗ Sample prediction failed: {e}")
        return False
    
    return True


def main():
    """Main training entry point."""
    print("=" * 70)
    print("SALARY PREDICTION - MODEL TRAINING PIPELINE")
    print("=" * 70)
    
    # Load and prepare data
    print("\n### Loading Data")
    df = load_data()
    
    print("\n### Preparing Features and Target")
    X, y = prepare_data(df)
    
    # Split data
    print("\n### Train/Test Split")
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Build pipeline
    print("\n### Building Pipeline")
    pipeline = build_pipeline()
    
    # Train model
    pipeline = train_model(pipeline, X_train, y_train)
    
    # Evaluate model
    metrics = evaluate_model(pipeline, X_train, X_test, y_train, y_test)
    
    # Save artifacts
    print("\n### Saving Artifacts")
    model_path, metadata_path = save_artifacts(pipeline, metrics, len(df))
    
    # Verify artifacts
    verify_artifacts(model_path, metadata_path)
    
    # Summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nArtifacts saved:")
    print(f"  1. {model_path}")
    print(f"  2. {metadata_path}")
    print("\nReady for API deployment (Task 4).")
    print("=" * 70)


if __name__ == "__main__":
    main()

