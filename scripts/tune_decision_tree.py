"""
Hyperparameter Tuning Experiment for DecisionTreeRegressor.

This script tunes the baseline DecisionTreeRegressor to reduce overfitting
while keeping the preprocessing pipeline identical to the baseline.

Approach:
- Load the cleaned dataset
- Use identical preprocessing (ColumnTransformer + OneHotEncoder)
- GridSearchCV with KFold cross-validation (5 folds, shuffled, random_state=42)
- Tune: max_depth, min_samples_leaf, min_samples_split, ccp_alpha
- Compare baseline vs tuned model on test set

Does NOT overwrite the baseline model.

Usage:
    python scripts/tune_decision_tree.py
"""

import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings

warnings.filterwarnings("ignore")


# ============================================================================
# Configuration and Paths
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
CLEANED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "ds_salaries_clean.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Model features (must match data_contract.py and baseline train.py)
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

# Feature types (must match baseline)
NUMERIC_FEATURES = ["work_year", "remote_ratio"]
CATEGORICAL_FEATURES = [
    "experience_level",
    "employment_type",
    "job_title",
    "employee_residence",
    "company_location",
    "company_size",
]

# Cross-validation configuration
CV_FOLDS = 3
CV_RANDOM_STATE = 42

# Test/train split (same as baseline)
TEST_SIZE = 0.2
TRAIN_RANDOM_STATE = 42

# Baseline hyperparameters (from train.py)
BASELINE_MAX_DEPTH = 10
BASELINE_MIN_SAMPLES_SPLIT = 5
BASELINE_MIN_SAMPLES_LEAF = 2
BASELINE_CCP_ALPHA = 0.01


def load_data():
    """Load the cleaned dataset."""
    if not CLEANED_DATA_PATH.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {CLEANED_DATA_PATH}")
    
    df = pd.read_csv(CLEANED_DATA_PATH)
    print(f"✓ Loaded cleaned dataset: {len(df)} rows, {len(df.columns)} columns")
    
    return df


def prepare_data(df):
    """Prepare features and target for training."""
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
        X, y, test_size=TEST_SIZE, random_state=TRAIN_RANDOM_STATE
    )
    
    print(f"\n✓ Train/Test split (80/20):")
    print(f"  - Train: {len(X_train)} samples")
    print(f"  - Test: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test


def build_base_pipeline():
    """Build the base preprocessing pipeline (identical to baseline)."""
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("num", "passthrough", NUMERIC_FEATURES),
        ]
    )
    
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", DecisionTreeRegressor(random_state=TRAIN_RANDOM_STATE)),
        ]
    )
    
    return pipeline


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Evaluate model on train and test sets."""
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    return {
        "train": {"rmse": train_rmse, "mae": train_mae, "r2": train_r2},
        "test": {"rmse": test_rmse, "mae": test_mae, "r2": test_r2},
    }


def get_baseline_metrics(X_train, X_test, y_train, y_test):
    """Train and evaluate baseline model."""
    
    print("\n" + "=" * 70)
    print("BASELINE MODEL (from train.py)")
    print("=" * 70)
    print(f"\nBaseline Hyperparameters:")
    print(f"  - max_depth: {BASELINE_MAX_DEPTH}")
    print(f"  - min_samples_split: {BASELINE_MIN_SAMPLES_SPLIT}")
    print(f"  - min_samples_leaf: {BASELINE_MIN_SAMPLES_LEAF}")
    print(f"  - ccp_alpha: {BASELINE_CCP_ALPHA}")
    
    baseline_pipeline = build_base_pipeline()
    baseline_pipeline.set_params(
        model__max_depth=BASELINE_MAX_DEPTH,
        model__min_samples_split=BASELINE_MIN_SAMPLES_SPLIT,
        model__min_samples_leaf=BASELINE_MIN_SAMPLES_LEAF,
        model__ccp_alpha=BASELINE_CCP_ALPHA,
    )
    
    print("\n### Training Baseline Model")
    baseline_pipeline.fit(X_train, y_train)
    baseline_metrics = evaluate_model(baseline_pipeline, X_train, X_test, y_train, y_test)
    
    print("\nBaseline Metrics:")
    print(f"  Train RMSE: ${baseline_metrics['train']['rmse']:,.2f}")
    print(f"  Train MAE:  ${baseline_metrics['train']['mae']:,.2f}")
    print(f"  Train R²:   {baseline_metrics['train']['r2']:.4f}")
    print(f"  Test RMSE:  ${baseline_metrics['test']['rmse']:,.2f}")
    print(f"  Test MAE:   ${baseline_metrics['test']['mae']:,.2f}")
    print(f"  Test R²:    {baseline_metrics['test']['r2']:.4f}")
    
    # Calculate overfitting metrics
    rmse_diff = baseline_metrics['test']['rmse'] - baseline_metrics['train']['rmse']
    r2_diff = baseline_metrics['train']['r2'] - baseline_metrics['test']['r2']
    print(f"\n  Overfitting Indicators:")
    print(f"    - RMSE gap (test - train): ${rmse_diff:,.2f}")
    print(f"    - R² gap (train - test):  {r2_diff:.4f}")
    
    return baseline_metrics, baseline_pipeline


def tune_model(X_train, X_test, y_train, y_test):
    """Tune DecisionTreeRegressor using GridSearchCV with KFold CV."""
    
    print("\n" + "=" * 70)
    print("HYPERPARAMETER TUNING")
    print("=" * 70)
    
    # Define parameter grid
    # Strategy: start with conservative search space, focus on regularization
    param_grid = {
        "model__max_depth": [5, 7, 10, 12, 15],
        "model__min_samples_split": [3, 5, 7, 10],
        "model__min_samples_leaf": [1, 2, 3, 4],
        "model__ccp_alpha": [0.0, 0.001, 0.005],
    }
    
    print(f"\nParameter Grid:")
    print(f"  - max_depth: {param_grid['model__max_depth']}")
    print(f"  - min_samples_split: {param_grid['model__min_samples_split']}")
    print(f"  - min_samples_leaf: {param_grid['model__min_samples_leaf']}")
    print(f"  - ccp_alpha: {param_grid['model__ccp_alpha']}")
    
    total_combinations = (
        len(param_grid["model__max_depth"]) *
        len(param_grid["model__min_samples_split"]) *
        len(param_grid["model__min_samples_leaf"]) *
        len(param_grid["model__ccp_alpha"])
    )
    print(f"\n  Total combinations: {total_combinations}")
    print(f"  CV folds: {CV_FOLDS}")
    print(f"  Total fits: {total_combinations * CV_FOLDS}")
    
    # Build base pipeline for tuning
    tuning_pipeline = build_base_pipeline()
    
    # GridSearchCV with KFold cross-validation
    print("\n### Running GridSearchCV (this may take a minute)...")
    
    cv_splitter = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_RANDOM_STATE)
    
    grid_search = GridSearchCV(
        tuning_pipeline,
        param_grid,
        cv=cv_splitter,
        scoring="r2",  # Optimize for R² (handles negative values well)
        n_jobs=-1,  # Use all CPU cores
        verbose=0,
    )
    
    grid_search.fit(X_train, y_train)
    
    print("✓ GridSearchCV completed")
    
    # Get best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_r2 = grid_search.best_score_
    
    print(f"\n### Best Parameters Found:")
    print(f"  - max_depth: {best_params['model__max_depth']}")
    print(f"  - min_samples_split: {best_params['model__min_samples_split']}")
    print(f"  - min_samples_leaf: {best_params['model__min_samples_leaf']}")
    print(f"  - ccp_alpha: {best_params['model__ccp_alpha']}")
    print(f"  - Best CV R²: {best_cv_r2:.4f}")
    
    # Evaluate on test set
    tuned_metrics = evaluate_model(best_model, X_train, X_test, y_train, y_test)
    
    print(f"\nTuned Model Metrics (Test Set):")
    print(f"  Train RMSE: ${tuned_metrics['train']['rmse']:,.2f}")
    print(f"  Train MAE:  ${tuned_metrics['train']['mae']:,.2f}")
    print(f"  Train R²:   {tuned_metrics['train']['r2']:.4f}")
    print(f"  Test RMSE:  ${tuned_metrics['test']['rmse']:,.2f}")
    print(f"  Test MAE:   ${tuned_metrics['test']['mae']:,.2f}")
    print(f"  Test R²:    {tuned_metrics['test']['r2']:.4f}")
    
    # Calculate overfitting metrics
    rmse_diff = tuned_metrics['test']['rmse'] - tuned_metrics['train']['rmse']
    r2_diff = tuned_metrics['train']['r2'] - tuned_metrics['test']['r2']
    print(f"\n  Overfitting Indicators:")
    print(f"    - RMSE gap (test - train): ${rmse_diff:,.2f}")
    print(f"    - R² gap (train - test):  {r2_diff:.4f}")
    
    return best_model, best_params, best_cv_r2, tuned_metrics


def compare_models(baseline_metrics, tuned_metrics, best_params, best_cv_r2):
    """Compare baseline vs tuned model."""
    
    print("\n" + "=" * 70)
    print("BASELINE vs TUNED COMPARISON")
    print("=" * 70)
    
    print("\n┌─ Test Set Performance ─────────────────────────────────────┐")
    print("│ Metric              │ Baseline        │ Tuned           │ Change")
    print("├─────────────────────┼─────────────────┼─────────────────┼─────────")
    
    # RMSE
    base_rmse = baseline_metrics['test']['rmse']
    tuned_rmse = tuned_metrics['test']['rmse']
    rmse_change = tuned_rmse - base_rmse
    rmse_pct = (rmse_change / base_rmse) * 100
    print(f"│ Test RMSE           │ ${base_rmse:>13,.2f} │ ${tuned_rmse:>13,.2f} │ {rmse_pct:>+6.1f}%")
    
    # MAE
    base_mae = baseline_metrics['test']['mae']
    tuned_mae = tuned_metrics['test']['mae']
    mae_change = tuned_mae - base_mae
    mae_pct = (mae_change / base_mae) * 100
    print(f"│ Test MAE            │ ${base_mae:>13,.2f} │ ${tuned_mae:>13,.2f} │ {mae_pct:>+6.1f}%")
    
    # R²
    base_r2 = baseline_metrics['test']['r2']
    tuned_r2 = tuned_metrics['test']['r2']
    r2_change = tuned_r2 - base_r2
    print(f"│ Test R²             │ {base_r2:>16.4f} │ {tuned_r2:>16.4f} │ {r2_change:>+6.4f}")
    
    print("└─────────────────────┴─────────────────┴─────────────────┴─────────")
    
    print("\n┌─ Overfitting Analysis ────────────────────────────────────┐")
    print("│ Metric              │ Baseline        │ Tuned           │ Change")
    print("├─────────────────────┼─────────────────┼─────────────────┼─────────")
    
    # Train vs Test RMSE Gap
    base_rmse_gap = baseline_metrics['test']['rmse'] - baseline_metrics['train']['rmse']
    tuned_rmse_gap = tuned_metrics['test']['rmse'] - tuned_metrics['train']['rmse']
    rmse_gap_change = tuned_rmse_gap - base_rmse_gap
    rmse_gap_pct = (rmse_gap_change / base_rmse_gap) * 100
    print(f"│ RMSE Gap (T-Tr)     │ ${base_rmse_gap:>13,.2f} │ ${tuned_rmse_gap:>13,.2f} │ {rmse_gap_pct:>+6.1f}%")
    
    # Train vs Test R² Gap
    base_r2_gap = baseline_metrics['train']['r2'] - baseline_metrics['test']['r2']
    tuned_r2_gap = tuned_metrics['train']['r2'] - tuned_metrics['test']['r2']
    r2_gap_change = tuned_r2_gap - base_r2_gap
    print(f"│ R² Gap (Tr-Te)      │ {base_r2_gap:>16.4f} │ {tuned_r2_gap:>16.4f} │ {r2_gap_change:>+6.4f}")
    
    print("└─────────────────────┴─────────────────┴─────────────────┴─────────")
    
    print("\n### Interpretation")
    if rmse_gap_change < 0:
        print("✓ GOOD: RMSE gap decreased - model generalizes better")
    else:
        print("✗ WORSE: RMSE gap increased - model is more overfit")
    
    if r2_gap_change < 0:
        print("✓ GOOD: R² gap decreased - model generalizes better")
    else:
        print("✗ WORSE: R² gap increased - model is more overfit")
    
    if tuned_r2 > base_r2:
        print("✓ GOOD: Test R² improved - model predicts better")
    elif tuned_r2 == base_r2:
        print("→ SAME: Test R² unchanged")
    else:
        print("✗ WORSE: Test R² decreased - predictions degraded")


def save_artifacts(tuned_model, best_params, best_cv_r2, tuned_metrics):
    """Save tuned model and results."""
    
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save tuned model
    tuned_model_path = ARTIFACTS_DIR / "tuned_model.joblib"
    joblib.dump(tuned_model, tuned_model_path)
    print(f"\n✓ Tuned model saved to {tuned_model_path}")
    
    # Extract feature names for metadata
    preprocessor = tuned_model.named_steps["preprocessor"]
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_feature_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    all_feature_names = cat_feature_names + NUMERIC_FEATURES
    
    # Save metadata
    tuned_metadata = {
        "model_type": "DecisionTreeRegressor (Tuned)",
        "model_version": "v1_tuned",
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
        "tuned_hyperparameters": {
            "max_depth": int(best_params["model__max_depth"]),
            "min_samples_split": int(best_params["model__min_samples_split"]),
            "min_samples_leaf": int(best_params["model__min_samples_leaf"]),
            "ccp_alpha": float(best_params["model__ccp_alpha"]),
        },
        "cross_validation": {
            "folds": CV_FOLDS,
            "best_cv_r2": float(best_cv_r2),
        },
        "metrics": tuned_metrics,
        "feature_names_after_preprocessing": all_feature_names,
    }
    
    tuned_metadata_path = ARTIFACTS_DIR / "tuned_model_metadata.json"
    with open(tuned_metadata_path, "w") as f:
        json.dump(tuned_metadata, f, indent=2)
    print(f"✓ Tuned metadata saved to {tuned_metadata_path}")
    
    return tuned_model_path, tuned_metadata_path


def main():
    """Main tuning entry point."""
    print("=" * 70)
    print("DECISION TREE HYPERPARAMETER TUNING EXPERIMENT")
    print("=" * 70)
    
    # Load and prepare data
    print("\n### Loading Data")
    df = load_data()
    
    print("\n### Preparing Features and Target")
    X, y = prepare_data(df)
    
    print("\n### Train/Test Split")
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Get baseline metrics
    baseline_metrics, baseline_model = get_baseline_metrics(X_train, X_test, y_train, y_test)
    
    # Tune model
    tuned_model, best_params, best_cv_r2, tuned_metrics = tune_model(
        X_train, X_test, y_train, y_test
    )
    
    # Compare models
    compare_models(baseline_metrics, tuned_metrics, best_params, best_cv_r2)
    
    # Save artifacts
    print("\n### Saving Artifacts")
    tuned_model_path, tuned_metadata_path = save_artifacts(
        tuned_model, best_params, best_cv_r2, tuned_metrics
    )
    
    # Final summary
    print("\n" + "=" * 70)
    print("TUNING EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"\nArtifacts saved:")
    print(f"  1. {tuned_model_path}")
    print(f"  2. {tuned_metadata_path}")
    print("\nNote: Baseline model in artifacts/model.joblib is unchanged.")
    print("=" * 70)


if __name__ == "__main__":
    main()
