"""
Ensemble Model Comparison Experiment.

This script compares three regression models:
1. DecisionTreeRegressor (baseline)
2. Lasso (linear model)
3. VotingRegressor (ensemble combining both)

Purpose: Evaluate whether ensemble approach improves generalization.

Note: This is an isolated experiment. Does not overwrite baseline training.

Usage:
    python scripts/compare_ensemble_models.py
"""

import pandas as pd
import numpy as np
import csv
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso
from sklearn.ensemble import VotingRegressor
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
RESULTS_DIR = PROJECT_ROOT / "experiments"

# Model features
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

# Train/test split
TEST_SIZE = 0.2
RANDOM_STATE = 42


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
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    print(f"\n✓ Train/Test split (80/20):")
    print(f"  - Train: {len(X_train)} samples")
    print(f"  - Test: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test


def build_tree_pipeline():
    """Build pipeline for DecisionTreeRegressor (no scaling needed for tree)."""
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("num", "passthrough", NUMERIC_FEATURES),
        ]
    )
    
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", DecisionTreeRegressor(
                max_depth=5,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=RANDOM_STATE,
            )),
        ]
    )
    
    return pipeline


def build_lasso_pipeline():
    """Build pipeline for Lasso (with scaling for numeric features)."""
    
    # For Lasso, we need to scale numeric features
    # Categorical features are one-hot encoded (0/1), which are already on similar scale
    # But for best practice, we can also scale them or keep them as-is
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ]
    )
    
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", Lasso(
                alpha=1.0,
                max_iter=5000,
                random_state=RANDOM_STATE,
            )),
        ]
    )
    
    return pipeline


def build_voting_ensemble():
    """Build VotingRegressor ensemble combining DecisionTree and Lasso."""
    
    tree_pipeline = build_tree_pipeline()
    lasso_pipeline = build_lasso_pipeline()
    
    # VotingRegressor with equal weights
    ensemble = VotingRegressor(
        estimators=[
            ("tree", tree_pipeline),
            ("lasso", lasso_pipeline),
        ],
        weights=[1, 1],  # Equal weight to both models
    )
    
    return ensemble


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Evaluate model and return metrics."""
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    metrics = {
        "model": model_name,
        "train_rmse": train_rmse,
        "train_mae": train_mae,
        "train_r2": train_r2,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "test_r2": test_r2,
        "rmse_gap": test_rmse - train_rmse,
        "r2_gap": train_r2 - test_r2,
    }
    
    return metrics


def print_model_results(metrics):
    """Print results for a single model."""
    
    print(f"\n### {metrics['model']} Results")
    print(f"  Train RMSE: ${metrics['train_rmse']:,.2f}")
    print(f"  Train MAE:  ${metrics['train_mae']:,.2f}")
    print(f"  Train R²:   {metrics['train_r2']:.4f}")
    print(f"\n  Test RMSE:  ${metrics['test_rmse']:,.2f}")
    print(f"  Test MAE:   ${metrics['test_mae']:,.2f}")
    print(f"  Test R²:    {metrics['test_r2']:.4f}")
    print(f"\n  RMSE Gap:   ${metrics['rmse_gap']:,.2f}")
    print(f"  R² Gap:     {metrics['r2_gap']:.4f}")


def compare_models(all_metrics):
    """Compare all three models side-by-side."""
    
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    
    # Extract metrics for easier comparison
    tree_m = all_metrics[0]
    lasso_m = all_metrics[1]
    ensemble_m = all_metrics[2]
    
    print("\n┌─ Test Set Performance ────────────────────────────────────────────┐")
    print("│ Metric      │ DecisionTree    │ Lasso           │ VotingEnsemble")
    print("├─────────────┼─────────────────┼─────────────────┼─────────────────")
    
    # Test RMSE
    print(f"│ Test RMSE   │ ${tree_m['test_rmse']:>13,.2f} │ ${lasso_m['test_rmse']:>13,.2f} │ ${ensemble_m['test_rmse']:>13,.2f}")
    
    # Test MAE
    print(f"│ Test MAE    │ ${tree_m['test_mae']:>13,.2f} │ ${lasso_m['test_mae']:>13,.2f} │ ${ensemble_m['test_mae']:>13,.2f}")
    
    # Test R²
    print(f"│ Test R²     │ {tree_m['test_r2']:>16.4f} │ {lasso_m['test_r2']:>16.4f} │ {ensemble_m['test_r2']:>16.4f}")
    
    print("└─────────────┴─────────────────┴─────────────────┴─────────────────")
    
    print("\n┌─ Overfitting Analysis ────────────────────────────────────────────┐")
    print("│ Metric      │ DecisionTree    │ Lasso           │ VotingEnsemble")
    print("├─────────────┼─────────────────┼─────────────────┼─────────────────")
    
    # RMSE Gap
    print(f"│ RMSE Gap    │ ${tree_m['rmse_gap']:>13,.2f} │ ${lasso_m['rmse_gap']:>13,.2f} │ ${ensemble_m['rmse_gap']:>13,.2f}")
    
    # R² Gap
    print(f"│ R² Gap      │ {tree_m['r2_gap']:>16.4f} │ {lasso_m['r2_gap']:>16.4f} │ {ensemble_m['r2_gap']:>16.4f}")
    
    print("└─────────────┴─────────────────┴─────────────────┴─────────────────")
    
    # Analysis
    print("\n### Interpretation")
    
    # Best test RMSE
    best_rmse = min(tree_m['test_rmse'], lasso_m['test_rmse'], ensemble_m['test_rmse'])
    if ensemble_m['test_rmse'] == best_rmse:
        print(f"✓ Ensemble has best test RMSE: ${ensemble_m['test_rmse']:,.2f}")
    elif tree_m['test_rmse'] == best_rmse:
        print(f"• DecisionTree has best test RMSE: ${tree_m['test_rmse']:,.2f}")
    else:
        print(f"• Lasso has best test RMSE: ${lasso_m['test_rmse']:,.2f}")
    
    # Best test R²
    best_r2 = max(tree_m['test_r2'], lasso_m['test_r2'], ensemble_m['test_r2'])
    if ensemble_m['test_r2'] == best_r2:
        print(f"✓ Ensemble has best test R²: {ensemble_m['test_r2']:.4f}")
    elif tree_m['test_r2'] == best_r2:
        print(f"• DecisionTree has best test R²: {tree_m['test_r2']:.4f}")
    else:
        print(f"• Lasso has best test R²: {lasso_m['test_r2']:.4f}")
    
    # Overfitting
    best_rmse_gap = min(tree_m['rmse_gap'], lasso_m['rmse_gap'], ensemble_m['rmse_gap'])
    if ensemble_m['rmse_gap'] == best_rmse_gap:
        print(f"✓ Ensemble has smallest RMSE gap: ${ensemble_m['rmse_gap']:,.2f}")
    elif tree_m['rmse_gap'] == best_rmse_gap:
        print(f"• DecisionTree has smallest RMSE gap: ${tree_m['rmse_gap']:,.2f}")
    else:
        print(f"• Lasso has smallest RMSE gap: ${lasso_m['rmse_gap']:,.2f}")


def save_comparison_csv(all_metrics):
    """Save comparison results to CSV."""
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "model_comparison.csv"
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_metrics[0].keys()))
        writer.writeheader()
        for metrics in all_metrics:
            writer.writerow(metrics)
    
    print(f"\n✓ Comparison saved to {csv_path}")
    
    return csv_path


def main():
    """Main experiment entry point."""
    print("=" * 80)
    print("ENSEMBLE MODEL COMPARISON EXPERIMENT")
    print("=" * 80)
    
    # Load and prepare data
    print("\n### Loading Data")
    df = load_data()
    
    print("\n### Preparing Features and Target")
    X, y = prepare_data(df)
    
    print("\n### Train/Test Split")
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Train and evaluate DecisionTreeRegressor
    print("\n" + "=" * 80)
    print("MODEL 1: DECISION TREE REGRESSOR")
    print("=" * 80)
    print("\nPipeline Configuration:")
    print("  - Preprocessing: ColumnTransformer")
    print("    * Categorical: OneHotEncoder(handle_unknown='ignore')")
    print("    * Numeric: PassThrough (no scaling needed)")
    print("  - Model: DecisionTreeRegressor")
    print("    * max_depth=5, min_samples_split=10, min_samples_leaf=4")
    
    print("\n### Training DecisionTreeRegressor")
    tree_model = build_tree_pipeline()
    tree_model.fit(X_train, y_train)
    tree_metrics = evaluate_model(tree_model, X_train, X_test, y_train, y_test, "DecisionTreeRegressor")
    print_model_results(tree_metrics)
    
    # Train and evaluate Lasso
    print("\n" + "=" * 80)
    print("MODEL 2: LASSO REGRESSION")
    print("=" * 80)
    print("\nPipeline Configuration:")
    print("  - Preprocessing: ColumnTransformer")
    print("    * Categorical: OneHotEncoder(handle_unknown='ignore')")
    print("    * Numeric: StandardScaler (required for Lasso)")
    print("  - Model: Lasso")
    print("    * alpha=1.0, max_iter=5000")
    
    print("\n### Training Lasso")
    lasso_model = build_lasso_pipeline()
    lasso_model.fit(X_train, y_train)
    lasso_metrics = evaluate_model(lasso_model, X_train, X_test, y_train, y_test, "Lasso")
    print_model_results(lasso_metrics)
    
    # Train and evaluate VotingRegressor
    print("\n" + "=" * 80)
    print("MODEL 3: VOTING ENSEMBLE")
    print("=" * 80)
    print("\nEnsemble Configuration:")
    print("  - Base Model 1: DecisionTreeRegressor Pipeline")
    print("  - Base Model 2: Lasso Pipeline")
    print("  - Voting Strategy: Average predictions (equal weights)")
    print("  - Each base model has its own optimal preprocessing")
    
    print("\n### Training VotingRegressor Ensemble")
    ensemble_model = build_voting_ensemble()
    ensemble_model.fit(X_train, y_train)
    ensemble_metrics = evaluate_model(ensemble_model, X_train, X_test, y_train, y_test, "VotingRegressor")
    print_model_results(ensemble_metrics)
    
    # Compare models
    all_metrics = [tree_metrics, lasso_metrics, ensemble_metrics]
    compare_models(all_metrics)
    
    # Save results
    print("\n### Saving Results")
    csv_path = save_comparison_csv(all_metrics)
    
    # Summary
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to:")
    print(f"  - {csv_path}")
    print("\nNote: All baseline models remain unchanged in artifacts/")
    print("=" * 80)


if __name__ == "__main__":
    main()
