"""
Train Decision Tree model using cleaned data.

Builds ColumnTransformer + DecisionTreeRegressor pipeline.
Saves artifacts: model.pkl and metadata.json
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
import joblib
import json
from pathlib import Path


# TODO: Implement training logic
# - Load cleaned data from data/processed/ds_salaries_clean.csv
# - Build ColumnTransformer with OneHotEncoder(handle_unknown="ignore")
# - Fit DecisionTreeRegressor to training data
# - Extract and save feature metadata
# - Save pipeline to artifacts/decision_tree_v1.pkl
# - Save metadata to artifacts/model_metadata_v1.json
