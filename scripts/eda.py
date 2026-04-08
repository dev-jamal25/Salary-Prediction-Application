"""
Exploratory Data Analysis (EDA) and Data Cleaning.

Main reproducible entry point for dataset validation and cleaning.
Produces:
- data/cleaning_report.md (EDA findings and decisions)
- data/processed/ds_salaries_clean.csv (cleaned dataset)
- app/utils/data_contract.py (feature metadata with discovered allowed values)

Usage:
    python scripts/eda.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from collections import Counter


# Configuration
RAW_DATA_PATH = Path(__file__).parent.parent / "data" / "raw" / "ds_salaries.csv"
PROCESSED_DATA_PATH = Path(__file__).parent.parent / "data" / "processed"
CONTRACT_PATH = Path(__file__).parent.parent / "app" / "utils" / "data_contract.py"

# Expected schema
EXPECTED_COLUMNS = [
    "work_year",
    "experience_level",
    "employment_type",
    "job_title",
    "salary",
    "salary_currency",
    "salary_in_usd",
    "employee_residence",
    "remote_ratio",
    "company_location",
    "company_size",
]

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


class EDAReport:
    """Accumulate EDA findings for structured reporting."""

    def __init__(self):
        self.findings = []
        self.transformations = []
        self.warnings = []
        self.data_quality = {}

    def add_finding(self, section, text):
        self.findings.append((section, text))

    def add_transformation(self, text):
        self.transformations.append(text)

    def add_warning(self, text):
        self.warnings.append(text)

    def to_markdown(self, df_original, df_clean):
        """Generate markdown report."""
        md = ["# Data Cleaning and EDA Report\n"]

        # Dataset Overview
        md.append("## 1. Dataset Overview\n")
        md.append(f"**Original rows:** {len(df_original)}")
        md.append(f"**Final rows:** {len(df_clean)}")
        md.append(f"**Columns:** {len(df_clean.columns)}\n")
        md.append("**Columns:** " + ", ".join(df_clean.columns) + "\n")

        # Data Quality
        md.append("\n## 2. Data Quality Checks\n")
        md.append("| Check | Result |")
        md.append("|-------|--------|")
        md.append(f"| Null count | {df_clean.isnull().sum().sum()} |")
        md.append(f"| Duplicates | {df_clean.duplicated().sum()} |")
        # Count only whitespace warnings (those starting with "Column")
        whitespace_warnings = [w for w in self.warnings if "whitespace" in w.lower() or "empty" in w.lower()]
        md.append(f"| Whitespace issues | {len(whitespace_warnings)} found |")
        md.append("")

        # Dtypes
        md.append("\n## 3. Data Types\n")
        md.append("| Column | Type | Sample |")
        md.append("|--------|------|--------|")
        for col in df_clean.columns:
            sample = str(df_clean[col].iloc[0])[:30]
            dtype = str(df_clean[col].dtype)
            md.append(f"| {col} | {dtype} | {sample} |")
        md.append("")

        # Categorical Distributions
        md.append("\n## 4. Categorical Features Distribution\n")
        for col in ["experience_level", "employment_type", "company_size", "remote_ratio"]:
            if col in df_clean.columns:
                counts = df_clean[col].value_counts().sort_index()
                md.append(f"\n### {col}")
                md.append("| Value | Count |")
                md.append("|-------|-------|")
                for val, count in counts.items():
                    md.append(f"| {val} | {count} |")

        # Top job titles
        md.append("\n### job_title (Top 10)")
        md.append("| Title | Count |")
        md.append("|-------|-------|")
        for title, count in df_clean["job_title"].value_counts().head(10).items():
            md.append(f"| {title} | {count} |")

        # Top locations
        md.append("\n### company_location (Top 10)")
        md.append("| Location | Count |")
        md.append("|----------|-------|")
        for loc, count in df_clean["company_location"].value_counts().head(10).items():
            md.append(f"| {loc} | {count} |")

        # Top residences
        md.append("\n### employee_residence (Top 10)")
        md.append("| Residence | Count |")
        md.append("|-----------|-------|")
        for res, count in df_clean["employee_residence"].value_counts().head(10).items():
            md.append(f"| {res} | {count} |")

        # Numeric sanity
        md.append("\n## 5. Numeric Features - Sanity Checks\n")
        md.append("### work_year")
        md.append(f"- Min: {df_clean['work_year'].min()}")
        md.append(f"- Max: {df_clean['work_year'].max()}")
        md.append(f"- Unique values: {sorted(df_clean['work_year'].unique().tolist())}\n")

        md.append("\n### remote_ratio")
        md.append(f"- Min: {df_clean['remote_ratio'].min()}")
        md.append(f"- Max: {df_clean['remote_ratio'].max()}")
        md.append(f"- Unique values: {sorted(df_clean['remote_ratio'].unique().tolist())}\n")

        md.append("\n### salary_in_usd")
        md.append(f"- Min: ${df_clean['salary_in_usd'].min():,.2f}")
        md.append(f"- Max: ${df_clean['salary_in_usd'].max():,.2f}")
        md.append(f"- Mean: ${df_clean['salary_in_usd'].mean():,.2f}")
        md.append(f"- Median: ${df_clean['salary_in_usd'].median():,.2f}")
        md.append(f"- Std: ${df_clean['salary_in_usd'].std():,.2f}\n")

        # Transformations
        if self.transformations:
            md.append("\n## 6. Transformations Applied\n")
            for i, trans in enumerate(self.transformations, 1):
                md.append(f"{i}. {trans}")
        else:
            md.append("\n## 6. Transformations Applied\n")
            md.append("No transformations needed - dataset is already clean.\n")

        # Warnings
        if self.warnings:
            md.append("\n## 7. Warnings / Issues Found\n")
            for warning in self.warnings:
                md.append(f"- {warning}")
        else:
            md.append("\n## 7. Warnings / Issues Found\n")
            md.append("None.\n")

        # Data Dictionary
        md.append("\n## 8. Data Dictionary (Final)\n")
        md.append("| Feature | Type | Allowed Values / Range |")
        md.append("|---------|------|------------------------|")
        
        for col in df_clean.columns:
            dtype = df_clean[col].dtype
            if col in ["experience_level", "employment_type", "company_size"]:
                values = sorted(df_clean[col].unique().tolist())
                md.append(f"| {col} | categorical | {values} |")
            elif col == "remote_ratio":
                values = sorted(df_clean[col].unique().tolist())
                md.append(f"| {col} | numeric | {values} |")
            elif col == "work_year":
                values = sorted(df_clean[col].unique())
                md.append(f"| {col} | numeric | {values} |")
            elif col == "job_title":
                count = df_clean[col].nunique()
                md.append(f"| {col} | categorical | {count} unique values |")
            elif col in ["employee_residence", "company_location"]:
                count = df_clean[col].nunique()
                md.append(f"| {col} | categorical | {count} unique values (ISO codes) |")
            elif col == "salary_in_usd":
                min_val = df_clean[col].min()
                max_val = df_clean[col].max()
                md.append(f"| {col} | numeric | ${min_val:,.0f} - ${max_val:,.0f} |")
            else:
                md.append(f"| {col} | {dtype} | (excluded from model) |")

        return "\n".join(md)


def verify_schema(df):
    """Verify all expected columns are present."""
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    extra = set(df.columns) - set(EXPECTED_COLUMNS)
    
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if extra:
        print(f"⚠ Extra columns not in schema: {extra}")
    
    print(f"✓ Schema verified: {len(df.columns)} columns")
    return True


def check_dtypes(df):
    """Check and report data types."""
    print("\n### Data Types")
    expected_numeric = ["work_year", "salary", "remote_ratio", "salary_in_usd"]
    expected_string = [
        "experience_level",
        "employment_type",
        "job_title",
        "salary_currency",
        "employee_residence",
        "company_location",
        "company_size",
    ]
    
    for col in expected_numeric:
        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"⚠ {col}: Expected numeric, got {df[col].dtype}")
    
    for col in expected_string:
        if not pd.api.types.is_object_dtype(df[col]):
            print(f"⚠ {col}: Expected object/string, got {df[col].dtype}")
    
    print(f"✓ Dtypes verified")
    return True


def check_nulls(df, report):
    """Check for null values."""
    print("\n### Null Checks")
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        print("Null values found:")
        print(null_counts[null_counts > 0])
        report.add_warning(f"Null values: {null_counts.sum()} total across dataset")
        return False
    print("✓ No null values found")
    return True


def check_duplicates(df, report):
    """Check for duplicate rows."""
    print("\n### Duplicate Checks")
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        print(f"Found {dup_count} duplicate rows")
        report.add_warning(f"Duplicates: {dup_count} rows")
        return False
    print("✓ No duplicate rows")
    return True


def check_whitespace_issues(df, report):
    """Check for leading/trailing whitespace in string columns."""
    print("\n### Whitespace Checks")
    issues = 0
    string_cols = df.select_dtypes(include=['object']).columns
    
    for col in string_cols:
        # Check for leading/trailing whitespace
        has_whitespace = df[col].astype(str).str.match(r'^\s|\s$').any()
        if has_whitespace:
            issues += 1
            report.add_warning(f"Column '{col}' has leading/trailing whitespace")
    
    # Check for empty strings
    for col in string_cols:
        empty_count = (df[col].astype(str).str.len() == 0).sum()
        if empty_count > 0:
            issues += 1
            report.add_warning(f"Column '{col}' has {empty_count} empty strings")
    
    if issues == 0:
        print("✓ No whitespace issues detected")
    else:
        print(f"⚠ {issues} whitespace issues detected")
    
    return issues == 0


def check_categorical_values(df, report):
    """Check unique values for categorical features."""
    print("\n### Categorical Value Checks")
    
    categorical_cols = {
        "experience_level": None,
        "employment_type": None,
        "company_size": None,
        "job_title": None,
        "employee_residence": None,
        "company_location": None,
    }
    
    for col in categorical_cols:
        unique_vals = df[col].unique()
        print(f"  {col}: {len(unique_vals)} unique values")
        categorical_cols[col] = unique_vals
    
    print("✓ Categorical checks complete")
    return categorical_cols


def check_numeric_sanity(df, report):
    """Check numeric sanity for work_year, remote_ratio, salary_in_usd."""
    print("\n### Numeric Sanity Checks")
    
    # work_year
    wy_min, wy_max = df["work_year"].min(), df["work_year"].max()
    print(f"  work_year: {wy_min} to {wy_max}")
    if wy_min < 1990 or wy_max > 2030:
        report.add_warning(f"work_year range unusual: {wy_min} to {wy_max}")
    
    # remote_ratio
    rr_min, rr_max = df["remote_ratio"].min(), df["remote_ratio"].max()
    rr_unique = sorted(df["remote_ratio"].unique())
    print(f"  remote_ratio: {rr_min} to {rr_max}, unique: {rr_unique}")
    if rr_min < 0 or rr_max > 100:
        report.add_warning(f"remote_ratio out of range [0, 100]: {rr_min} to {rr_max}")
    
    # salary_in_usd
    sal_min, sal_max = df["salary_in_usd"].min(), df["salary_in_usd"].max()
    sal_mean = df["salary_in_usd"].mean()
    print(f"  salary_in_usd: ${sal_min:,.0f} to ${sal_max:,.0f} (mean: ${sal_mean:,.0f})")
    if sal_min < 10000:
        report.add_warning(f"Unusually low salary: ${sal_min:,.0f}")
    if sal_max > 1000000:
        report.add_warning(f"Unusually high salary: ${sal_max:,.0f}")
    
    print("✓ Numeric sanity checks complete")
    return True


def clean_dataset(df, report):
    """Apply transformations if needed."""
    df = df.copy()
    
    # Note: Per requirements, don't invent unnecessary cleaning
    # We'll only trim whitespace if found, and remove true duplicates
    
    initial_rows = len(df)
    initial_cols = len(df.columns)
    
    # Drop accidental index column if present (e.g., Unnamed: 0)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
        report.add_transformation("Dropped accidental index column (Unnamed: 0)")
    
    # Remove duplicate rows if any exist
    if df.duplicated().sum() > 0:
        df = df.drop_duplicates()
        report.add_transformation(f"Removed {initial_rows - len(df)} duplicate rows")
    
    # Trim whitespace from string columns (if needed)
    string_cols = df.select_dtypes(include=['object']).columns
    for col in string_cols:
        df[col] = df[col].str.strip()
    
    if len(df) < initial_rows or any(df[col].astype(str).str.match(r'^\s|\s$').any() for col in string_cols):
        report.add_transformation("Trimmed leading/trailing whitespace from string columns")
    
    return df


def generate_data_contract(df):
    """Generate Python data contract with discovered allowed values."""
    
    # Get actual discovered values, converting numpy types to native Python types
    experience_level = [str(x) for x in sorted(df["experience_level"].unique())]
    employment_type = [str(x) for x in sorted(df["employment_type"].unique())]
    company_size = [str(x) for x in sorted(df["company_size"].unique())]
    work_year = [int(x) for x in sorted(df["work_year"].unique())]
    remote_ratio = [int(x) for x in sorted(df["remote_ratio"].unique())]
    job_titles = sorted([str(x) for x in df["job_title"].unique()])
    residences = sorted([str(x) for x in df["employee_residence"].unique()])
    locations = sorted([str(x) for x in df["company_location"].unique()])
    
    # Top-N subsets for scenario generation
    top_5_titles = [str(x) for x in df["job_title"].value_counts().head(5).index.tolist()]
    top_5_locations = [str(x) for x in df["company_location"].value_counts().head(5).index.tolist()]
    top_10_residences = [str(x) for x in df["employee_residence"].value_counts().head(10).index.tolist()]
    
    contract_code = f'''"""
Data Contract: Feature Metadata and Allowed Values

Discovered during EDA phase.
Used for API input validation and scenario generation.
"""

from typing import Dict, List, Set

# Categorical features with allowed values (discovered from data)
EXPERIENCE_LEVEL_VALUES = {repr(experience_level)}
EMPLOYMENT_TYPE_VALUES = {repr(employment_type)}
COMPANY_SIZE_VALUES = {repr(company_size)}

# Numeric features with allowed values
WORK_YEAR_VALUES = {repr(work_year)}
REMOTE_RATIO_VALUES = {repr(remote_ratio)}

# Categorical features with high cardinality (lists for reference)
JOB_TITLES = {repr(job_titles)}
EMPLOYEE_RESIDENCES = {repr(residences)}
COMPANY_LOCATIONS = {repr(locations)}

# Model schema
MODEL_TARGET = "salary_in_usd"

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

# ============================================================================
# API VALIDATION: Full discovered categorical values
# USE: Input validation in API, ensure only observed categories accepted
# ============================================================================
CATEGORICAL_FEATURES = {{
    "experience_level": EXPERIENCE_LEVEL_VALUES,
    "employment_type": EMPLOYMENT_TYPE_VALUES,
    "company_size": COMPANY_SIZE_VALUES,
    "work_year": WORK_YEAR_VALUES,
    "remote_ratio": REMOTE_RATIO_VALUES,
    "job_title": JOB_TITLES,  # Full list of all discovered titles
    "employee_residence": EMPLOYEE_RESIDENCES,  # Full list of all discovered residences
    "company_location": COMPANY_LOCATIONS,  # Full list of all discovered locations
}}

# ============================================================================
# SCENARIO GENERATION: Top-N subsets for tractable batch prediction
# USE: Scenario generation only (not for API validation)
# ============================================================================
TOP_JOB_TITLES = {repr(top_5_titles)}
TOP_COMPANY_LOCATIONS = {repr(top_5_locations)}
TOP_EMPLOYEE_RESIDENCES = {repr(top_10_residences)}

CATEGORICAL_FEATURES_FOR_SCENARIOS = {{
    "experience_level": EXPERIENCE_LEVEL_VALUES,  # All values (low cardinality: 4)
    "employment_type": EMPLOYMENT_TYPE_VALUES,  # All values (low cardinality: 4)
    "company_size": COMPANY_SIZE_VALUES,  # All values (low cardinality: 3)
    "work_year": WORK_YEAR_VALUES,  # All values (3)
    "remote_ratio": REMOTE_RATIO_VALUES,  # All values (3)
    "job_title": TOP_JOB_TITLES,  # Subset: Top 5 only for tractability
    "employee_residence": TOP_EMPLOYEE_RESIDENCES,  # Subset: Top 10 only for tractability
    "company_location": TOP_COMPANY_LOCATIONS,  # Subset: Top 5 only for tractability
}}

# Salary range for sanity checks
SALARY_IN_USD_MIN = {int(df["salary_in_usd"].min())}
SALARY_IN_USD_MAX = {int(df["salary_in_usd"].max())}
'''
    
    return contract_code


def main():
    """Main EDA entry point."""
    print("=" * 70)
    print("SALARY PREDICTION - EXPLORATORY DATA ANALYSIS")
    print("=" * 70)
    
    # Initialize report
    report = EDAReport()
    
    # Load data
    print(f"\nLoading data from {RAW_DATA_PATH}...")
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW_DATA_PATH}")
    
    df_original = pd.read_csv(RAW_DATA_PATH)
    print(f"✓ Loaded {len(df_original)} rows, {len(df_original.columns)} columns")
    
    # Verify schema
    print("\n### Schema Verification")
    verify_schema(df_original)
    
    # Check dtypes
    check_dtypes(df_original)
    
    # Check data quality
    check_nulls(df_original, report)
    check_duplicates(df_original, report)
    check_whitespace_issues(df_original, report)
    
    # Check categorical values
    categorical_dict = check_categorical_values(df_original, report)
    
    # Check numeric sanity
    check_numeric_sanity(df_original, report)
    
    # Clean dataset
    print("\n### Cleaning Dataset")
    df_clean = clean_dataset(df_original, report)
    print(f"✓ Cleaned dataset: {len(df_clean)} rows")
    
    # Save cleaned dataset
    PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
    clean_path = PROCESSED_DATA_PATH / "ds_salaries_clean.csv"
    df_clean.to_csv(clean_path, index=False)
    print(f"✓ Saved cleaned data to {clean_path}")
    
    # Generate and save cleaning report
    print("\n### Generating Report")
    report_md = report.to_markdown(df_original, df_clean)
    report_path = Path(__file__).parent.parent / "data" / "cleaning_report.md"
    report_path.write_text(report_md)
    print(f"✓ Saved report to {report_path}")
    
    # Generate and save data contract
    print("\n### Generating Data Contract")
    contract_code = generate_data_contract(df_clean)
    CONTRACT_PATH.write_text(contract_code)
    print(f"✓ Saved data contract to {CONTRACT_PATH}")
    
    # Summary
    print("\n" + "=" * 70)
    print("EDA SUMMARY")
    print("=" * 70)
    print(f"✓ Original rows: {len(df_original)}")
    print(f"✓ Clean rows: {len(df_clean)}")
    print(f"✓ Rows removed: {len(df_original) - len(df_clean)}")
    print(f"✓ Transformations applied: {len(report.transformations)}")
    print(f"✓ Warnings: {len(report.warnings)}")
    print("\nOutputs generated:")
    print(f"  1. {clean_path}")
    print(f"  2. {report_path}")
    print(f"  3. {CONTRACT_PATH}")
    print("\nReady for model training (Phase 3).")
    print("=" * 70)


if __name__ == "__main__":
    main()
