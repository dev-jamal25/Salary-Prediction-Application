"""
Data Contract: Feature Metadata and Allowed Values

Discovered during EDA phase.
Used for API input validation and scenario generation.
"""

from typing import Dict, List, Set

# Categorical features with allowed values (discovered from data)
EXPERIENCE_LEVEL_VALUES = ['EN', 'EX', 'MI', 'SE']
EMPLOYMENT_TYPE_VALUES = ['CT', 'FL', 'FT', 'PT']
COMPANY_SIZE_VALUES = ['L', 'M', 'S']

# Numeric features with allowed values
WORK_YEAR_VALUES = [2020, 2021, 2022]
REMOTE_RATIO_VALUES = [0, 50, 100]

# Categorical features with high cardinality (lists for reference)
JOB_TITLES = ['3D Computer Vision Researcher', 'AI Scientist', 'Analytics Engineer', 'Applied Data Scientist', 'Applied Machine Learning Scientist', 'BI Data Analyst', 'Big Data Architect', 'Big Data Engineer', 'Business Data Analyst', 'Cloud Data Engineer', 'Computer Vision Engineer', 'Computer Vision Software Engineer', 'Data Analyst', 'Data Analytics Engineer', 'Data Analytics Lead', 'Data Analytics Manager', 'Data Architect', 'Data Engineer', 'Data Engineering Manager', 'Data Science Consultant', 'Data Science Engineer', 'Data Science Manager', 'Data Scientist', 'Data Specialist', 'Director of Data Engineering', 'Director of Data Science', 'ETL Developer', 'Finance Data Analyst', 'Financial Data Analyst', 'Head of Data', 'Head of Data Science', 'Head of Machine Learning', 'Lead Data Analyst', 'Lead Data Engineer', 'Lead Data Scientist', 'Lead Machine Learning Engineer', 'ML Engineer', 'Machine Learning Developer', 'Machine Learning Engineer', 'Machine Learning Infrastructure Engineer', 'Machine Learning Manager', 'Machine Learning Scientist', 'Marketing Data Analyst', 'NLP Engineer', 'Principal Data Analyst', 'Principal Data Engineer', 'Principal Data Scientist', 'Product Data Analyst', 'Research Scientist', 'Staff Data Scientist']
EMPLOYEE_RESIDENCES = ['AE', 'AR', 'AT', 'AU', 'BE', 'BG', 'BO', 'BR', 'CA', 'CH', 'CL', 'CN', 'CO', 'CZ', 'DE', 'DK', 'DZ', 'EE', 'ES', 'FR', 'GB', 'GR', 'HK', 'HN', 'HR', 'HU', 'IE', 'IN', 'IQ', 'IR', 'IT', 'JE', 'JP', 'KE', 'LU', 'MD', 'MT', 'MX', 'MY', 'NG', 'NL', 'NZ', 'PH', 'PK', 'PL', 'PR', 'PT', 'RO', 'RS', 'RU', 'SG', 'SI', 'TN', 'TR', 'UA', 'US', 'VN']
COMPANY_LOCATIONS = ['AE', 'AS', 'AT', 'AU', 'BE', 'BR', 'CA', 'CH', 'CL', 'CN', 'CO', 'CZ', 'DE', 'DK', 'DZ', 'EE', 'ES', 'FR', 'GB', 'GR', 'HN', 'HR', 'HU', 'IE', 'IL', 'IN', 'IQ', 'IR', 'IT', 'JP', 'KE', 'LU', 'MD', 'MT', 'MX', 'MY', 'NG', 'NL', 'NZ', 'PK', 'PL', 'PT', 'RO', 'RU', 'SG', 'SI', 'TR', 'UA', 'US', 'VN']

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
CATEGORICAL_FEATURES = {
    "experience_level": EXPERIENCE_LEVEL_VALUES,
    "employment_type": EMPLOYMENT_TYPE_VALUES,
    "company_size": COMPANY_SIZE_VALUES,
    "work_year": WORK_YEAR_VALUES,
    "remote_ratio": REMOTE_RATIO_VALUES,
    "job_title": JOB_TITLES,  # Full list of all discovered titles
    "employee_residence": EMPLOYEE_RESIDENCES,  # Full list of all discovered residences
    "company_location": COMPANY_LOCATIONS,  # Full list of all discovered locations
}

# ============================================================================
# SCENARIO GENERATION: Top-N subsets for tractable batch prediction
# USE: Scenario generation only (not for API validation)
# ============================================================================
TOP_JOB_TITLES = ['Data Scientist', 'Data Engineer', 'Data Analyst', 'Machine Learning Engineer', 'Research Scientist']
TOP_COMPANY_LOCATIONS = ['US', 'GB', 'CA', 'DE', 'IN']
TOP_EMPLOYEE_RESIDENCES = ['US', 'GB', 'IN', 'CA', 'DE', 'FR', 'ES', 'GR', 'JP', 'PK']

CATEGORICAL_FEATURES_FOR_SCENARIOS = {
    "experience_level": EXPERIENCE_LEVEL_VALUES,  # All values (low cardinality: 4)
    "employment_type": EMPLOYMENT_TYPE_VALUES,  # All values (low cardinality: 4)
    "company_size": COMPANY_SIZE_VALUES,  # All values (low cardinality: 3)
    "work_year": WORK_YEAR_VALUES,  # All values (3)
    "remote_ratio": REMOTE_RATIO_VALUES,  # All values (3)
    "job_title": TOP_JOB_TITLES,  # Subset: Top 5 only for tractability
    "employee_residence": TOP_EMPLOYEE_RESIDENCES,  # Subset: Top 10 only for tractability
    "company_location": TOP_COMPANY_LOCATIONS,  # Subset: Top 5 only for tractability
}

# Salary range for sanity checks
SALARY_IN_USD_MIN = 2859
SALARY_IN_USD_MAX = 600000
