# Salary Prediction Application

End-to-end ML pipeline project for predicting data science salaries and presenting results through a deployed dashboard.

## Project Goal

This project predicts expected salary in USD for data science roles using job-related inputs such as experience level, employment type, company size, location, and job title.

It includes:
- dataset cleaning and EDA
- Decision Tree model training
- a standalone FastAPI prediction API
- a Python script that calls the API
- local LLM analysis through Ollama
- persistence in Supabase
- a Streamlit dashboard that reads from Supabase only

## Dataset

Dataset columns:
- work_year
- experience_level
- employment_type
- job_title
- salary
- salary_currency
- salary_in_usd
- employee_residence
- remote_ratio
- company_location
- company_size

## Prediction Contract

### Target
- `salary_in_usd`

### Input Features
- `work_year`
- `experience_level`
- `employment_type`
- `job_title`
- `employee_residence`
- `remote_ratio`
- `company_location`
- `company_size`

### Excluded from model inputs
- `salary`
- `salary_currency`
- `salary_in_usd`

## Architecture

### Local pipeline
Raw data → EDA/cleaning → model training → local FastAPI calls → Ollama analysis → Supabase storage

### Deployed components
- Standalone FastAPI API
- Streamlit dashboard reading from Supabase only

## Tech Stack

- Python
- pandas
- scikit-learn
- FastAPI
- Streamlit
- Supabase
- Ollama
- joblib

## Repo Structure

```text
week1_assignment/
├── app/
│   ├── api/
│   ├── dashboard/
│   ├── llm/
│   ├── pipeline/
│   ├── storage/
│   └── utils/
├── artifacts/
├── data/
│   ├── raw/
│   └── processed/
├── tests/
├── scripts/
├── docs/
├── .github/
└── README.md