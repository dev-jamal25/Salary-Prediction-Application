End-to-end machine learning application for predicting data science salaries and presenting results through an interactive dashboard.

## Overview

This project predicts expected salary in USD for data science roles using job-related inputs such as experience level, employment type, company size, location, and job title.

It covers the full workflow from data preparation to model serving, analysis, storage, and visualisation.

## Features

- Data cleaning and exploratory data analysis
- Decision Tree regression model for salary prediction
- Standalone FastAPI prediction API
- Scenario-based batch prediction pipeline
- Local LLM analysis using Ollama
- Supabase storage for runs, predictions, and analysis
- Streamlit dashboard that reads from Supabase only

## Dataset

The model was trained and tested on Data Science Job Salaries from Kaggle: kaggle.com/datasets/ruchi798/data-science-job-salaries

The dataset includes the following columns:

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

## Workflow

1. Clean and validate the raw dataset
2. Train the salary prediction model
3. Serve predictions through FastAPI
4. Generate batch scenarios and collect predictions
5. Analyse prediction results with a local LLM
6. Store runs, predictions, and analysis in Supabase
7. Display results in Streamlit

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

## Running the Project

### 1. Set up the environment
Create a `.env` file and configure the required values for:
- Supabase
- Ollama
- FastAPI base URL

### 2. Train the model
Run the training pipeline to generate the model artifact and metadata.

### 3. Start the API
Launch the FastAPI app locally.

### 4. Run the pipeline
Execute the orchestration script to:
- generate scenarios
- call the API
- run LLM analysis
- persist outputs to Supabase

### 5. Launch the dashboard
Start the Streamlit dashboard to explore stored runs, predictions, and analysis.


## Repo Structure

```text
Salary-Prediction-Application/
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