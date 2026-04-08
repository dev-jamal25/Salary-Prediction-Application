"""
Supabase schema migrations.

Creates three tables for v1:
- prediction_runs
- predictions
- llm_analyses
"""

# TODO: Write SQL DDL
# CREATE TABLE prediction_runs (
#   id BIGINT PRIMARY KEY,
#   run_name TEXT,
#   model_version TEXT,
#   dataset_version TEXT,
#   started_at TIMESTAMP,
#   completed_at TIMESTAMP,
#   status TEXT
# );
#
# CREATE TABLE predictions (
#   id BIGINT PRIMARY KEY,
#   run_id BIGINT REFERENCES prediction_runs(id),
#   work_year INT,
#   experience_level TEXT,
#   employment_type TEXT,
#   job_title TEXT,
#   employee_residence TEXT,
#   remote_ratio INT,
#   company_location TEXT,
#   company_size TEXT,
#   predicted_salary_usd FLOAT,
#   api_status TEXT,
#   created_at TIMESTAMP
# );
#
# CREATE TABLE llm_analyses (
#   id BIGINT PRIMARY KEY,
#   run_id BIGINT REFERENCES prediction_runs(id),
#   title TEXT,
#   summary TEXT,
#   key_insights JSONB,
#   chart_type TEXT,
#   chart_spec_json JSONB,
#   chart_data_json JSONB,
#   created_at TIMESTAMP
# );
