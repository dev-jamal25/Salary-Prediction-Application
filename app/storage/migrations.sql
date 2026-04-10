-- Salary Prediction Application Schema
-- Tables: prediction_runs, predictions, llm_analyses

-- Create prediction_runs table
CREATE TABLE IF NOT EXISTS prediction_runs (
    id BIGSERIAL PRIMARY KEY,
    run_name TEXT,
    model_name TEXT NOT NULL DEFAULT 'decision_tree',
    model_version TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'completed', 'failed')),
    scenario_count INT NOT NULL DEFAULT 0,
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create predictions table (row-level data for each prediction)
CREATE TABLE IF NOT EXISTS predictions (
    id BIGSERIAL PRIMARY KEY,
    run_id BIGINT NOT NULL REFERENCES prediction_runs(id) ON DELETE CASCADE,
    work_year INT NOT NULL,
    experience_level TEXT NOT NULL,
    employment_type TEXT NOT NULL,
    job_title TEXT NOT NULL,
    employee_residence TEXT NOT NULL,
    remote_ratio INT NOT NULL,
    company_location TEXT NOT NULL,
    company_size TEXT NOT NULL,
    predicted_salary_usd NUMERIC(12, 2) NOT NULL,
    api_status TEXT DEFAULT 'success',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create llm_analyses table (LLM-generated analysis for each run)
CREATE TABLE IF NOT EXISTS llm_analyses (
    id BIGSERIAL PRIMARY KEY,
    run_id BIGINT NOT NULL REFERENCES prediction_runs(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    summary TEXT NOT NULL,
    key_insights_json JSONB NOT NULL,
    chart_spec_json JSONB NOT NULL,
    raw_llm_response_json JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_predictions_run_id ON predictions(run_id);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at);
CREATE INDEX IF NOT EXISTS idx_predictions_experience_level ON predictions(experience_level);
CREATE INDEX IF NOT EXISTS idx_predictions_job_title ON predictions(job_title);

CREATE INDEX IF NOT EXISTS idx_llm_analyses_run_id ON llm_analyses(run_id);
CREATE INDEX IF NOT EXISTS idx_llm_analyses_created_at ON llm_analyses(created_at);

CREATE INDEX IF NOT EXISTS idx_prediction_runs_status ON prediction_runs(status);
CREATE INDEX IF NOT EXISTS idx_prediction_runs_created_at ON prediction_runs(created_at);
