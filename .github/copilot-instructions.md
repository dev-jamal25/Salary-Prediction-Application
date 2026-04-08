# Copilot Instructions

You are helping build a Salary Prediction Application.

Project rules:
- Use Python.
- Keep code modular and easy to explain.
- Make small, reviewable changes.
- Prefer updating existing files over duplicating logic.
- After each task, summarize changed files, assumptions, and commands to run.
- Implement only this step and then stop. Do not proceed to the next phase until I approve the current one.

Architecture rules:
- Model target is `salary_in_usd`.
- Model input features are:
  - `work_year`
  - `experience_level`
  - `employment_type`
  - `job_title`
  - `employee_residence`
  - `remote_ratio`
  - `company_location`
  - `company_size`
- Do not use `salary`, `salary_currency`, or `salary_in_usd` as model inputs.
- Use `DecisionTreeRegressor`.
- FastAPI is a standalone deployed deliverable.
- Streamlit must read from Supabase only.
- Ollama is local only.
- Do not create a separate `analysis_charts` table in v1.
- Store chart JSON inside `llm_analyses`.

Workflow rules:
- Inspect the workspace before coding.
- If something is uncertain, choose a sensible default and document it.
- Do not generate the entire application in one step unless explicitly asked.