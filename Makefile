.PHONY: help venv install clean test eda train run-api run-pipeline run-dashboard

help:
	@echo "Available targets:"
	@echo "  make venv            - Create virtual environment"
	@echo "  make install         - Install dependencies in .venv"
	@echo "  make clean           - Remove artifacts and __pycache__"
	@echo "  make test            - Run pytest"
	@echo "  make eda             - Run EDA script (Phase 1)"
	@echo "  make train           - Train model (Phase 2)"
	@echo "  make run-api         - Run FastAPI server locally"
	@echo "  make run-pipeline    - Run end-to-end pipeline"
	@echo "  make run-dashboard   - Run Streamlit dashboard"

venv:
	python -m venv .venv
	@echo "Virtual environment created at .venv"
	@echo "Activate with: .venv\Scripts\activate (Windows) or source .venv/bin/activate (Unix)"

install: venv
	.venv\Scripts\pip install --upgrade pip
	.venv\Scripts\pip install -r requirements.txt
	@echo "Dependencies installed in .venv"

clean:
	@echo "Cleaning artifacts and cache..."
	-rmdir /s /q artifacts 2>nul
	-rmdir /s /q __pycache__ 2>nul
	-rmdir /s /q .pytest_cache 2>nul
	-rmdir /s /q data\processed 2>nul
	@echo "Cleaned."

test:
	.venv\Scripts\pytest tests/ -v

eda:
	.venv\Scripts\python scripts/eda.py

train:
	.venv\Scripts\python app/pipeline/train.py

run-api:
	.venv\Scripts\uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000

run-pipeline:
	.venv\Scripts\python scripts/run_pipeline.py

run-dashboard:
	.venv\Scripts\streamlit run app/dashboard/streamlit_app.py
