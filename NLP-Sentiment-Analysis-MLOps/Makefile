"""
Main Makefile for NLP MLOps project.
"""

.PHONY: help install test train serve deploy clean lint format

help:
	@echo "NLP Sentiment Analysis - MLOps Project (Local Mode)"
	@echo "====================================================="
	@echo "Available commands:"
	@echo ""
	@echo "Core Commands:"
	@echo "  make install       - Install dependencies"
	@echo "  make test          - Run tests"
	@echo "  make train         - Train model (local mode)"
	@echo "  make serve         - Start FastAPI server locally"
	@echo "  make lint          - Run code linting"
	@echo "  make format        - Format code with black"
	@echo "  make clean         - Clean up cache files"
	@echo ""
	@echo "MLOps Pipeline Commands:"
	@echo "  make demo          - Run complete system demo"
	@echo "  make pipeline      - Run full training pipeline"
	@echo "  make validate      - Run data quality validation"
	@echo "  make mlflow-ui     - Start MLflow UI (port 5000)"
	@echo "  make optimize      - Run hyperparameter optimization"
	@echo ""
	@echo "Setup Commands:"
	@echo "  make local-setup   - Setup for local development"
	@echo ""
	@echo "Local Mode Features:"
	@echo "  ✓ Multi-language support (10+ languages)"
	@echo "  ✓ Automated hyperparameter tuning (Optuna)"
	@echo "  ✓ MLflow experiment tracking"
	@echo "  ✓ Data quality validation"
	@echo "  ✓ Canary deployment strategy"
	@echo "  ✓ Vibecoding-styled visualizations"
	@echo "  ✓ Uses distilbert for faster training"
	@echo "  ✓ Reduced dataset size (500 samples)"
	@echo "  ✓ 1 epoch training by default"
	@echo "  ✓ No AWS/GPU requirements"

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --cov=src --cov-report=html

train:
	python scripts/train_pipeline.py --config config.yaml

serve:
	python -m uvicorn src.app.main:app --reload --host 0.0.0.0 --port 8001

lint:
	pylint src/ scripts/ || true
	flake8 src/ scripts/ || true

format:
	black src/ scripts/ tests/

local-setup:
	@echo "Setting up local development environment..."
	pip install -r requirements.txt
	mkdir -p data models logs pipeline_results
	@echo "Local setup complete! Run 'make demo' to see the system in action."

# MLOps Pipeline Commands
demo:
	@echo "Running complete system demo..."
	python scripts/demo_pipeline.py

pipeline:
	@echo "Running full training pipeline..."
	python -c "from src.pipeline.orchestrator import TrainingPipeline; p = TrainingPipeline(); r = p.run_full_pipeline(['Sample text'], validate=True); print('Pipeline completed')"

validate:
	@echo "Running data quality validation..."
	python -c "from scripts.demo_pipeline import demo_data_quality; demo_data_quality()"

optimize:
	@echo "Running hyperparameter optimization..."
	python -c "from scripts.demo_pipeline import demo_hyperparameter_optimization; demo_hyperparameter_optimization()"

mlflow-ui:
	@echo "Starting MLflow UI at http://localhost:5000"
	mlflow ui --host 127.0.0.1 --port 5000

# Cleanup
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf build dist *.egg-info
	@echo "Cleanup complete!"

deploy:
	bash aws/deploy.sh

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov dist build *.egg-info
	rm -rf mlruns

docker-build:
	docker build -t nlp-sentiment:latest .

docker-run:
	docker run -p 8000:8000 -e MODEL_PATH=/app/models/sentiment-model nlp-sentiment:latest

.DEFAULT_GOAL := help
