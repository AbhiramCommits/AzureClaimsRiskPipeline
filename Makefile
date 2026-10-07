.PHONY: setup generate features baselines train evaluate batch-score serve test lint all

setup:
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .

generate:
	python -m claims_risk.generate
	python -m claims_risk.profile

features:
	python -m claims_risk.features

baselines:
	python -m claims_risk.baselines

train: baselines
	python -m claims_risk.train_gbm
	python -m claims_risk.registry
	python -m claims_risk.train_torch

evaluate:
	python -m claims_risk.evaluate

batch-score:
	python -m claims_risk.batch_score

serve:
	uvicorn claims_risk.serve:app --host 0.0.0.0 --port 8000

test:
	pytest -q --cov=src/claims_risk

lint:
	ruff check src/ tests/

all: setup generate features train evaluate test
