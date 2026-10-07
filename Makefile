.PHONY: setup generate features train evaluate serve test all

setup:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

generate:
	python -m claims_risk.generate

features:
	python -m claims_risk.features

train:
	python -m claims_risk.train_gbm
	python -m claims_risk.train_torch

evaluate:
	python -m claims_risk.evaluate

serve:
	uvicorn claims_risk.serve:app --host 0.0.0.0 --port 8000

test:
	pytest -q --cov=src/claims_risk

all: setup generate features train evaluate test
