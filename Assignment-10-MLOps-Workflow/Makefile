# Cross-platform task shortcuts for the MLOps workflow.
# Linux/macOS: make train | make test | make run | make docker-build | ...
# Windows (PowerShell): use the equivalent commands from README "Setup",
# or the cross-platform `python scripts/deploy.py`.

.PHONY: train smoke test app docker-build docker-run deploy verify clean ui

train:            ## full training + MLflow logging (4000 samples)
	python -m src.train

smoke:            ## fast smoke training (CI/bootstrap size)
	python -m src.train --smoke

test:             ## run the test suite
	python -m pytest tests -q

app:              ## run the API locally with uvicorn
	python -m uvicorn src.app:app --host 0.0.0.0 --port 8000

docker-build:     ## build the prediction-service image
	docker build -t techpulse-mlops:1.0.0 .

docker-run:       ## run the API in a container
	docker run --rm -p 8000:8000 --name techpulse-ml-api techpulse-mlops:1.0.0

deploy:           ## full local deployment simulation (build+run+verify)
	python scripts/deploy.py

verify:           ## verify an already-running service
	python scripts/healthcheck.py

ui:               ## launch the MLflow UI against ./mlruns (mlflow 3 needs the opt-in flag)
	MLFLOW_ALLOW_FILE_STORE=true mlflow ui --backend-store-uri ./mlruns
# Windows PowerShell equivalent:
#   $env:MLFLOW_ALLOW_FILE_STORE="true"; mlflow ui --backend-store-uri ./mlruns

clean:            ## remove caches and generated artifacts (keeps .gitkeep)
	rm -rf .pytest_cache src/__pycache__ tests/__pycache__ scripts/__pycache__
	rm -f artifacts/*.joblib artifacts/*.json artifacts/evaluation.txt
	rm -rf mlruns/TechPulse-Assignment-10
