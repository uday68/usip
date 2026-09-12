.PHONY: help install compile train test run docker-up docker-down

help:
	@echo "USIP — Underwater Sonar Intelligence Platform"
	@echo "Available commands:"
	@echo "  make install     Install local dependencies in virtual env"
	@echo "  make compile     Compile unified leakage-resistant dataset"
	@echo "  make train       Train YOLOv8n detector on local GPU"
	@echo "  make test        Run automated test suite"
	@echo "  make run         Start FastAPI application on port 8000"
	@echo "  make docker-up   Start containerized backend and PostGIS database"
	@echo "  make docker-down Stop containerized services"

compile:
	python -m ml.data.compiler

train:
	python -m ml.training.train_detector --epochs 15 --batch 8 --device 0

test:
	pytest tests/

run:
	python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload

docker-up:
	docker-compose up -d --build

docker-down:
	docker-compose down

