# Makefile for C-RAG V3
.PHONY: help install test lint format clean docker run

help:
	@echo "C-RAG V3 Development Commands:"
	@echo "  make install      - Install dependencies"
	@echo "  make test         - Run tests"
	@echo "  make lint         - Run linters"
	@echo "  make format       - Format code"
	@echo "  make clean        - Clean build artifacts"
	@echo "  make docker       - Build Docker image"
	@echo "  make run          - Run API server locally"
	@echo "  make load-test    - Run load tests"

install:
	poetry install

test:
	poetry run pytest tests/ -v --cov=crag --cov-report=html --cov-report=term

test-fast:
	poetry run pytest tests/ -v -m "not slow and not integration"

test-integration:
	poetry run pytest tests/test_integration.py -v --run-integration

test-all:
	poetry run pytest tests/ -v --run-slow --run-integration

lint:
	poetry run black --check src/ tests/
	poetry run isort --check-only src/ tests/
	poetry run mypy src/ --ignore-missing-imports

format:
	poetry run black src/ tests/
	poetry run isort src/ tests/

clean:
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache .coverage htmlcov/
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	find . -type f -name '*.pyc' -delete

docker:
	docker build -t crag-v3:latest .

docker-compose-up:
	docker-compose up -d

docker-compose-down:
	docker-compose down -v

run:
	poetry run python -m uvicorn crag.api.server:app --reload --host 0.0.0.0 --port 8000

run-interactive:
	poetry run python -m crag.run_exp interactive

run-experiment:
	poetry run python -m crag.run_exp experiment --dataset data/test.jsonl

load-test:
	poetry run locust -f tests/load_test.py --host http://localhost:8000

benchmark:
	poetry run pytest tests/test_performance.py --benchmark-only

preprocess:
	poetry run python scripts/preprocess_graph.py \
		--nodes data/nodes.jsonl \
		--edges data/edges.jsonl \
		--output data/graph.pt

train-gnn:
	poetry run python scripts/train_gnn.py \
		--graph_path data/graph.pt \
		--queries_path data/train_queries.json \
		--epochs 100

build-colbert:
	poetry run python scripts/build_colbert_matrices.py \
		--graph_path data/graph.pt \
		--output_path data/colbert.pt
