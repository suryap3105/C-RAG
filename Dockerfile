# Production Dockerfile for C-RAG V3
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==1.7.0

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Install dependencies (no dev packages in production)
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/

# Create directories
RUN mkdir -p /app/data /app/checkpoints /app/experiments

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose API port
EXPOSE 8000

# Run API server
CMD ["python", "-m", "uvicorn", "crag.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]


# Development stage
FROM base as development

RUN poetry install --no-interaction --no-ansi

# Install dev tools
RUN pip install pytest pytest-cov black isort mypy ipython

CMD ["python", "-m", "crag.run_exp", "interactive"]


# Testing stage
FROM development as testing

COPY tests/ ./tests/

RUN pytest tests/ -v --cov=crag --cov-report=html --cov-report=term

CMD ["pytest", "tests/", "-v"]
