FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# Copy configuration
COPY pyproject.toml poetry.lock* README.md /app/

# Install dependencies (no dev)
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --only main --no-root

# Copy source code
COPY src /app/src
COPY experiments /app/experiments
# Ensure src is in pythonpath
ENV PYTHONPATH=/app/src

# Run command
CMD ["uvicorn", "src.crag.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
