# Deploying C-RAG

## Docker Deployment (Production)

Use the provided Docker Compose setup to launch the C-RAG API service along with a local vector database (Chroma/FAISS) and Ollama sidecar.

### 1. Build and Run
```bash
docker-compose up --build
```

### 2. Access
- **API Documentation**: `http://localhost:8000/docs`
- **Query Endpoint**: `POST http://localhost:8000/query`

## Requirements
- Docker & Docker Compose
- NVIDIA GPU (Recommended for Ollama)
