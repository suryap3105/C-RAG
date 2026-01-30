# C-RAG V3: Neuro-Symbolic Knowledge Graph Retrieval

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**State-of-the-art graph-structured retrieval combining neural subgraph matching with semantic vector search.**

## 🌟 Key Features

### Novel Architecture
- **Neural Subgraph Matching**: GAT + GIN with multi-scale pooling for structural pattern recognition
- **Late-Interaction ColBERT Routing**: MaxSim-based partition selection with learned token representations
- **Adaptive Fusion**: Learned gating network dynamically balances vector and graph retrieval
- **Query Graph Generation**: LLM-based natural language to graph structure parsing
- **Adversarial Robustness**: Systematic noise injection for resilience testing

### Production-Ready
- FAISS-backed dense retrieval
- Multi-provider LLM support (Ollama, OpenAI, Anthropic)
- Comprehensive evaluation metrics (Recall, MRR, NDCG, F1)
- Parallel retrieval with thread pooling
- Hybrid reranking (ColBERT + Cross-Encoder)
- Full training pipeline with InfoNCE contrastive learning

## 🚀 Quick Start

### Installation

```bash
# Install with Poetry (recommended)
poetry install

# Or with pip
pip install -e .
```

### Basic Usage

```python
from crag import GraphEngine, FaissVectorStore, NeuroHybridRetrievalModule
from crag import QueryGraphGenerator, NeuralSubgraphMatcher, ColBERTPartitionRouter
from crag import create_llm_client

# 1. Load graph
graph_engine = GraphEngine()
graph_engine.load_graph("data/edges.jsonl", "data/nodes.jsonl")
graph_engine.compute_embeddings()

# 2. Build vector store
vector_store = FaissVectorStore()
vector_store.add_documents([
    {'id': i, 'text': info['text']}
    for i, info in graph_engine.node_text_map.items()
])

# 3. Initialize components
llm = create_llm_client(provider='mock')
query_gen = QueryGraphGenerator(llm)
neural_matcher = NeuralSubgraphMatcher()
router = ColBERTPartitionRouter()

# 4. Build retrieval pipeline
pipeline = NeuroHybridRetrievalModule(
    vector_store=vector_store,
    graph_engine=graph_engine,
    query_gen=query_gen,
    neural_matcher=neural_matcher,
    colbert_router=router
)

# 5. Retrieve
results = pipeline.retrieve("Who directed Inception?", k=10)
```

## 📊 Command Line Interface

### Run Experiments
```bash
# Standard evaluation
python -m crag.run_exp experiment --dataset data/metaqa.jsonl --config configs/default.yaml

# With noise injection
python -m crag.run_exp experiment --dataset data/metaqa.jsonl --noise_level 0.5

# Robustness testing
python -m crag.run_exp robustness --dataset data/metaqa.jsonl
```

### Interactive Mode
```bash
python -m crag.run_exp interactive --config configs/default.yaml
```

## 🧠 Architecture

```
Query → [Query Graph Generator] → Query Graph (G_Q)
                                      ↓
                    [ColBERT Router] → Partition Selection
                                      ↓
        ┌─────────────────────────────┴─────────────────────────┐
        ↓                                                         ↓
[Vector Search]                                        [Graph Search]
  FAISS E5                                          GAT+GIN Matching
        ↓                                                         ↓
        └─────────────────────→ [Adaptive Fusion] ←──────────────┘
                                  RRF + Gating
                                       ↓
                              [Hybrid Reranker]
                           ColBERT → Cross-Encoder
                                       ↓
                                  Final Results
```

## 🔬 Training

### Train Neural Matcher

```python
from crag.training.gnn_trainer import GNNTrainer, ContrastiveGraphDataset
from torch.utils.data import DataLoader

# Prepare data
dataset = ContrastiveGraphDataset(graph_engine, queries)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Train
trainer = GNNTrainer(neural_matcher)
trainer.train(train_loader, val_loader, epochs=100)
```

## 📈 Evaluation Metrics

- **Retrieval**: Recall@K, MRR, NDCG@10
- **QA**: Exact Match, F1 Score
- **Performance**: Latency (ms), Throughput (QPS)
- **Robustness**: Degradation curves under noise

## 🛡️ Robustness Testing

```python
from crag.evaluation.robustness import GraphMuddier

# Inject noise
muddier = GraphMuddier(graph_engine, {
    'phantom_nodes_ratio': 0.5,
    'bridge_noise_ratio': 0.3,
    'attribute_noise_std': 0.1
})
noisy_graph = muddier.apply()
```

## 📦 Components

| Component | Description | Key Technology |
|-----------|-------------|----------------|
| `GraphEngine` | Knowledge graph management | PyG, subgraph extraction |
| `FaissVectorStore` | Dense retrieval | FAISS, E5 embeddings |
| `NeuralSubgraphMatcher` | Structural matching | GAT, GIN, JK-Net |
| `ColBERTPartitionRouter` | Partition selection | Late interaction, MaxSim |
| `NeuroHybridRetrievalModule` | Fusion pipeline | Adaptive gating, RRF |
| `QueryGraphGenerator` | NL→Graph parsing | LLM prompting, NER fallback |
| `GraphPartitioner` | Graph partitioning | METIS, Leiden, Spectral |

## 🔧 Configuration

Edit `configs/default.yaml`:

```yaml
graph:
  nodes_path: data/nodes.jsonl
  edges_path: data/edges.jsonl

llm:
  provider: ollama  # or openai, anthropic
  model: llama3.2

gnn:
  hidden_channels: 256
  num_gin_layers: 4
```

---

##  Complete Project Structure

### Core Source Code (`src/crag/`)

#### Main Entry Point
- **`run_exp.py`** - CLI entry point for experiments, interactive mode, and robustness testing
  ```bash
  # Run experiment
  python -m crag.run_exp experiment --dataset data/test.jsonl
  
  # Interactive mode
  python -m crag.run_exp interactive
  
  # Robustness testing
  python -m crag.run_exp robustness --dataset data/test.jsonl
  ```

#### Graph Operations (`graph/`)
- **`engine.py`** - Core graph engine with FAISS embedding storage
  - Graph loading from JSONL files
  - K-hop subgraph extraction
  - Partition-aware retrieval
  - Save/load functionality
  
- **`partitioning.py`** - Graph partitioning algorithms
  - METIS graph cuts
  - Leiden community detection
  - Spectral clustering
  - Quality metrics computation

#### Neural Models (`model/`)
- **`gnn.py`** - Neural subgraph matcher
  - Multi-head GAT encoder (4 heads)
  - Deep GIN encoder (4 layers)
  - JK-Net multi-scale aggregation
  - Contrastive learning head
  
- **`query_graph.py`** - Natural language to graph structure
  - LLM-based query parsing
  - Schema validation
  - NER fallback
  
- **`cross_encoder.py`** - Reranking models
  - ColBERT MaxSim reranker
  - Cross-encoder reranker
  - Hybrid two-stage pipeline

#### Retrieval (`retrieval/`)
- **`vector_store.py`** - FAISS-based dense retrieval
  - E5 sentence embeddings
  - Batch indexing
  - Save/load with documents
  
- **`neural_hybrid.py`** - Main retrieval pipeline
  - Parallel vector + graph retrieval
  - Adaptive fusion with learned gating
  - Weighted RRF scoring
  - Optional reranking

#### Routing (`routing/`)
- **`colbert.py`** - ColBERT partition router
  - K-means token clustering
  - MaxSim late interaction
  - Top-K partition selection

#### LLM Interface (`llm/`)
- **`interface.py`** - Multi-provider LLM client
  - Mock LLM (for testing)
  - Ollama client
  - OpenAI client
  - Anthropic client

#### Training (`training/`)
- **`gnn_trainer.py`** - GNN training pipeline
  ```bash
  # Train neural matcher
  python scripts/train_gnn.py \
    --graph_path data/graph.pt \
    --queries_path data/train_queries.json \
    --epochs 100 \
    --batch_size 32
  ```
  - InfoNCE contrastive loss
  - AdamW optimizer
  - Cosine annealing scheduler
  - Checkpoint management

#### Evaluation (`evaluation/`)
- **`experiment_manager.py`** - Experiment orchestration
  - Recall@K, MRR, NDCG@10
  - Exact Match, F1 Score
  - Latency, throughput
  - Markdown report generation
  
- **`robustness.py`** - Adversarial noise injection
  - Phantom node injection
  - Bridge edge noise
  - Attribute perturbation
  - Edge dropping
  - Noise statistics

#### API (`api/`)
- **`server.py`** - FastAPI REST server
  ```bash
  # Start API server
  python -m uvicorn crag.api.server:app --reload --port 8000
  
  # Or with make
  make run
  ```
  - POST `/query` - Execute retrieval
  - GET `/health` - Health check
  - GET `/metrics` - Prometheus metrics
  - GET `/metrics/json` - JSON metrics
  - GET `/info` - System information

#### Utilities (`utils/`)
- **`resilience.py`** - Production resilience patterns
  - Circuit breakers (3-state)
  - Retry with exponential backoff
  - Timeout handling
  - Fallback strategies
  - Error budgets
  
- **`monitoring.py`** - Observability infrastructure
  - Metrics collection
  - Health checks
  - Distributed tracing
  - System monitoring (CPU/memory/disk)
  
- **`config.py`** - Configuration validation
  - Rule-based validation
  - Type checking
  - Range validation
  - Config merging
  
- **`data_quality.py`** - Data validation
  - Node/edge file validation
  - Schema checks
  - Integrity verification
  - Quality reports
  
- **`profiling.py`** - Performance profiling
  - Memory profiling
  - Time profiling
  - cProfile integration

---

## 🧪 Testing Infrastructure

### Unit Tests
```bash
# Run all tests
pytest tests/ -v

# Run specific test suite
pytest tests/test_graph_engine.py -v
pytest tests/test_gnn.py -v
```

- **`test_graph_engine.py`** - GraphEngine tests (16 tests)
  - Initialization, loading, embeddings
  - Subgraph extraction edge cases
  - Save/load, concurrency, memory efficiency
  
- **`test_gnn.py`** - Neural model tests (19 tests)
  - Forward pass, batching
  - Matching similar/dissimilar graphs
  - Gradient flow, eval mode
  - GPU compatibility, state dict
  
- **`test_query_gen.py`** - Query generator tests

### Integration Tests
```bash
# Run integration tests (requires --run-integration flag)
pytest tests/test_integration.py -v --run-integration
```

- **`test_integration.py`** - Full pipeline integration
  - End-to-end retrieval
  - Reranking pipeline
  - Concurrent queries
  - Save/load pipeline

### Property-Based Tests
```bash
pytest tests/test_property_based.py -v
```

- **`test_property_based.py`** - Hypothesis tests
  - GNN output normalization
  - Match symmetry
  - Self-similarity
  - Invariant properties

### Performance Tests
```bash
# Run benchmarks
pytest tests/test_performance.py -v --benchmark-only

# Or with make
make benchmark
```

- **`test_performance.py`** - Performance benchmarks
  - GNN forward speed
  - Vector search throughput
  - Subgraph extraction latency
  - Memory usage
  - Batch processing speedup
  - GPU acceleration

### Chaos Engineering
```bash
pytest tests/test_chaos.py -v
```

- **`test_chaos.py`** - Resilience testing
  - Random LLM failures
  - Network latency injection
  - Memory pressure
  - Concurrent load spikes
  - Circuit breaker activation
  - Gradual degradation

### Load Testing
```bash
# Run load test against running API
locust -f tests/load_test.py --host http://localhost:8000

# Or with make
make load-test
```

- **`load_test.py`** - Locust load testing
  - Multiple user profiles
  - Weighted task distribution
  - Automatic reporting

### Test Configuration
- **`conftest.py`** - pytest configuration
  - Fixtures (test_data_dir, device, mock_graph_data)
  - Custom markers (slow, integration, gpu)
  - Command-line options
  - Reproducible random seeds

---

## 📜 Preprocessing & Training Scripts

### Preprocess Graph
```bash
# Basic preprocessing
python scripts/preprocess_graph.py \
  --nodes data/nodes.jsonl \
  --edges data/edges.jsonl \
  --output data/graph.pt

# With partitioning
python scripts/preprocess_graph.py \
  --nodes data/nodes.jsonl \
  --edges data/edges.jsonl \
  --output data/graph.pt \
  --partition_method metis \
  --n_partitions 10

# Or with make
make preprocess
```

### Build ColBERT Matrices
```bash
python scripts/build_colbert_matrices.py \
  --graph_path data/graph.pt \
  --output_path data/colbert.pt \
  --num_tokens_per_partition 16

# Or with make
make build-colbert
```

### Train GNN
```bash
python scripts/train_gnn.py \
  --graph_path data/graph.pt \
  --queries_path data/train_queries.json \
  --epochs 100 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --checkpoint_dir checkpoints/

# Or with make
make train-gnn
```

### Verify Robustness (Windows)
```bash
python scripts/verify_robustness_win.py
```

---

## 📚 Example Scripts

### Basic Retrieval
```bash
python examples/basic_retrieval.py
```
Simple query execution example showing core retrieval functionality.

### Training Pipeline
```bash
python examples/train_pipeline.py
```
End-to-end example of training the neural subgraph matcher.

### Robustness Evaluation
```bash
python examples/robustness_eval.py
```
Example of testing system robustness under adversarial noise.

---

## 🐳 Docker Deployment

### Build Docker Image
```bash
# Build production image
docker build -t crag-v3:latest .

# Or with make
make docker
```

### Run with Docker Compose
```bash
# Start all services (API + Prometheus + Grafana + Redis)
docker-compose up -d

# View logs
docker-compose logs -f crag-api

# Stop services
docker-compose down -v

# Or with make
make docker-compose-up
make docker-compose-down
```

Services:
- **C-RAG API**: http://localhost:8000
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)
- **Redis**: localhost:6379

---

## 🛠️ Development Commands

The `Makefile` provides convenient commands for development:

```bash
# Installation & Setup
make install          # Install dependencies with Poetry
make clean           # Clean build artifacts

# Testing
make test            # Run all tests with coverage
make test-fast       # Skip slow and integration tests
make test-integration # Run integration tests only
make test-all        # Run all tests including slow

# Code Quality
make lint            # Run linters (black, isort, mypy)
make format          # Format code with black and isort

# Running
make run             # Start API server locally
make run-interactive # Run in interactive mode
make run-experiment  # Run evaluation experiment

# Load Testing
make load-test       # Run Locust load tests

# Benchmarking
make benchmark       # Run performance benchmarks

# Docker
make docker          # Build Docker image
make docker-compose-up   # Start Docker Compose stack
make docker-compose-down # Stop Docker Compose stack

# Data Preprocessing
make preprocess      # Preprocess graph data
make train-gnn       # Train neural matcher
make build-colbert   # Build ColBERT matrices
```

---

## ⚙️ Configuration Files

### `configs/default.yaml`
Main configuration file with all hyperparameters:
- Graph paths and partitioning settings
- Vector store configuration
- GNN architecture (layers, dimensions, dropout)
- LLM provider settings
- Training hyperparameters

### `pyproject.toml`
Poetry project configuration:
- Dependencies (core + optional)
- Dev dependencies
- Tool configurations (black, isort, mypy, pytest)
- Extras groups (partitioning, llm, monitoring)

### `requirements.txt`
pip-installable dependencies for non-Poetry setups.

---

## 🔄 CI/CD Pipeline

GitHub Actions workflow (`.github/workflows/ci-cd.yml`) includes:

1. **Lint & Format** - black, isort, mypy
2. **Multi-OS Testing** - Ubuntu, Windows, macOS
3. **Python Versions** - 3.10, 3.11
4. **Integration Tests** - Full pipeline verification
5. **Performance Benchmarks** - Automated benchmarking
6. **Docker Build** - Multi-stage image builds
7. **Security Scan** - Trivy vulnerability scanning
8. **Deployment** - Staging and production automation

---

## 📊 Monitoring & Observability

### Metrics
```bash
# View Prometheus metrics
curl http://localhost:8000/metrics

# View JSON metrics
curl http://localhost:8000/metrics/json
```

Tracked metrics:
- HTTP request duration, count, errors
- Query retrieval latency (p50, p95, p99)
- Success/failure counts
- System resources (CPU, memory, disk)

### Health Checks
```bash
curl http://localhost:8000/health
```

Returns component-level health status.

### Distributed Tracing
Automatic request tracing with span tracking for performance debugging.

---

## 🚦 Production Features

### Resilience
- **Circuit Breakers**: Prevent cascading failures
- **Retry Logic**: Exponential backoff on transient errors
- **Timeouts**: Enforce execution limits
- **Fallbacks**: Graceful degradation strategies
- **Error Budgets**: SLO tracking and enforcement

### Data Quality
```python
from crag.utils.data_quality import DataValidator

validator = DataValidator()
issues = validator.validate_nodes_file("data/nodes.jsonl")
print(validator.generate_report())
```

### Profiling
```python
from crag.utils.profiling import profile_memory, profile_time

with profile_memory():
    # Your code here
    pass

with profile_time("operation_name"):
    # Your code here
    pass
```

---

## 📝 Data Format

### Nodes File (`nodes.jsonl`)
```json
{"id": 0, "name": "Entity Name", "text": "Entity description"}
{"id": 1, "name": "Another Entity", "text": "Another description"}
```

### Edges File (`edges.jsonl`)
```json
{"src": 0, "dst": 1, "relation": "RELATIONSHIP_TYPE"}
{"src": 1, "dst": 2, "relation": "ANOTHER_TYPE"}
```

---

## 🎯 Common Workflows

### 1. Complete Setup from Scratch
```bash
# Install
poetry install

# Or on Windows
.\setup.ps1

# Preprocess data
python scripts/preprocess_graph.py \
  --nodes data/nodes.jsonl \
  --edges data/edges.jsonl \
  --output data/graph.pt

# Build ColBERT matrices
python scripts/build_colbert_matrices.py \
  --graph_path data/graph.pt \
  --output_path data/colbert.pt

# Train GNN (optional)
python scripts/train_gnn.py \
  --graph_path data/graph.pt \
  --queries_path data/train_queries.json

# Run interactive mode
python -m crag.run_exp interactive
```

### 2. Testing with Ollama
```bash
# Install and start Ollama
ollama pull mistral

# Update configs/default.yaml:
# llm:
#   provider: "ollama"
#   model: "mistral"

# Run interactive
python -m crag.run_exp interactive
```

### 3. Production Deployment
```bash
# Build and run with Docker Compose
docker-compose up -d

# Monitor health
curl http://localhost:8000/health

# View metrics in Grafana
# http://localhost:3000

# Test API
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Who directed Inception?", "k": 10}'
```

### 4. Running Full Evaluation
```bash
# Basic evaluation
python -m crag.run_exp experiment \
  --dataset data/test.jsonl \
  --config configs/default.yaml

# With robustness testing
python -m crag.run_exp robustness \
  --dataset data/test.jsonl

# Results saved to experiments/
```

---

## 🏗️ Architecture Deep Dive

### Component Flow

1. **Query Reception** → Natural language query
2. **Query Graph Generation** → LLM parses to graph structure
3. **Partition Routing** → ColBERT selects relevant partitions
4. **Parallel Retrieval**:
   - Vector: FAISS dense retrieval
   - Graph: GNN subgraph matching
5. **Adaptive Fusion** → Learned gating combines results
6. **Reranking** → ColBERT + CrossEncoder refinement
7. **Results** → Ranked, scored documents

### Key Technologies

- **PyTorch** + **PyTorch Geometric** - Deep learning
- **FAISS** - Dense vector indexing
- **Sentence Transformers** - Text embeddings (E5)
- **FastAPI** - REST API framework
- **Prometheus** + **Grafana** - Monitoring
- **Docker** - Containerization
- **GitHub Actions** - CI/CD
- **pytest** + **Hypothesis** - Testing
- **Locust** - Load testing

---

## 🤝 Enterprise Support

This implementation includes:
- ✅ Comprehensive test coverage (9 test suites)
- ✅ Production resilience (circuit breakers, retries, timeouts)
- ✅ Observability (metrics, health checks, tracing)
- ✅ API server with async processing
- ✅ Docker deployment with monitoring stack
- ✅ CI/CD pipeline with security scanning
- ✅ Data quality validation
- ✅ Performance profiling tools
- ✅ Chaos engineering tests
- ✅ Load testing infrastructure

---

## 📚 Citation

```bibtex
@article{crag2025,
  title={C-RAG V3: Neuro-Symbolic Knowledge Graph Retrieval},
  author={C-RAG Team},
  year={2025}
}
```

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

Contributions welcome! This system represents cutting-edge research in graph-based retrieval.

## 📧 Contact

For questions and support, please open an issue.
