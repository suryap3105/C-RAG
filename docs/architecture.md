# C-RAG System Architecture

## Overview

The **Cognitive Retrieval-Augmented Generation (C-RAG)** system is a hybrid question-answering architecture that combines dense vector retrieval, knowledge graph traversal, neural reranking, and iterative reasoning to answer complex multi-hop questions.

---

## System Components

### 1. Hybrid Retrieval Module (HRM)

**Purpose**: Retrieve relevant candidates from both vector and graph modalities.

**Components**:
- **Vector Store** (FAISS): Dense embeddings of entity descriptions
- **Knowledge Graph**: Entity-relation-entity triples with neighbor expansion
- **Retrieval Strategy**: Union of vector top-k and graph 1-hop neighbors

**Code**: `src/crag/retrieval/hybrid.py`

```python
class HybridRetrievalModule:
    def retrieve_initial_candidates(query, k=10):
        # 1. Vector retrieval
        vector_results = vector_store.search(query, k)
        
        # 2. Seed entity extraction
        seed_entities = extract_entities(query)
        
        # 3. Graph neighbors
        graph_results = kg.get_neighbors(seed_entities)
        
        # Merge and deduplicate
        return merge(vector_results, graph_results)
```

---

### 2. ColBERT Reranker

**Purpose**: Contextualized scoring of candidates against the query.

**Architecture**: Cross-encoder that computes token-level similarity.

**Input**: Query text + Candidate text
**Output**: Relevance score ∈ [0, 1]

**Code**: `src/crag/model/colbert.py`

**Performance**: ~1.7s overhead per ranking batch (CPU inference)

---

### 3. Cognitive Retrieval Agent (CRA)

**Purpose**: Iterative multi-hop reasoning over retrieved candidates.

**Reasoning Loop**:
```
for hop in [1, 2, 3]:
    1. THINK: Analyze context, generate hypothesis
    2. ACT: Expand graph neighbors of top candidates
    3. OBSERVE: Rerank new candidates, update context
    
    if answer_found:
        return answer
```

**Termination Conditions**:
- `success`: Answer confidently identified
- `max_steps_reached`: Hop budget exhausted
- `exhausted_context`: No new candidates available
- `llm_error`: LLM parsing failure

**Code**: `src/crag/agent/cra.py`

---

### 4. LLM Backend

**Interface**: Unified API for multiple backends

**Implementations**:
- `MockLLMClient`: Deterministic responses for testing
- `OllamaClient`: Local LLM inference
- `OpenAIClient`: Cloud API (future)

**Responsibilities**:
- Generate reasoning hypotheses
- Predict next expansion targets
- Extract final answers from context

**Code**: `src/crag/llm/interface.py`

---

## Data Flow

```
Input Query
    ↓
┌───────────────────────────────────────┐
│  Hybrid Retrieval Module (HRM)       │
│  - Vector: top-k embeddings           │
│  - Graph: 1-hop from seed entities    │
└───────────────────────────────────────┘
    ↓
┌───────────────────────────────────────┐
│  ColBERT Reranker                     │
│  - Score each candidate               │
│  - Select top-5 for context           │
└───────────────────────────────────────┘
    ↓
┌───────────────────────────────────────┐
│  Cognitive Retrieval Agent (CRA)      │
│  ┌─────────────────────────────────┐  │
│  │ Hop 1:                          │  │
│  │  THINK → ACT → OBSERVE          │  │
│  ├─────────────────────────────────┤  │
│  │ Hop 2:                          │  │
│  │  THINK → ACT → OBSERVE          │  │
│  ├─────────────────────────────────┤  │
│  │ Hop 3:                          │  │
│  │  THINK → ACT → OBSERVE          │  │
│  └─────────────────────────────────┘  │
│  Termination: success / max_steps    │
└───────────────────────────────────────┘
    ↓
Final Answer
```

---

## Configuration System

All systems are configured via YAML files in `configs/`:

**Example**: `configs/crag_full.yaml`
```yaml
experiment_name: "crag_full_v2"
description: "Full C-RAG with Agentic Reasoning and ColBERT Reranking"

llm:
  provider: "mock"
  model: "llama3"

retrieval:
  k: 10
  type: "hybrid"
  use_reranker: true

agent:
  max_hops: 3
  max_expansions: 5
  reasoning: true
```

**Baseline Variants**:
- `baseline_vector.yaml`: Vector-only RAG
- `baseline_graph.yaml`: Static graph traversal
- `crag_no_rerank.yaml`: Agent without ColBERT
- `crag_full.yaml`: Complete system

---

## Evaluation Framework

**Experiment Manager**: `src/crag/evaluation/experiment_manager.py`

**Metrics Logged**:
```json
{
  "qid": "SQuAD.42",
  "question": "Who directed Inception?",
  "gold_answers": ["Christopher Nolan"],
  "prediction": "Christopher Nolan",
  "is_correct": true,
  "latency": 1.74,
  "hops": 2,
  "nodes_expanded": 5,
  "termination_reason": "success"
}
```

**Output**: `runs/<run_id>/metrics.jsonl`

---

## Analysis Pipeline

**1. Reranker Audit**
```bash
python src/crag/analysis/rerank_audit.py
```
Compares C-RAG Full vs No-Rerank to measure ColBERT's impact.

**2. Termination Analysis**
```bash
python src/crag/analysis/termination_audit.py
```
Aggregates termination reasons across systems.

**3. Results Aggregation**
```bash
python src/crag/analysis/analyze_results.py
```
Generates:
- `experiments/analysis_report.md` - Aggregate stats
- `experiments/pareto_curve.png` - Accuracy vs Latency

---

## Reproducibility

**Dependencies**: `requirements.txt`
- PyTorch, Transformers, Sentence-Transformers
- FAISS (CPU/GPU)
- PyYAML, Pandas, Matplotlib

**Environment**:
```bash
export PYTHONPATH=src
```

**Quick Start**:
```bash
# Run all baselines
powershell experiments/reproduce_table1.ps1

# Analyze results
python src/crag/analysis/analyze_results.py
```

---

## Design Decisions

**Why Hybrid Retrieval?**
- Vector search excels at semantic similarity
- Graph traversal captures structured relationships
- Union provides high recall for multi-hop questions

**Why ColBERT?**
- Token-level interaction captures fine-grained relevance
- More accurate than bag-of-words scoring
- Tradeoff: +1.7s latency for +X% accuracy

**Why Iterative Agent?**
- Single-hop retrieval insufficient for multi-hop questions
- Iterative expansion allows answer discovery across multiple hops
- LLM reasoning filters irrelevant paths

**Why MockLLM for Testing?**
- Deterministic, fast infrastructure validation
- Unblocks development while Ollama integration pending
- Ensures repeatability in CI/CD pipelines

---

## Future Extensions

**Performance**:
- GPU-accelerated ColBERT inference
- Batched retrieval for throughput
- Caching for repeated queries

**Capabilities**:
- Multi-document evidence aggregation
- Conversational follow-ups
- Explainable reasoning traces

**Scalability**:
- Distributed graph partitioning (METIS)
- Approximate nearest neighbor search (HNSW)
- Streaming evaluation for large datasets

---

## References

**Core Papers**:
- ColBERT: Khattab & Zaharia, 2020
- RAG: Lewis et al., 2020
- Chain-of-Thought: Wei et al., 2022

**Datasets**:
- SQuAD: Rajpurkar et al., 2018
- WebQSP: Yih et al., 2016
- MetaQA: Zhang et al., 2018
