"""
Production REST API Server
FastAPI-based HTTP interface for C-RAG
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
import time
import uvicorn
from contextlib import asynccontextmanager

from crag import NeuroHybridRetrievalModule, GraphEngine, FaissVectorStore
from crag.model.gnn import NeuralSubgraphMatcher
from crag.routing.colbert import ColBERTPartitionRouter
from crag.model.query_graph import QueryGraphGenerator
from crag.llm.interface import create_llm_client
from crag.utils.monitoring import get_metrics_collector, get_health_check, PerformanceTracker
from crag.utils.resilience import with_retry, CircuitBreaker

logger = logging.getLogger(__name__)


# Request/Response Models
class QueryRequest(BaseModel):
    """Query request schema."""
    query: str = Field(..., description="Search query")
    k: int = Field(10, ge=1, le=100, description="Number of results")
    use_reranking: bool = Field(True, description="Apply hybrid reranking")
    include_metadata: bool = Field(True, description="Include metadata in results")


class RetrievalResult(BaseModel):
    """Single retrieval result."""
    id: Any
    text: str
    score: float
    source: str
    metadata: Optional[Dict[str, Any]] = None


class QueryResponse(BaseModel):
    """Query response schema."""
    results: List[RetrievalResult]
    query: str
    latency_ms: float
    model_version: str = "v3.0.0"


class HealthStatus(BaseModel):
    """Health check response."""
    status: str
    healthy: bool
    timestamp: str
    checks: Dict[str, Any]


class MetricsResponse(BaseModel):
    """Metrics response."""
    metrics: Dict[str, Dict[str, float]]


# Global state
pipeline: Optional[NeuroHybridRetrievalModule] = None
circuit_breaker = CircuitBreaker()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting C-RAG API server...")
    await initialize_pipeline()
    
    # Register health checks
    health = get_health_check()
    health.register("pipeline", lambda: pipeline is not None)
    health.register("vector_store", lambda: pipeline.vector_store is not None if pipeline else False)
    health.register("graph_engine", lambda: pipeline.graph_engine.data is not None if pipeline else False)
    
    logger.info("C-RAG API server ready")
    
    yield
    
    # Shutdown
    logger.info("Shutting down C-RAG API server...")


app = FastAPI(
    title="C-RAG V3 API",
    description="Neuro-Symbolic Knowledge Graph Retrieval API",
    version="3.0.0",
    lifespan=lifespan
)


async def initialize_pipeline():
    """Initialize the retrieval pipeline."""
    global pipeline
    
    try:
        # Load components (simplified for demo)
        ge = GraphEngine()
        vs = FaissVectorStore()
        llm = create_llm_client(provider='mock')
        query_gen = QueryGraphGenerator(llm)
        matcher = NeuralSubgraphMatcher()
        router = ColBERTPartitionRouter()
        
        pipeline = NeuroHybridRetrievalModule(
            vector_store=vs,
            graph_engine=ge,
            query_gen=query_gen,
            neural_matcher=matcher,
            colbert_router=router
        )
        
        logger.info("Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        raise


@app.middleware("http")
async def add_metrics_middleware(request: Request, call_next):
    """Middleware to track request metrics."""
    metrics = get_metrics_collector()
    start_time = time.time()
    
    try:
        response = await call_next(request)
        
        duration_ms = (time.time() - start_time) * 1000
        metrics.timing("http.request", duration_ms, {
            'method': request.method,
            'path': request.url.path,
            'status': str(response.status_code)
        })
        
        metrics.increment("http.requests.total", tags={
            'method': request.method,
            'status': str(response.status_code)
        })
        
        return response
    except Exception as e:
        metrics.increment("http.errors.total", tags={
            'method': request.method,
            'error': type(e).__name__
        })
        raise


@app.post("/query", response_model=QueryResponse)
@with_retry(max_attempts=2)
async def query(request: QueryRequest):
    """
    Execute retrieval query.
    
    Args:
        request: Query parameters
        
    Returns:
        Query results with metadata
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
        
    metrics = get_metrics_collector()
    
    try:
        with PerformanceTracker(metrics, "query.retrieval", {'use_reranking': str(request.use_reranking)}):
            # Execute with circuit breaker
            results = circuit_breaker.call(
                pipeline.retrieve,
                request.query,
                k=request.k,
                use_reranking=request.use_reranking
            )
            
        # Convert to response model
        response_results = [
            RetrievalResult(
                id=r.get('id'),
                text=r.get('text', ''),
                score=r.get('score', 0.0),
                source=r.get('source', 'unknown'),
                metadata=r.get('metadata') if request.include_metadata else None
            )
            for r in results
        ]
        
        return QueryResponse(
            results=response_results,
            query=request.query,
            latency_ms=metrics.get_stats("query.retrieval", window_seconds=1).get('p50', 0)
        )
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        metrics.increment("query.failures")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthStatus)
async def health():
    """
    Health check endpoint.
    
    Returns:
        Service health status
    """
    health_check = get_health_check()
    status = health_check.get_status()
    
    return HealthStatus(
        status="healthy" if status['healthy'] else "unhealthy",
        healthy=status['healthy'],
        timestamp=status['timestamp'],
        checks=status['checks']
    )


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    """
    Prometheus-compatible metrics endpoint.
    
    Returns:
        Metrics in Prometheus format
    """
    metrics_collector = get_metrics_collector()
    return metrics_collector.export_prometheus()


@app.get("/metrics/json", response_model=MetricsResponse)
async def metrics_json():
    """
    Get metrics in JSON format.
    
    Returns:
        Metrics statistics
    """
    metrics_collector = get_metrics_collector()
    return MetricsResponse(metrics=metrics_collector.get_all_metrics())


@app.get("/info")
async def info():
    """
    Get system information.
    
    Returns:
        System metadata
    """
    return {
        "name": "C-RAG V3",
        "version": "3.0.0",
        "description": "Neuro-Symbolic Knowledge Graph Retrieval",
        "components": {
            "graph_nodes": pipeline.graph_engine.data.num_nodes if pipeline and pipeline.graph_engine.data else 0,
            "vector_index_size": len(pipeline.vector_store.documents) if pipeline else 0
        }
    }


@app.post("/reload")
async def reload(background_tasks: BackgroundTasks):
    """
    Reload pipeline in background.
    
    Returns:
        Reload status
    """
    background_tasks.add_task(initialize_pipeline)
    return {"status": "reloading", "message": "Pipeline reload initiated"}


def start_server(host: str = "0.0.0.0", port: int = 8000, workers: int = 1):
    """Start the API server."""
    uvicorn.run(
        "crag.api.server:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info"
    )


if __name__ == "__main__":
    start_server()
