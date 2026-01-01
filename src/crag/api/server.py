from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

# Import our core modules
# Note: In a real deploy, these would be initialized appropriately
# We will do lazy loading or dependency injection in a real app
from ..agent.cra import CognitiveRetrievalAgent
from ..retrieval.hybrid import HybridRetrievalModule
from ..retrieval.vector_store import VectorStore, FaissVectorStore
from ..graph.wikidata import WikidataKG
from ..model.colbert import ColBERTReranker
from ..llm.interface import OllamaClient, MockLLMClient

app = FastAPI(title="C-RAG API", description="Cognitive Graph-RAG Reasoning Service", version="1.0.0")

# Global instances (simplified for demo)
# Global instances (simplified for demo)
kg = WikidataKG()
try:
    vs = FaissVectorStore()
except Exception:
    vs = None

hrm = HybridRetrievalModule(kg, vs)
reranker = ColBERTReranker()
llm = MockLLMClient() # Default to mock for safety, env var can switch to OllamaClient
agent = CognitiveRetrievalAgent(hrm, reranker) # Will update agent to use LLM next

class QueryRequest(BaseModel):
    query: str

class StepResponse(BaseModel):
    step: int
    action: str
    candidates: int

class QueryResponse(BaseModel):
    query: str
    answer: str
    path: List[StepResponse]

@app.on_event("startup")
def load_resources():
    print("Loading C-RAG Resources...")
    # Here we would load the trained GraphContrastiveModel
    # vs.load_index("path/to/index")

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Endpoints for the Reasoning Agent.
    """
    try:
        # Check if Ollama is available, else use Mock
        # In production this logic is in config
        
        print(f"Received Query: {request.query}")
        result = agent.solve(request.query)
        
        return QueryResponse(
            query=request.query,
            answer=result["answer"],
            path=result["path"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy", "version": "0.1.0"}
