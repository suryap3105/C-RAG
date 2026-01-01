
import logging
import sys
import os

# Create logger
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DebugAgent")

# Mock classes imports (simulating run_exp.py environment)
sys.path.append("src")

from crag.agent.cra import CognitiveRetrievalAgent
from crag.retrieval.hybrid import HybridRetrievalModule
from crag.graph.wikidata import WikidataKG
from crag.retrieval.vector_store import VectorStore
from crag.llm.interface import MockLLMClient
from crag.model.colbert import ColBERTReranker

class DebugVectorStore(VectorStore):
    def search(self, query, k=10):
        # Return empty to force KG fallback
        return []
    
    def add_texts(self, texts, metadatas=None):
        pass
        
    def save(self, path):
        pass
        
    def load(self, path):
        pass

def debug_run():
    logger.info("Initializing Debug Run...")
    
    # 1. Setup
    kg = WikidataKG()
    vs = DebugVectorStore()
    hrm = HybridRetrievalModule(kg, vs)
    reranker = ColBERTReranker() # CPU based in default env
    llm = MockLLMClient()
    
    agent = CognitiveRetrievalAgent(hrm, reranker, llm_client=llm)
    agent.max_steps = 3
    agent.max_expansions = 5
    
    query = "who directed the movie Inception?"
    logger.info(f"Query: {query}")
    
    # 2. Execution
    state = agent.solve(query)
    
    # 3. Report
    logger.info(f"Final Answer: {state.final_answer}")
    logger.info(f"Termination Reason: {state.termination_reason}")
    logger.info(f"Trajectory Steps: {len(state.trajectory)}")
    
    for i, step in enumerate(state.trajectory):
        logger.info(f"Step {i}: {step.action} | Thought: {step.thought[:50]}...")
        
    # Check context in final state
    context_names = [c.get('name') for c in state.knowledge_graph_context]
    logger.info(f"Final Context: {context_names}")

if __name__ == "__main__":
    debug_run()
