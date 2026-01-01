
from typing import Dict, Any, List
from ..retrieval.vector_store import VectorStore
from ..llm.interface import LLMClient

class VectorBaseline:
    """
    Topic 3A: Vector-only RAG Baseline.
    Retrieves top-k chunks -> LLM Generate.
    """
    def __init__(self, vector_store: VectorStore, llm: LLMClient, k: int = 5):
        self.vector_store = vector_store
        self.llm = llm
        self.k = k
        
    def solve(self, query: str) -> Dict[str, Any]:
        # 1. Retrieve
        docs = self.vector_store.search(query, k=self.k)
        
        # 2. Augmented Generation
        context = "\n".join([f"- {d.get('text', '')}" for d in docs])
        
        prompt = f"""
        Query: {query}
        Context:
        {context}
        
        Answer the query based on the context. If unknown, say "I don't know".
        ANSWER:
        """
        
        response = self.llm.generate(prompt)
        
        return {
            "answer": response,
            "path": [], # No reasoning path in vector baseline
            "docs_retrieved": len(docs)
        }
