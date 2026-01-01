
from typing import Dict, Any
from ..graph.kg_interface import KnowledgeGraph
from ..retrieval.hybrid import HybridRetrievalModule
from ..llm.interface import LLMClient

class StaticGraphBaseline:
    """
    Topic 3B: Subgraph Retrieval Baseline (Non-Agentic).
    Retrieves entry points -> Expands 1-hop -> Reranks -> LLM Generate.
    """
    def __init__(self, hrm: HybridRetrievalModule, llm: LLMClient):
        self.hrm = hrm
        self.llm = llm
        
    def solve(self, query: str) -> Dict[str, Any]:
        # 1. Get Entry Points
        candidates = self.hrm.retrieve_initial_candidates(query, k=5)
        
        # 2. Static Expansion (One Hop)
        expanded = self.hrm.expand_candidates(candidates, limit_per_node=3)
        
        # 3. Contextualize - Extract metadata properly
        context = []
        for c in expanded[:10]:
            meta = c.get('metadata', {})
            name = c.get('name') or meta.get('name') or c.get('text', 'Unknown')
            desc = c.get('description') or meta.get('description', '')
            context.append(f"{name} ({desc})")
        
        context_str = "\n".join(context)
        
        prompt = f"""
        Query: {query}
        Graph Context:
        {context_str}
        
        Answer the query using the graph context.
        ANSWER:
        """
        
        response = self.llm.generate(prompt)
         
        return {
            "answer": response,
            "path": [], 
            "docs_retrieved": len(expanded)
        }
