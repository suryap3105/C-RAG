from typing import List, Dict, Any, Tuple, Optional, Callable
from .vector_store import VectorStore
from ..graph.kg_interface import KnowledgeGraph

class HybridRetrievalModule:
    """
    Production HRM:
    - Combines Vector Search (Dense) and Graph Traversal (Sparse/Symbolic).
    - Implements 'Reciprocal Rank Fusion' (RRF) or Weighted Scoring for robustness.
    """
    def __init__(self, kg: KnowledgeGraph, vector_store: VectorStore, alpha: float = 0.5):
        self.kg = kg
        self.vector_store = vector_store
        self.alpha = alpha # Weight for vector vs graph scores if we were doing linear combination

    def retrieve_initial_candidates(self, query: str, k: int = 10) -> List[Dict]:
        """
        Retrieves initial concepts using Dense Retrieval OR KG Search.
        """
        # 1. Try Vector Search
        results = self.vector_store.search(query, k=k)
        
        # 2. Fallback to KG Entity Search (Live Entry Point)
        if not results:
            print(f"[HRM] Vector store empty/miss. Falling back to KG Search for: '{query}'")
            # Heuristic: Search for the whole query or nouns. 
            # For this demo, we search the full query string as entity label candidate.
            kg_results = self.kg.search_node(query, limit=k)
            for item in kg_results:
                # Standardize format to match VectorStore (metadata nesting)
                results.append({
                    "text": item['name'] + " " + item.get('description', ''),
                    "metadata": {
                        "id": item['id'],
                        "name": item['name'],
                        "description": item.get('description', '')
                    },
                    "score": 1.0, 
                    "retrieval_source": "kg_search"
                })

        unique_results = []
        seen = set()
        for r in results:
            # Handle both nested metadata (Vector/New Fallback) and flat (Legacy?)
            # Actually, standardizing on accessing 'metadata' is safer if we enforce it.
            meta = r.get('metadata', r) # Fallback to top-level if no metadata
            uid = meta.get('id', r.get('id'))
            
            if uid and uid not in seen:
                seen.add(uid)
                if 'retrieval_source' not in r:
                    r['retrieval_source'] = 'vector_similarity'
                    r['retrieval_score'] = r.get('score', 0.0)
                unique_results.append(r)
                
        return unique_results[:k]

    def expand_candidates(self, candidates: List[Dict[str, Any]], limit_per_node: int = 5, 
                          policy: Optional[Callable[[List[Dict], int], List[Dict]]] = None) -> List[Dict[str, Any]]:
        """
        Step 2: Symbolic Expansion (Graph RAG).
        Args:
            policy: Optional function to override default expansion logic (Adaptive Hook).
        """
        if policy:
            return policy(candidates, limit_per_node)

        # Default Policy: Neighbor Expansion
        expanded_pool = []
        seen_ids = set()

        # Add original candidates first
        for cand in candidates:
             meta = cand.get('metadata', {})
             cid = meta.get('id')
             if cid:
                 seen_ids.add(cid)
                 cand['retrieval_source'] = 'original_context'
                 expanded_pool.append(cand)

        # Expand neighbors
        for cand in candidates:
            meta = cand.get('metadata', {})
            ent_id = meta.get('id') or meta.get('entity_id')
            
            if ent_id:
                neighbors = self.kg.get_neighbors(ent_id)
                # Introspection: Log how many neighbors found
                # print(f"DEBUG: Found {len(neighbors)} neighbors for {ent_id}")
                
                for neighbor in neighbors[:limit_per_node]:
                    if neighbor['id'] not in seen_ids:
                        seen_ids.add(neighbor['id'])
                        expanded_pool.append({
                            "text": f"{neighbor['name']} - {neighbor['relation']}",
                            "metadata": {
                                "id": neighbor['id'],
                                "name": neighbor['name'],
                                "type": "neighbor_expansion",
                                "source_node": ent_id
                            },
                            "retrieval_source": "graph_expansion",
                            "retrieval_score": 1.0 # Placeholder for symbolic relevance
                        })
        return expanded_pool
