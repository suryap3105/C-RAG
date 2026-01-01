import json
import networkx as nx
from typing import List, Dict, Any, Optional
from .kg_interface import KnowledgeGraph

class LocalKnowledgeGraph(KnowledgeGraph):
    """
    Local Graph interface for generated KGs.
    """
    def __init__(self, path: str):
        self.path = path
        self.graph = None
        self._load()

    def _load(self):
        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.graph = nx.adjacency_graph(data)
            print(f"[LocalKG] Loaded graph from {self.path}")
            print(f"          Nodes: {self.graph.number_of_nodes()}")
            print(f"          Edges: {self.graph.number_of_edges()}")
        except Exception as e:
            print(f"[ERROR] Failed to load local KG: {e}")
            self.graph = nx.Graph()

    def get_neighbors(self, node_id: str) -> List[Dict[str, Any]]:
        """
        Get neighbors from local NetworkX graph.
        Returns list of dicts: [{'id': ..., 'name': ..., 'description': relation}]
        """
        if node_id not in self.graph:
            return []
        
        neighbors = []
        for nbr in self.graph.neighbors(node_id):
            edge_data = self.graph.get_edge_data(node_id, nbr)
            weight = edge_data.get('weight', 1)
            
            neighbors.append({
                "id": nbr,
                "name": nbr,
                "description": f"co-occurs ({weight})",
                "type": "entity"
            })
            
        return neighbors

    def search_node(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Simple exact/partial match search.
        In real apps, use vector index for this.
        """
        results = []
        query_lower = query.lower()
        
        # Limit search space for speed
        count = 0
        for node in self.graph.nodes():
            if query_lower in node.lower():
                results.append({
                    "id": node,
                    "name": node,
                    "description": "Local Entity"
                })
                count += 1
                if count >= limit: break
        return results

    def get_node_properties(self, node_id: str) -> Dict[str, Any]:
        """
        Get properties for a local node.
        """
        if node_id not in self.graph:
            return {}
        return self.graph.nodes[node_id]
