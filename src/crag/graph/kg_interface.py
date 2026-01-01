from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class KnowledgeGraph(ABC):
    """
    Abstract Base Class for Knowledge Graph interfaces.
    """

    @abstractmethod
    def get_neighbors(self, node_id: str) -> List[Dict[str, Any]]:
        """
        Get immediate neighbors of a node.
        Returns a list of dictionaries, each containing:
        - 'id': neighbor node id
        - 'name': neighbor node name/label
        - 'relation': edge label/id
        """
        pass

    @abstractmethod
    def search_node(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for nodes by name/string query.
        Returns list of {'id': ..., 'name': ..., 'description': ...}
        """
        pass

    @abstractmethod
    def get_node_properties(self, node_id: str) -> Dict[str, Any]:
        """
        Get attributes/properties of a specific node.
        """
        pass

class NetworkXKG(KnowledgeGraph):
    """
    In-memory NetworkX implementation for testing and small datasets.
    """
    def __init__(self, nx_graph):
        self.graph = nx_graph

    def get_neighbors(self, node_id: str) -> List[Dict[str, Any]]:
        if node_id not in self.graph:
            return []
        neighbors = []
        for n_id in self.graph.neighbors(node_id):
            edge_data = self.graph.get_edge_data(node_id, n_id)
            # Assuming simple graph or taking first edge
            relation = edge_data.get('relation', 'connected_to') if edge_data else 'connected_to'
            neighbors.append({
                'id': n_id,
                'name': self.graph.nodes[n_id].get('name', str(n_id)),
                'relation': relation
            })
        return neighbors

    def search_node(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        # Simple substring search
        results = []
        for n_id, data in self.graph.nodes(data=True):
            name = data.get('name', str(n_id))
            if query.lower() in name.lower():
                results.append({'id': n_id, 'name': name, 'description': data.get('description', '')})
                if len(results) >= limit:
                    break
        return results

    def get_node_properties(self, node_id: str) -> Dict[str, Any]:
        if node_id not in self.graph:
            return {}
        return self.graph.nodes[node_id]
