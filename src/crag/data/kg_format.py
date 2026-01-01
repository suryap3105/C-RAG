
from dataclasses import dataclass
from typing import List, Optional, Dict

@dataclass
class KGNode:
    """
    Standard Node format for C-RAG.
    """
    id: str
    name: str # Label
    description: Optional[str] = None
    properties: Optional[Dict] = None

@dataclass
class KGEdge:
    """
    Standard Edge format.
    """
    source_id: str
    target_id: str
    relation: str
    weight: float = 1.0

@dataclass
class KnowledgeGraphData:
    """
    Container for a full KG dataset.
    """
    nodes: Dict[str, KGNode]
    edges: List[KGEdge]
    
    def add_node(self, node: KGNode):
        self.nodes[node.id] = node
        
    def add_edge(self, edge: KGEdge):
        self.edges.append(edge)
