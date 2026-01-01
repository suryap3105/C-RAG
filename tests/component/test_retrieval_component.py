import pytest
import networkx as nx
from crag.graph.kg_interface import NetworkXKG
from crag.retrieval.hybrid import HybridRetrievalModule
from crag.retrieval.vector_store import FaissVectorStore

@pytest.fixture
def memory_components():
    # Setup in-memory graph
    g = nx.DiGraph()
    g.add_node("Q1", name="Entity1")
    g.add_node("Q2", name="Entity2")
    g.add_edge("Q1", "Q2", relation="relates")
    kg = NetworkXKG(g)
    
    # Setup real (but local) FAISS store
    vs = FaissVectorStore()
    vs.add_texts(["Entity1 Description"], metadatas=[{"id": "Q1", "entity_id": "Q1"}])
    
    return kg, vs

def test_retrieval_flow(memory_components):
    """
    Component Test: Verifies HRM works with actual (in-memory) KG and VectorStore implementations.
    """
    kg, vs = memory_components
    hrm = HybridRetrievalModule(kg, vs)
    
    # 1. Search (hits VectorStore)
    candidates = hrm.retrieve_initial_candidates("Entity1")
    assert len(candidates) > 0
    assert candidates[0]['metadata']['id'] == "Q1"
    
    # 2. Expand (hits KG)
    expanded = hrm.expand_candidates(candidates)
    expanded_ids = [c['metadata']['id'] for c in expanded]
    
    # Should find Q2 via relation
    assert "Q2" in expanded_ids
