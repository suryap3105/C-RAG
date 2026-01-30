import torch
from crag.llm.interface import MockLLMClient
from crag.model.query_graph import QueryGraphGenerator

def test_query_generation():
    print("Testing Query Graph Generator...")
    llm = MockLLMClient()
    gen = QueryGraphGenerator(llm)
    
    query = "Who directed Inception?"
    data = gen.parse(query)
    
    print(f"Parsed Graph: {data}")
    print(f"Nodes: {data.num_nodes}, Edges: {data.edge_index.size(1)}")
    
    # Mock LLM returns 2 nodes, 1 edge
    assert data.num_nodes == 2, f"Expected 2 nodes, got {data.num_nodes}"
    assert data.edge_index.size(1) == 1, f"Expected 1 edge, got {data.edge_index.size(1)}"
    
    print("SUCCESS: Query Graph Generator Passed!")

if __name__ == "__main__":
    test_query_generation()
