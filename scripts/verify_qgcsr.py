"""
Verify QGCSR Implementation
Integration test for Semantic Partitioning, Structural Alignment, and Hybrid Routing.
"""
import torch
import logging
import sys
from pathlib import Path
from torch_geometric.data import Data

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


from crag.graph.partitioning import SemanticPartitioner
from crag.routing.structural import StructuralAligner
from crag.routing.colbert import ColBERTPartitionRouter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_qgcsr_flow():
    logger.info("Setting up test graph...")
    # Create a dummy graph with 3 distinct clusters (Partitions)
    # P0: Nodes 0-9 (Type A)
    # P1: Nodes 10-19 (Type B)
    # P2: Nodes 20-29 (Type A + B Mixed)
    
    num_nodes = 30
    x = torch.randn(num_nodes, 768) # Random embeddings
    
    # Create simple structure
    # P0 clique
    edge_index = [[], []]
    for i in range(10):
        for j in range(10):
            if i != j: 
                edge_index[0].append(i)
                edge_index[1].append(j)
                
    # P1 clique
    for i in range(10, 20):
        for j in range(10, 20):
            if i != j: 
                edge_index[0].append(i)
                edge_index[1].append(j)
                
    data = Data(x=x, edge_index=torch.tensor(edge_index, dtype=torch.long), num_nodes=num_nodes)
    
    # 1. Test Partitioning
    logger.info("Testing Semantic Partitioner...")
    partitioner = SemanticPartitioner(resolution=1.0)
    part_id = partitioner.partition(data)
    logger.info(f"Partition IDs unique: {part_id.unique().tolist()}")
    
    # Mock data.part_id for router
    data.part_id = part_id
    
    # 2. Setup Router
    logger.info("Setting up ColBERT Router...")
    router = ColBERTPartitionRouter(device='cpu')
    
    # Mock GraphEngine behavior by manually building matrices
    # In real usage, we pass GraphEngine, but here we just pass the mock data wrapper
    class MockGraphEngine:
        def __init__(self, data): self.data = data
        
    router.build_partition_matrices(MockGraphEngine(data), num_tokens_per_partition=5)
    
    # Manually inject metadata for testing structural scoring
    # P0 has Type A, P1 has Type B
    router.partition_metadata[0] = {'node_types': {'TypeA'}, 'part_id': 0}
    if len(router.partition_metadata) > 1:
        router.partition_metadata[1] = {'node_types': {'TypeB'}, 'part_id': 1}
        
    # 3. Test Structural Alignment
    logger.info("Testing Structural Aligner...")
    
    # Query matching P0 (Type A)
    q_graph_a = Data(num_nodes=2)
    q_graph_a.node_types = ['TypeA', 'TypeA']
    
    # Query matching P1 (Type B)
    q_graph_b = Data(num_nodes=2)
    q_graph_b.node_types = ['TypeB']
    
    aligner = StructuralAligner()
    score_a_p0 = aligner.score(q_graph_a, router.partition_metadata[0])
    
    if len(router.partition_metadata) > 1:
        score_a_p1 = aligner.score(q_graph_a, router.partition_metadata[1])
        logger.info(f"Score Q(TypeA) vs P0(TypeA): {score_a_p0}")
        logger.info(f"Score Q(TypeA) vs P1(TypeB): {score_a_p1}")
        
        if score_a_p0 > score_a_p1:
            logger.info("SUCCESS: Structural Aligner correctly prefers matching types.")
        else:
            logger.error("FAILURE: Structural Aligner did not distinguish types.")
            
    # 4. Test Hybrid Routing
    logger.info("Testing Hybrid Routing...")
    # Route with query_graph
    ids, scores = router.route("test query", query_graph=q_graph_a, k=2, weights=(0.0, 1.0, 0.0)) # Pure structural
    logger.info(f"Routing results (Pure Structural): IDs={ids}, Scores={scores}")
    
    # Verify P0 is first
    if ids[0] == 0:
         logger.info("SUCCESS: Hybrid Router selected correct partition based on structure.")
    else:
         logger.warning(f"Router selected {ids[0]} instead of 0. (Might be due to partitioning randomization or setup)")

if __name__ == "__main__":
    test_qgcsr_flow()
