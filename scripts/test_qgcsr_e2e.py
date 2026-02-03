#!/usr/bin/env python3
"""
End-to-End QGCSR Test Suite
Tests query graph generation and complete QGCSR routing pipeline dynamically.
"""
import sys
import logging
from pathlib import Path
import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crag.graph.partitioning import SemanticPartitioner
from crag.routing.structural import StructuralAligner
from crag.routing.colbert import ColBERTPartitionRouter
from crag.graph.engine import GraphEngine
from crag.model.query_graph import QueryGraphGenerator, GraphSchema
from crag.llm.interface import create_llm_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_movie_knowledge_graph():
    """Create a realistic movie knowledge graph with multiple types."""
    G = nx.DiGraph()
    
    # Directors
    G.add_node(0, type='Person', name='Christopher Nolan', label='Director')
    G.add_node(1, type='Person', name='James Cameron', label='Director')
    
    # Actors
    G.add_node(2, type='Person', name='Leonardo DiCaprio', label='Actor')
    G.add_node(3, type='Person', name='Kate Winslet', label='Actor')
    
    # Movies
    G.add_node(4, type='Movie', name='Inception', label='Movie')
    G.add_node(5, type='Movie', name='Titanic', label='Movie')
    
    # Companies
    G.add_node(6, type='Company', name='Warner Bros', label='Company')
    G.add_node(7, type='Company', name='Paramount', label='Company')
    
    # Genres
    G.add_node(8, type='Genre', name='Sci-Fi', label='Genre')
    G.add_node(9, type='Genre', name='Romance', label='Genre')
    
    # Director relationships
    G.add_edge(0, 4, relation='DIRECTED')
    G.add_edge(1, 5, relation='DIRECTED')
    
    # Actor relationships
    G.add_edge(2, 4, relation='ACTED_IN')
    G.add_edge(2, 5, relation='ACTED_IN')
    G.add_edge(3, 5, relation='ACTED_IN')
    
    # Production relationships
    G.add_edge(6, 4, relation='PRODUCED')
    G.add_edge(7, 5, relation='PRODUCED')
    
    # Genre relationships
    G.add_edge(4, 8, relation='HAS_GENRE')
    G.add_edge(5, 9, relation='HAS_GENRE')
    
    return G

def setup_graph_engine(G):
    """Convert NetworkX graph to GraphEngine format."""
    logger.info("Setting up GraphEngine...")
    
    # Convert to PyG Data
    data = from_networkx(G)
    
    # Add node embeddings (random for testing)
    num_nodes = data.num_nodes
    data.x = torch.randn(num_nodes, 128)
    
    # Extract metadata
    node_text_map = {}
    edge_attr_map = {}
    
    for node, attrs in G.nodes(data=True):
        node_text_map[node] = {
            'id': node,
            'type': attrs.get('type', 'Unknown'),
            'label': attrs.get('label', 'Unknown'),
            'name': attrs.get('name', f'Node_{node}'),
            'text': f"{attrs.get('name', f'Node_{node}')} ({attrs.get('type', 'Unknown')})"
        }
    
    for src, dst, attrs in G.edges(data=True):
        edge_attr_map[(src, dst)] = {
            'relation': attrs.get('relation', 'RELATED_TO'),
            'type': attrs.get('relation', 'RELATED_TO')
        }
    
    # Create GraphEngine
    engine = GraphEngine()
    engine.data = data
    engine.node_text_map = node_text_map
    engine.edge_attr_map = edge_attr_map
    
    logger.info(f"GraphEngine ready: {num_nodes} nodes, {data.edge_index.size(1)} edges")
    return engine

def test_query_graph_generation():
    """Test 1: Query graph generation from natural language."""
    logger.info("\n" + "="*70)
    logger.info("TEST 1: Query Graph Generation")
    logger.info("="*70)
    
    # Define schema
    schema = GraphSchema(
        node_types=['Person', 'Movie', 'Company', 'Genre'],
        edge_types=['DIRECTED', 'ACTED_IN', 'PRODUCED', 'HAS_GENRE']
    )
    
    # Use MockLLM for deterministic testing
    llm = create_llm_client('mock')
    query_gen = QueryGraphGenerator(schema=schema, llm_client=llm)
    
    # Test queries
    test_queries = [
        "Who directed Inception?",
        "Which movies did Leonardo DiCaprio act in?",
        "What company produced Titanic?"
    ]
    
    results = []
    for query in test_queries:
        logger.info(f"\nQuery: '{query}'")
        query_graph = query_gen.parse(query)
        
        # Validate query graph
        assert query_graph is not None, f"Failed to generate query graph for: {query}"
        assert query_graph.num_nodes > 0, f"Query graph has no nodes for: {query}"
        assert hasattr(query_graph, 'node_types'), "Query graph missing node_types"
        assert hasattr(query_graph, 'edge_types'), "Query graph missing edge_types"
        
        logger.info(f"  Nodes: {query_graph.num_nodes}")
        logger.info(f"  Node types: {query_graph.node_types}")
        logger.info(f"  Edge types: {query_graph.edge_types}")
        
        results.append({
            'query': query,
            'graph': query_graph
        })
    
    logger.info("\n✅ Query graph generation successful!")
    return results

def test_semantic_partitioning(engine):
    """Test 2: Semantic partitioning on real graph."""
    logger.info("\n" + "="*70)
    logger.info("TEST 2: Semantic Partitioning")
    logger.info("="*70)
    
    partitioner = SemanticPartitioner(resolution=1.0)
    part_ids = partitioner.partition(engine.data)
    
    assert part_ids is not None, "Partitioning failed"
    assert part_ids.size(0) == engine.data.num_nodes, "Partition size mismatch"
    
    num_partitions = int(part_ids.max().item()) + 1
    logger.info(f"Created {num_partitions} partitions")
    
    # Analyze partition composition
    for pid in range(num_partitions):
        mask = (part_ids == pid)
        nodes_in_partition = mask.nonzero(as_tuple=True)[0].tolist()
        types_in_partition = set([engine.node_text_map[n]['type'] for n in nodes_in_partition])
        logger.info(f"  Partition {pid}: {len(nodes_in_partition)} nodes, types={types_in_partition}")
    
    engine.data.part_id = part_ids
    logger.info("✅ Semantic partitioning successful!")
    return num_partitions

def test_structural_alignment(query_results, engine):
    """Test 3: Structural alignment scoring."""
    logger.info("\n" + "="*70)
    logger.info("TEST 3: Structural Alignment Scoring")
    logger.info("="*70)
    
    aligner = StructuralAligner()
    
    # Build partition metadata
    num_partitions = int(engine.data.part_id.max().item()) + 1
    partition_metadata = []
    
    for pid in range(num_partitions):
        mask = (engine.data.part_id == pid)
        nodes = mask.nonzero(as_tuple=True)[0].tolist()
        
        node_types = set([engine.node_text_map[n]['type'] for n in nodes])
        edge_types = set()
        
        for src, dst in engine.edge_attr_map.keys():
            if src in nodes and dst in nodes:
                edge_types.add(engine.edge_attr_map[(src, dst)]['relation'])
        
        partition_metadata.append({
            'part_id': pid,
            'num_nodes': len(nodes),
            'node_types': node_types,
            'edge_types': edge_types
        })
    
    # Score each query against partitions
    for result in query_results:
        query = result['query']
        query_graph = result['graph']
        
        logger.info(f"\nQuery: '{query}'")
        scores = aligner.batch_score(query_graph, partition_metadata)
        
        assert len(scores) == num_partitions, "Score count mismatch"
        
        # Show top partitions
        sorted_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        logger.info("  Top partitions:")
        for pid, score in sorted_scores[:3]:
            metadata = partition_metadata[pid]
            logger.info(f"    P{pid}: score={score:.3f}, types={metadata['node_types']}")
    
    logger.info("\n✅ Structural alignment successful!")
    return partition_metadata

def test_hybrid_routing(query_results, engine, partition_metadata):
    """Test 4: Hybrid ColBERT routing."""
    logger.info("\n" + "="*70)
    logger.info("TEST 4: Hybrid ColBERT Routing")
    logger.info("="*70)
    
    router = ColBERTPartitionRouter(device='cpu')
    router.build_partition_matrices(engine, num_tokens_per_partition=8)
    
    assert router.partition_embs is not None, "Router not initialized"
    logger.info(f"Router initialized with {router.partition_embs.size(0)} partitions")
    
    # Test different weight configurations
    weight_configs = [
        (1.0, 0.0, 0.0, "Vector only"),
        (0.0, 1.0, 0.0, "Structural only"),
        (0.0, 0.0, 1.0, "ColBERT only"),
        (0.33, 0.33, 0.34, "Balanced hybrid"),
    ]
    
    for result in query_results:
        query = result['query']
        query_graph = result['graph']
        
        logger.info(f"\nQuery: '{query}'")
        
        for alpha, beta, gamma, desc in weight_configs:
            partition_ids, scores = router.route(
                query,
                query_graph=query_graph,
                k=3,
                weights=(alpha, beta, gamma)
            )
            
            assert len(partition_ids) <= 3, "Too many partitions returned"
            assert len(partition_ids) == len(scores), "Score count mismatch"
            
            logger.info(f"  {desc}: P{partition_ids[:3]} (scores={[f'{s:.3f}' for s in scores[:3]]})")
    
    logger.info("\n✅ Hybrid routing successful!")

def test_end_to_end():
    """Complete end-to-end test."""
    logger.info("\n" + "="*70)
    logger.info("TEST 5: End-to-End Pipeline")
    logger.info("="*70)
    
    # Create knowledge graph
    G = create_movie_knowledge_graph()
    engine = setup_graph_engine(G)
    
    # Define schema
    schema = GraphSchema(
        node_types=['Person', 'Movie', 'Company', 'Genre'],
        edge_types=['DIRECTED', 'ACTED_IN', 'PRODUCED', 'HAS_GENRE']
    )
    
    # Create components
    llm = create_llm_client('mock')
    query_gen = QueryGraphGenerator(schema=schema, llm_client=llm)
    partitioner = SemanticPartitioner(resolution=1.0)
    router = ColBERTPartitionRouter(device='cpu')
    
    # Partition graph
    engine.data.part_id = partitioner.partition(engine.data)
    router.build_partition_matrices(engine, num_tokens_per_partition=8)
    
    # Test queries
    queries = [
        "Who directed Inception?",
        "Which movies did Leonardo DiCaprio act in?"
    ]
    
    for query in queries:
        logger.info(f"\n{'='*60}")
        logger.info(f"QUERY: {query}")
        logger.info('='*60)
        
        # 1. Parse query
        query_graph = query_gen.parse(query)
        logger.info(f"✓ Generated query graph: {query_graph.num_nodes} nodes")
        logger.info(f"  Node types: {query_graph.node_types}")
        logger.info(f"  Edge types: {query_graph.edge_types}")
        
        # 2. Route to partitions
        partition_ids, scores = router.route(
            query,
            query_graph=query_graph,
            k=3,
            weights=(0.3, 0.4, 0.3)  # Emphasize structural
        )
        
        logger.info(f"✓ Routed to partitions: {partition_ids}")
        logger.info(f"  Scores: {[f'{s:.3f}' for s in scores]}")
        
        # 3. Examine selected partitions
        for pid in partition_ids[:2]:
            mask = (engine.data.part_id == pid)
            nodes = mask.nonzero(as_tuple=True)[0].tolist()
            node_names = [engine.node_text_map[n]['name'] for n in nodes]
            node_types = set([engine.node_text_map[n]['type'] for n in nodes])
            
            logger.info(f"\n  Partition {pid}:")
            logger.info(f"    Types: {node_types}")
            logger.info(f"    Entities: {', '.join(node_names[:5])}")
    
    logger.info("\n✅ End-to-end pipeline successful!")

if __name__ == "__main__":
    logger.info("="*70)
    logger.info("QGCSR END-TO-END TEST SUITE")
    logger.info("="*70)
    
    try:
        # Create shared resources
        G = create_movie_knowledge_graph()
        engine = setup_graph_engine(G)
        
        # Run tests
        query_results = test_query_graph_generation()
        num_partitions = test_semantic_partitioning(engine)
        partition_metadata = test_structural_alignment(query_results, engine)
        test_hybrid_routing(query_results, engine, partition_metadata)
        test_end_to_end()
        
        logger.info("\n" + "="*70)
        logger.info("✅ ALL TESTS PASSED!")
        logger.info("="*70)
        
    except Exception as e:
        logger.error(f"\n❌ TEST FAILED: {e}", exc_info=True)
        sys.exit(1)
