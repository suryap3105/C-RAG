"""
Integration Tests for Full C-RAG Pipeline
"""
import pytest
import torch
import json
import tempfile
from pathlib import Path

from crag import (
    GraphEngine,
    FaissVectorStore,
    NeuroHybridRetrievalModule,
    NeuralSubgraphMatcher,
    ColBERTPartitionRouter,
    QueryGraphGenerator,
    create_llm_client,
    GraphPartitioner
)


class TestEndToEndPipeline:
    """Integration tests for complete pipeline."""
    
    @pytest.fixture
    def mini_graph_dataset(self, tmp_path):
        """Create minimal graph dataset for testing."""
        # Nodes
        nodes = [
            {'id': i, 'name': f'Entity_{i}', 'text': f'This is entity {i}'}
            for i in range(20)
        ]
        
        nodes_path = tmp_path / "nodes.jsonl"
        with open(nodes_path, 'w') as f:
            for node in nodes:
                f.write(json.dumps(node) + '\n')
                
        # Edges
        edges = []
        for i in range(19):
            edges.append({'src': i, 'dst': i + 1})
            
        edges_path = tmp_path / "edges.jsonl"
        with open(edges_path, 'w') as f:
            for edge in edges:
                f.write(json.dumps(edge) + '\n')
                
        return str(edges_path), str(nodes_path)
        
    @pytest.fixture
    def built_pipeline(self, mini_graph_dataset, tmp_path):
        """Build complete pipeline."""
        edges_path, nodes_path = mini_graph_dataset
        
        # Graph
        ge = GraphEngine()
        ge.load_graph(edges_path, nodes_path)
        ge.compute_embeddings(batch_size=10)
        
        # Partition
        partitioner = GraphPartitioner(method='spectral', n_partitions=3)
        partitioner.partition(ge)
        
        # Vector store
        vs = FaissVectorStore()
        docs = [
            {'id': nid, 'text': info['text'], 'metadata': {'id': nid}}
            for nid, info in ge.node_text_map.items()
        ]
        vs.add_documents(docs, batch_size=10)
        
        # Components
        llm = create_llm_client(provider='mock')
        query_gen = QueryGraphGenerator(llm)
        matcher = NeuralSubgraphMatcher()
        router = ColBERTPartitionRouter()
        router.build_partition_matrices(ge, num_tokens_per_partition=8)
        
        # Pipeline
        pipeline = NeuroHybridRetrievalModule(
            vector_store=vs,
            graph_engine=ge,
            query_gen=query_gen,
            neural_matcher=matcher,
            colbert_router=router,
            use_adaptive_gating=False
        )
        
        return pipeline
        
    def test_pipeline_retrieval(self, built_pipeline):
        """Test end-to-end retrieval."""
        query = "Find Entity 5"
        results = built_pipeline.retrieve(query, k=5)
        
        assert len(results) > 0
        assert all('text' in r for r in results)
        assert all('score' in r for r in results)
        assert all('id' in r.get('metadata', {}) for r in results)
        
    def test_pipeline_with_reranking(self, built_pipeline):
        """Test retrieval with reranking."""
        query = "Find Entity 10"
        results = built_pipeline.retrieve(query, k=5, use_reranking=True)
        
        assert len(results) <= 5
        assert all('rerank_score' in r or 'score' in r for r in results)
        
    def test_pipeline_empty_query(self, built_pipeline):
        """Test with empty query."""
        results = built_pipeline.retrieve("", k=5)
        # Should handle gracefully
        assert isinstance(results, list)
        
    def test_pipeline_large_k(self, built_pipeline):
        """Test with k larger than dataset."""
        results = built_pipeline.retrieve("test", k=1000)
        assert len(results) <= 20  # Dataset size
        
    def test_parallel_queries(self, built_pipeline):
        """Test concurrent query processing."""
        import threading
        
        queries = [f"Query {i}" for i in range(10)]
        results = []
        
        def run_query(q):
            r = built_pipeline.retrieve(q, k=3)
            results.append(len(r))
            
        threads = [threading.Thread(target=run_query, args=(q,)) for q in queries]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
            
        assert len(results) == 10
        assert all(r >= 0 for r in results)
        
    def test_save_load_pipeline(self, built_pipeline, tmp_path):
        """Test saving and loading pipeline components."""
        # Save components
        ge_path = tmp_path / "graph.pt"
        vs_path = tmp_path / "vector_store"
        router_path = tmp_path / "router.pt"
        matcher_path = tmp_path / "matcher.pt"
        
        built_pipeline.graph_engine.save(str(ge_path))
        built_pipeline.vector_store.save(str(vs_path))
        built_pipeline.colbert_router.save(str(router_path))
        torch.save(built_pipeline.neural_matcher.state_dict(), matcher_path)
        
        # Load and verify
        ge2 = GraphEngine()
        ge2.load(str(ge_path))
        assert ge2.data.num_nodes == 20
        
        vs2 = FaissVectorStore()
        vs2.load(str(vs_path))
        assert len(vs2.documents) == 20
        
    def test_pipeline_robustness(self, built_pipeline):
        """Test pipeline with various query types."""
        test_queries = [
            "simple query",
            "What is Entity 7?",
            "Find connections between items",
            "Who directed what?",
            "very long query " * 50,
            "query with special chars: @#$%",
            "UPPERCASE QUERY",
            "123456789",
            ""
        ]
        
        for query in test_queries:
            try:
                results = built_pipeline.retrieve(query, k=3)
                assert isinstance(results, list)
            except Exception as e:
                pytest.fail(f"Query '{query[:20]}...' failed: {e}")


class TestComponentIntegration:
    """Test integration between specific components."""
    
    def test_colbert_router_integration(self, tmp_path):
        """Test ColBERT router with actual graph."""
        # Create graph
        ge = GraphEngine()
        ge.data.num_nodes = 50
        ge.data.edge_index = torch.randint(0, 50, (2, 200))
        ge.data.x = torch.randn(50, 768)
        ge.data.part_id = torch.randint(0, 5, (50,))
        
        # Build router
        router = ColBERTPartitionRouter()
        router.build_partition_matrices(ge, num_tokens_per_partition=16)
        
        # Test routing
        partition_ids, scores = router.route("test query", k=3)
        
        assert len(partition_ids) == 3
        assert len(scores) == 3
        assert all(0 <= pid < 5 for pid in partition_ids)
        
    def test_query_graph_generator_integration(self):
        """Test query graph generator with LLM."""
        llm = create_llm_client(provider='mock')
        gen = QueryGraphGenerator(llm)
        
        queries = [
            "Who directed Inception?",
            "What movies did Nolan make?",
            "Find all actors in the movie"
        ]
        
        for query in queries:
            graph = gen.parse(query)
            assert graph.num_nodes > 0
            assert hasattr(graph, 'x')
            
    def test_partitioning_integration(self):
        """Test graph partitioning integration."""
        ge = GraphEngine()
        ge.data.num_nodes = 100
        ge.data.edge_index = torch.randint(0, 100, (2, 500))
        
        for method in ['spectral']:  # METIS/Leiden may not be available
            partitioner = GraphPartitioner(method=method, n_partitions=5)
            part_ids = partitioner.partition(ge)
            
            assert len(part_ids) == 100
            assert part_ids.max() < 5
            
            quality = partitioner.compute_partition_quality(ge)
            assert 'cut_ratio' in quality
            assert 'balance_score' in quality
