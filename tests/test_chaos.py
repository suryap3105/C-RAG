"""
Chaos Engineering Tests
Inject failures to test system resilience
"""
import pytest
import time
import random
import threading
from unittest.mock import patch, Mock

from crag import GraphEngine, FaissVectorStore, NeuroHybridRetrievalModule
from crag.utils.resilience import CircuitBreaker


class TestChaosEngineering:
    """Chaos engineering tests."""
    
    def test_random_llm_failures(self):
        """Test system resilience to random LLM failures."""
        from crag.llm.interface import MockLLMClient
        
        class FlakyLLMClient(MockLLMClient):
            def __init__(self, failure_rate=0.3):
                super().__init__()
                self.failure_rate = failure_rate
                
            def generate(self, prompt, **kwargs):
                if random.random() < self.failure_rate:
                    raise RuntimeError("LLM timeout")
                return super().generate(prompt, **kwargs)
                
        from crag.model.query_graph import QueryGraphGenerator
        
        flaky_llm = FlakyLLMClient(failure_rate=0.5)
        gen = QueryGraphGenerator(flaky_llm)
        
        # Try multiple queries
        successes = 0
        failures = 0
        
        for _ in range(20):
            try:
                graph = gen.parse("test query")
                if graph.num_nodes > 0:
                    successes += 1
            except Exception:
                failures += 1
                
        # Should handle some failures gracefully
        assert successes > 0
        assert failures > 0
        
    def test_network_latency_injection(self):
        """Test system under high latency conditions."""
        original_search = FaissVectorStore.search
        
        def slow_search(self, query, k=10):
            time.sleep(random.uniform(0.5, 2.0))  # Inject latency
            return original_search(self, query, k)
            
        with patch.object(FaissVectorStore, 'search', slow_search):
            vs = FaissVectorStore()
            vs.add_documents([
                {'id': i, 'text': f'doc {i}', 'metadata': {'id': i}}
                for i in range(10)
            ])
            
            start = time.time()
            results = vs.search("test", k=5)
            elapsed = time.time() - start
            
            # Should complete despite high latency
            assert elapsed > 0.5
            assert len(results) > 0
            
    def test_memory_pressure(self):
        """Test system under memory pressure."""
        import gc
        
        # Create large graph
        ge = GraphEngine()
        ge.data.num_nodes = 50000
        ge.data.edge_index = torch.randint(0, 50000, (2, 250000))
        ge.data.x = torch.randn(50000, 768)
        
        # Force garbage collection
        gc.collect()
        
        # Should still extract subgraph
        subgraph = ge.extract_subgraph([0], num_hops=2, max_nodes=100)
        assert subgraph.num_nodes > 0
        
        # Cleanup
        del ge
        gc.collect()
        
    def test_concurrent_load_spike(self):
        """Test system under sudden concurrent load."""
        pipeline = self._create_mini_pipeline()
        
        results = []
        errors = []
        
        def query_task():
            try:
                r = pipeline.retrieve("test query", k=5)
                results.append(len(r))
            except Exception as e:
                errors.append(str(e))
                
        # Spike: 50 concurrent requests
        threads = [threading.Thread(target=query_task) for _ in range(50)]
        
        start = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.time() - start
        
        # Most requests should succeed
        success_rate = len(results) / (len(results) + len(errors))
        assert success_rate > 0.8  # 80% success rate
        
        print(f"\nConcurrent spike: {len(results)} success, {len(errors)} errors in {elapsed:.2f}s")
        
    def test_circuit_breaker_activation(self):
        """Test circuit breaker opens after failures."""
        from crag.utils.resilience import CircuitBreaker, CircuitBreakerError, CircuitBreakerConfig
        
        cb = CircuitBreaker(CircuitBreakerConfig(
            failure_threshold=3,
            timeout=1.0
        ))
        
        def failing_func():
            raise RuntimeError("Always fails")
            
        # Trigger enough failures to open circuit
        for _ in range(5):
            try:
                cb.call(failing_func)
            except (RuntimeError, CircuitBreakerError):
                pass
                
        # Circuit should be open now
        with pytest.raises(CircuitBreakerError):
            cb.call(failing_func)
            
    def test_gradual_degradation(self):
        """Test system degrades gracefully under increasing load."""
        pipeline = self._create_mini_pipeline()
        
        latencies = []
        
        for load in [1, 5, 10, 20]:
            results = []
            
            def query_with_load():
                start = time.time()
                pipeline.retrieve("test", k=5)
                results.append(time.time() - start)
                
            threads = [threading.Thread(target=query_with_load) for _ in range(load)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
                
            avg_latency = sum(results) / len(results)
            latencies.append(avg_latency)
            
        # Latency should increase, but not catastrophically
        assert latencies[-1] < latencies[0] * 10  # Less than 10x slower
        
    def test_partial_component_failure(self):
        """Test system continues with partial component failures."""
        pipeline = self._create_mini_pipeline()
        
        # Disable graph search
        pipeline.graph_engine = None
        
        # Should still work with vector search only
        results = pipeline.retrieve("test query", k=5)
        assert len(results) > 0
        
    def _create_mini_pipeline(self):
        """Create minimal pipeline for testing."""
        from crag.llm.interface import create_llm_client
        from crag.model.query_graph import QueryGraphGenerator
        from crag.model.gnn import NeuralSubgraphMatcher
        from crag.routing.colbert import ColBERTPartitionRouter
        
        ge = GraphEngine()
        ge.data.num_nodes = 20
        ge.data.edge_index = torch.randint(0, 20, (2, 40))
        ge.data.x = torch.randn(20, 768)
        
        vs = FaissVectorStore()
        vs.add_documents([
            {'id': i, 'text': f'doc {i}', 'metadata': {'id': i}}
            for i in range(20)
        ])
        
        llm = create_llm_client(provider='mock')
        query_gen = QueryGraphGenerator(llm)
        matcher = NeuralSubgraphMatcher()
        router = ColBERTPartitionRouter()
        
        return NeuroHybridRetrievalModule(
            vector_store=vs,
            graph_engine=ge,
            query_gen=query_gen,
            neural_matcher=matcher,
            colbert_router=router,
            use_adaptive_gating=False
        )


import torch
