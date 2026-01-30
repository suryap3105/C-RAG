"""
Performance Benchmarking Suite
"""
import pytest
import time
import torch
import numpy as np
from pathlib import Path
import json

from crag import GraphEngine, FaissVectorStore, NeuroHybridRetrievalModule
from crag.model.gnn import NeuralSubgraphMatcher


class TestPerformanceBenchmarks:
    """Performance benchmarks and profiling."""
    
    @pytest.fixture
    def benchmark_graph(self):
        """Create benchmark graph."""
        num_nodes = 1000
        num_edges = 5000
        
        ge = GraphEngine()
        ge.data.num_nodes = num_nodes
        ge.data.edge_index = torch.randint(0, num_nodes, (2, num_edges))
        ge.data.x = torch.randn(num_nodes, 768)
        ge.data.part_id = torch.randint(0, 10, (num_nodes,))
        
        return ge
        
    def test_gnn_forward_speed(self, benchmark):
        """Benchmark GNN forward pass speed."""
        model = NeuralSubgraphMatcher()
        model.eval()
        
        graph = torch_geometric.data.Data(
            x=torch.randn(100, 768),
            edge_index=torch.randint(0, 100, (2, 400)),
            num_nodes=100
        )
        
        def forward():
            with torch.no_grad():
                model(graph)
                
        result =benchmark(forward)
        # Should be fast
        assert result < 0.1  # Less than 100ms
        
    def test_vector_search_throughput(self):
        """Test vector search throughput."""
        vs = FaissVectorStore()
        
        # Add docs
        docs = [
            {'id': i, 'text': f'Document {i}', 'metadata': {'id': i}}
            for i in range(10000)
        ]
        vs.add_documents(docs, batch_size=100)
        
        # Benchmark search
        queries = [f"query {i}" for i in range(100)]
        
        start = time.time()
        for q in queries:
            vs.search(q, k=10)
        elapsed = time.time() - start
        
        qps = len(queries) / elapsed
        print(f"\nVector Search: {qps:.2f} QPS")
        assert qps > 10  # At least 10 QPS
        
    def test_subgraph_extraction_speed(self, benchmark_graph):
        """Test subgraph extraction performance."""
        times = []
        
        for _ in range(100):
            center = np.random.randint(0, 1000)
            
            start = time.time()
            benchmark_graph.extract_subgraph([center], num_hops=2, max_nodes=50)
            elapsed = time.time() - start
            
            times.append(elapsed)
            
        avg_time = np.mean(times) * 1000  # Convert to ms
        p95_time = np.percentile(times, 95) * 1000
        
        print(f"\nSubgraph extraction: avg={avg_time:.2f}ms, p95={p95_time:.2f}ms")
        assert avg_time < 50  # Average under 50ms
        assert p95_time < 100  # P95 under 100ms
        
    def test_memory_usage(self):
        """Test memory efficiency."""
        import tracemalloc
        
        tracemalloc.start()
        
        # Create large graph
        ge = GraphEngine()
        ge.data.num_nodes = 10000
        ge.data.edge_index = torch.randint(0, 10000, (2, 50000))
        ge.data.x = torch.randn(10000, 768)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        peak_mb = peak / 1024 / 1024
        print(f"\nPeak memory: {peak_mb:.2f} MB")
        assert peak_mb < 500  # Should be under 500 MB
        
    def test_batch_processing_speedup(self):
        """Test that batching improves throughput."""
        model = NeuralSubgraphMatcher()
        model.eval()
        
        graphs = [
            torch_geometric.data.Data(
                x=torch.randn(20, 768),
                edge_index=torch.randint(0, 20, (2, 40)),
                num_nodes=20
            )
            for _ in range(100)
        ]
        
        # Sequential
        start = time.time()
        with torch.no_grad():
            for g in graphs:
                model(g)
        seq_time = time.time() - start
        
        # Batched
        from torch_geometric.data import Batch
        batch = Batch.from_data_list(graphs)
        
        start = time.time()
        with torch.no_grad():
            model(batch)
        batch_time = time.time() - start
        
        speedup = seq_time / batch_time
        print(f"\nBatch speedup: {speedup:.2f}x")
        assert speedup > 2  # Batching should be at least 2x faster
        
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_acceleration(self):
        """Test GPU speedup."""
        model = NeuralSubgraphMatcher()
        graph = torch_geometric.data.Data(
            x=torch.randn(200, 768),
            edge_index=torch.randint(0, 200, (2, 800)),
            num_nodes=200
        )
        
        # CPU
        model.eval()
        start = time.time()
        with torch.no_grad():
            for _ in range(10):
                model(graph)
        cpu_time = time.time() - start
        
        # GPU
        model = model.cuda()
        graph = graph.cuda()
        
        start = time.time()
        with torch.no_grad():
            for _ in range(10):
                model(graph)
        gpu_time = time.time() - start
        
        speedup = cpu_time / gpu_time
        print(f"\nGPU speedup: {speedup:.2f}x")
        assert speedup > 1.5  # GPU should be faster


def generate_benchmark_report(results_dir="benchmark_results"):
    """Generate comprehensive benchmark report."""
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)
    
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "system": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available()
        },
        "benchmarks": {}
    }
    
    # Run benchmarks and collect results
    # (Would integrate with pytest-benchmark)
    
    with open(results_dir / "benchmark_report.json", 'w') as f:
        json.dump(report, f, indent=2)
        
    print(f"Benchmark report saved to {results_dir}")
