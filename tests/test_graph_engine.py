"""
Comprehensive Unit Tests for GraphEngine
"""
import pytest
import torch
import json
import tempfile
from pathlib import Path
from torch_geometric.data import Data

from crag.graph.engine import GraphEngine


class TestGraphEngine:
    """Test suite for GraphEngine with edge cases."""
    
    @pytest.fixture
    def sample_graph_data(self, tmp_path):
        """Create sample graph files."""
        nodes_path = tmp_path / "nodes.jsonl"
        edges_path = tmp_path / "edges.jsonl"
        
        # Create nodes
        nodes = [
            {'id': 0, 'name': 'Alice', 'text': 'Alice is a person'},
            {'id': 1, 'name': 'Bob', 'text': 'Bob is a person'},
            {'id': 2, 'name': 'Movie', 'text': 'A great movie'}
        ]
        
        with open(nodes_path, 'w') as f:
            for node in nodes:
                f.write(json.dumps(node) + '\n')
                
        # Create edges
        edges = [
            {'src': 0, 'dst': 2, 'relation': 'ACTED_IN'},
            {'src': 1, 'dst': 2, 'relation': 'DIRECTED'}
        ]
        
        with open(edges_path, 'w') as f:
            for edge in edges:
                f.write(json.dumps(edge) + '\n')
                
        return str(edges_path), str(nodes_path)
        
    def test_initialization(self):
        """Test GraphEngine initialization."""
        ge = GraphEngine()
        assert ge.data is not None
        assert isinstance(ge.node_text_map, dict)
        assert isinstance(ge.edge_attr_map, dict)
        
    def test_load_graph(self, sample_graph_data):
        """Test graph loading from files."""
        edges_path, nodes_path = sample_graph_data
        ge = GraphEngine()
        ge.load_graph(edges_path, nodes_path)
        
        assert ge.data.num_nodes == 3
        assert ge.data.edge_index.size(1) == 2
        assert len(ge.node_text_map) == 3
        
    def test_load_graph_empty_files(self, tmp_path):
        """Test loading empty graph files."""
        nodes_path = tmp_path / "empty_nodes.jsonl"
        edges_path = tmp_path / "empty_edges.jsonl"
        
        nodes_path.touch()
        edges_path.touch()
        
        ge = GraphEngine()
        ge.load_graph(str(edges_path), str(nodes_path))
        
        assert ge.data.num_nodes == 0
        
    def test_load_graph_malformed_json(self, tmp_path):
        """Test loading malformed JSON."""
        nodes_path = tmp_path / "bad_nodes.jsonl"
        edges_path = tmp_path / "bad_edges.jsonl"
        
        with open(nodes_path, 'w') as f:
            f.write("{invalid json\n")
            
        with open(edges_path, 'w') as f:
            f.write("{src: 0, dst: 1}\n")
            
        ge = GraphEngine()
        with pytest.raises(json.JSONDecodeError):
            ge.load_graph(str(edges_path), str(nodes_path))
            
    def test_compute_embeddings(self, sample_graph_data):
        """Test embedding computation."""
        edges_path, nodes_path = sample_graph_data
        ge = GraphEngine()
        ge.load_graph(edges_path, nodes_path)
        
        ge.compute_embeddings(batch_size=2)
        
        assert ge.data.x is not None
        assert ge.data.x.shape == (3, 768)
        assert torch.isfinite(ge.data.x).all()
        
    def test_extract_subgraph_basic(self, sample_graph_data):
        """Test basic subgraph extraction."""
        edges_path, nodes_path = sample_graph_data
        ge = GraphEngine()
        ge.load_graph(edges_path, nodes_path)
        ge.data.x = torch.randn(3, 768)
        
        subgraph = ge.extract_subgraph([0], num_hops=1)
        
        assert subgraph.num_nodes > 0
        assert subgraph.x is not None
        assert hasattr(subgraph, 'original_ids')
        
    def test_extract_subgraph_isolated_node(self, sample_graph_data):
        """Test extracting subgraph from isolated node."""
        edges_path, nodes_path = sample_graph_data
        ge = GraphEngine()
        ge.load_graph(edges_path, nodes_path)
        
        # Add isolated node
        ge.data.num_nodes = 4
        ge.data.x = torch.randn(4, 768)
        
        subgraph = ge.extract_subgraph([3], num_hops=2)
        assert subgraph.num_nodes == 1
        
    def test_extract_subgraph_max_nodes_limit(self, sample_graph_data):
        """Test subgraph size limiting."""
        edges_path, nodes_path = sample_graph_data
        ge = GraphEngine()
        ge.load_graph(edges_path, nodes_path)
        ge.data.x = torch.randn(3, 768)
        
        subgraph = ge.extract_subgraph([0, 1], num_hops=5, max_nodes=2)
        assert subgraph.num_nodes <= 2
        
    def test_get_neighbors(self, sample_graph_data):
        """Test neighbor retrieval."""
        edges_path, nodes_path = sample_graph_data
        ge = GraphEngine()
        ge.load_graph(edges_path, nodes_path)
        
        neighbors = ge.get_neighbors(0)
        assert len(neighbors) > 0
        assert 2 in neighbors.tolist()
        
    def test_save_and_load(self, sample_graph_data, tmp_path):
        """Test save/load functionality."""
        edges_path, nodes_path = sample_graph_data
        ge = GraphEngine()
        ge.load_graph(edges_path, nodes_path)
        ge.data.x = torch.randn(3, 768)
        
        save_path = tmp_path / "graph.pt"
        ge.save(str(save_path))
        
        ge2 = GraphEngine()
        ge2.load(str(save_path))
        
        assert ge2.data.num_nodes == 3
        assert torch.equal(ge2.data.x, ge.data.x)
        assert ge2.node_text_map == ge.node_text_map
        
    def test_save_without_data(self, tmp_path):
        """Test saving empty graph."""
        ge = GraphEngine()
        save_path = tmp_path / "empty.pt"
        
        ge.save(str(save_path))
        assert save_path.exists()
        
    def test_concurrent_access(self, sample_graph_data):
        """Test thread-safe operations."""
        import threading
        
        edges_path, nodes_path = sample_graph_data
        ge = GraphEngine()
        ge.load_graph(edges_path, nodes_path)
        ge.data.x = torch.randn(3, 768)
        
        results = []
        
        def extract():
            sub = ge.extract_subgraph([0], num_hops=1)
            results.append(sub.num_nodes)
            
        threads = [threading.Thread(target=extract) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
            
        assert len(results) == 10
        assert all(r > 0 for r in results)
        
    def test_large_graph_memory(self):
        """Test memory efficiency with large graph."""
        ge = GraphEngine()
        
        # Simulate large graph
        num_nodes = 10000
        num_edges = 50000
        
        ge.data.num_nodes = num_nodes
        ge.data.edge_index = torch.randint(0, num_nodes, (2, num_edges))
        ge.data.x = torch.randn(num_nodes, 768)
        
        # Should not crash
        subgraph = ge.extract_subgraph([0], num_hops=2, max_nodes=100)
        assert subgraph.num_nodes <= 100
        
    @pytest.mark.parametrize("num_hops", [1, 2, 3, 5])
    def test_variable_hops(self, sample_graph_data, num_hops):
        """Test different hop counts."""
        edges_path, nodes_path = sample_graph_data
        ge = GraphEngine()
        ge.load_graph(edges_path, nodes_path)
        ge.data.x = torch.randn(3, 768)
        
        subgraph = ge.extract_subgraph([0], num_hops=num_hops)
        assert subgraph.num_nodes > 0
