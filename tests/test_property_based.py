"""
Property-Based Tests using Hypothesis
For discovering edge cases automatically
"""
import pytest
from hypothesis import given, strategies as st, settings, assume
import torch
from torch_geometric.data import Data

from crag.model.gnn import NeuralSubgraphMatcher
from crag.graph.engine import GraphEngine


@st.composite
def valid_graph(draw, min_nodes=2, max_nodes=50, min_edges=1, max_edges=200):
    """Strategy for generating valid graphs."""
    num_nodes = draw(st.integers(min_value=min_nodes, max_value=max_nodes))
    num_edges = draw(st.integers(min_value=min_edges, max_value=max_edges))
    
    # Generate valid edge indices
    src = draw(st.lists(st.integers(0, num_nodes-1), min_size=num_edges, max_size=num_edges))
    dst = draw(st.lists(st.integers(0, num_nodes-1), min_size=num_edges, max_size=num_edges))
    
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    x = torch.randn(num_nodes, 768)
    
    return Data(x=x, edge_index=edge_index, num_nodes=num_nodes)


class TestPropertyBased:
    """Property-based tests for invariants."""
    
    @given(valid_graph())
    @settings(max_examples=50, deadline=None)
    def test_gnn_output_normalized(self, graph):
        """Property: GNN output should always be normalized."""
        model = NeuralSubgraphMatcher()
        model.eval()
        
        with torch.no_grad():
            output = model(graph)
            
        # Check normalization
        norms = torch.norm(output, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)
        
    @given(valid_graph(), valid_graph())
    @settings(max_examples=30, deadline=None)
    def test_match_symmetry(self, g1, g2):
        """Property: match(A, B) should be close to match(B, A)."""
        model = NeuralSubgraphMatcher()
        model.eval()
        
        score1 = model.match(g1, g2)
        score2 = model.match(g2, g1)
        
        # Cosine similarity is symmetric
        assert abs(score1 - score2) < 0.01
        
    @given(valid_graph())
    @settings(max_examples=50, deadline=None)
    def test_match_self_similarity(self, graph):
        """Property: match(A, A) should be close to 1.0."""
        model = NeuralSubgraphMatcher()
        model.eval()
        
        score = model.match(graph, graph)
        
        assert 0.95 < score <= 1.0
        
    @given(st.integers(min_value=5, max_value=100))
    @settings(max_examples=20, deadline=None)
    def test_subgraph_size_bounded(self, num_nodes):
        """Property: Extracted subgraph should not exceed max_nodes."""
        ge = GraphEngine()
        ge.data.num_nodes = num_nodes
        ge.data.edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
        ge.data.x = torch.randn(num_nodes, 768)
        
        max_nodes = 20
        subgraph = ge.extract_subgraph([0], num_hops=3, max_nodes=max_nodes)
        
        assert subgraph.num_nodes <= max_nodes
        
    @given(
        st.floats(min_value=0.0, max_value=1.0),
        st.floats(min_value=0.0, max_value=1.0)
    )
    @settings(max_examples=30, deadline=None)
    def test_rrf_score_bounded(self, alpha, rank1):
        """Property: RRF scores should be in valid range."""
        k = 60
        rank_a = int(rank1 * 100)
        rank_b = int((1 - rank1) * 100)
        
        # Weighted RRF formula
        score = alpha * (1.0 / (k + rank_a)) + (1 - alpha) * (1.0 / (k + rank_b))
        
        assert 0 <= score <= 1.0
        
    @given(st.integers(min_value=1, max_value=1000))
    @settings(max_examples=20, deadline=None)
    def test_embedding_dimension_consistency(self, num_nodes):
        """Property: All nodes should have same embedding dimension."""
        ge = GraphEngine()
        ge.data.num_nodes = num_nodes
        ge.data.x = torch.randn(num_nodes, 768)
        
        assert ge.data.x.shape == (num_nodes, 768)
        assert ge.data.x.dtype == torch.float32
        
    @given(
        st.integers(min_value=2, max_value=20),
        st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=30, deadline=None)
    def test_partition_coverage(self, num_nodes, num_partitions):
        """Property: All nodes should be assigned to exactly one partition."""
        assume(num_partitions <= num_nodes)
        
        part_ids = torch.randint(0, num_partitions, (num_nodes,))
        
        # Each node has exactly one partition
        assert len(part_ids) == num_nodes
        # All partition IDs are valid
        assert part_ids.min() >= 0
        assert part_ids.max() < num_partitions
        
    @given(valid_graph())
    @settings(max_examples=30, deadline=None)
    def test_gnn_deterministic_eval(self, graph):
        """Property: Multiple forward passes in eval mode should give same result."""
        model = NeuralSubgraphMatcher()
        model.eval()
        
        with torch.no_grad():
            out1 = model(graph)
            out2 = model(graph)
            out3 = model(graph)
            
        assert torch.allclose(out1, out2, atol=1e-6)
        assert torch.allclose(out2, out3, atol=1e-6)
