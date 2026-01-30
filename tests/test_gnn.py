"""
Comprehensive Unit Tests for Neural Subgraph Matcher
"""
import pytest
import torch
from torch_geometric.data import Data, Batch

from crag.model.gnn import NeuralSubgraphMatcher, GATEncoder, MultiHeadGATEncoder, DeepGINEncoder


class TestNeuralSubgraphMatcher:
    """Comprehensive test suite for GNN components."""
    
    @pytest.fixture
    def sample_graph(self):
        """Create sample graph."""
        x = torch.randn(10, 768)
        edge_index = torch.tensor([
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8, 9]
        ], dtype=torch.long)
        return Data(x=x, edge_index=edge_index, num_nodes=10)
        
    @pytest.fixture
    def model(self):
        """Create model instance."""
        return NeuralSubgraphMatcher(
            in_channels=768,
            hidden_channels=128,
            out_channels=128
        )
        
    def test_initialization(self):
        """Test model initialization."""
        model = NeuralSubgraphMatcher()
        assert model.input_proj is not None
        assert model.gat_encoder is not None
        assert model.gin_encoder is not None
        
    def test_forward_single_graph(self, model, sample_graph):
        """Test forward pass with single graph."""
        output = model(sample_graph)
        
        assert output.shape == (1, 128)
        assert torch.isfinite(output).all()
        assert torch.allclose(torch.norm(output, dim=-1), torch.ones(1), atol=1e-5)
        
    def test_forward_batched_graphs(self, model):
        """Test forward pass with batched graphs."""
        graphs = [
            Data(x=torch.randn(5, 768), edge_index=torch.randint(0, 5, (2, 10))),
            Data(x=torch.randn(7, 768), edge_index=torch.randint(0, 7, (2, 14))),
            Data(x=torch.randn(3, 768), edge_index=torch.randint(0, 3, (2, 6)))
        ]
        
        batch = Batch.from_data_list(graphs)
        output = model(batch)
        
        assert output.shape == (3, 128)
        assert torch.isfinite(output).all()
        
    def test_match_similar_graphs(self, model):
        """Test matching similar graphs."""
        # Create two similar graphs
        x = torch.randn(5, 768)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        
        g1 = Data(x=x, edge_index=edge_index, num_nodes=5)
        g2 = Data(x=x + 0.01 * torch.randn_like(x), edge_index=edge_index, num_nodes=5)
        
        score = model.match(g1, g2)
        assert 0.5 < score < 1.0  # Should be similar
        
    def test_match_dissimilar_graphs(self, model):
        """Test matching dissimilar graphs."""
        g1 = Data(x=torch.randn(5, 768), edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long), num_nodes=5)
        g2 = Data(x=torch.randn(10, 768), edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long), num_nodes=10)
        
        score = model.match(g1, g2)
        assert -1.0 <= score <= 1.0
        
    def test_match_batch(self, model):
        """Test batched matching."""
        query = Data(x=torch.randn(3, 768), edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long), num_nodes=3)
        
        candidates = [
            Data(x=torch.randn(5, 768), edge_index=torch.randint(0, 5, (2, 8)), num_nodes=5),
            Data(x=torch.randn(4, 768), edge_index=torch.randint(0, 4, (2, 6)), num_nodes=4),
            Data(x=torch.randn(6, 768), edge_index=torch.randint(0, 6, (2, 10)), num_nodes=6)
        ]
        
        scores = model.match_batch(query, candidates)
        
        assert scores.shape == (3,)
        assert torch.isfinite(scores).all()
        assert (scores >= -1.0).all() and (scores <= 1.0).all()
        
    def test_contrastive_loss(self, model):
        """Test contrastive loss computation."""
        anchor = Data(x=torch.randn(4, 768), edge_index=torch.randint(0, 4, (2, 6)), num_nodes=4)
        positive = Data(x=torch.randn(4, 768), edge_index=torch.randint(0, 4, (2, 6)), num_nodes=4)
        negatives = [
            Data(x=torch.randn(5, 768), edge_index=torch.randint(0, 5, (2, 8)), num_nodes=5),
            Data(x=torch.randn(3, 768), edge_index=torch.randint(0, 3, (2, 4)), num_nodes=3)
        ]
        
        loss = model.compute_contrastive_loss(anchor, positive, negatives)
        
        assert torch.isfinite(loss)
        assert loss > 0
        
    def test_gradient_flow(self, model, sample_graph):
        """Test gradient flow through model."""
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        output = model(sample_graph)
        loss = output.sum()
        
        optimizer.zero_grad()
        loss.backward()
        
        # Check gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert torch.isfinite(param.grad).all()
                
    def test_eval_mode(self, model, sample_graph):
        """Test evaluation mode behavior."""
        model.eval()
        
        with torch.no_grad():
            output1 = model(sample_graph)
            output2 = model(sample_graph)
            
        # Should be deterministic in eval mode
        assert torch.allclose(output1, output2)
        
    def test_empty_graph(self, model):
        """Test handling of empty graph."""
        empty_graph = Data(x=torch.randn(1, 768), edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=1)
        
        output = model(empty_graph)
        assert output.shape == (1, 128)
        assert torch.isfinite(output).all()
        
    def test_self_loop_graph(self, model):
        """Test graph with self-loops."""
        x = torch.randn(5, 768)
        edge_index = torch.tensor([[0, 1, 2, 0], [1, 2, 3, 0]], dtype=torch.long)  # Self-loop at 0
        graph = Data(x=x, edge_index=edge_index, num_nodes=5)
        
        output = model(graph)
        assert torch.isfinite(output).all()
        
    def test_large_graph(self, model):
        """Test with large graph."""
        x = torch.randn(1000, 768)
        edge_index = torch.randint(0, 1000, (2, 5000))
        graph = Data(x=x, edge_index=edge_index, num_nodes=1000)
        
        output = model(graph)
        assert output.shape == (1, 128)
        
    @pytest.mark.parametrize("in_dim,hidden_dim,out_dim", [
        (128, 64, 64),
        (768, 256, 256),
        (512, 128, 64),
    ])
    def test_different_dimensions(self, in_dim, hidden_dim, out_dim):
        """Test with different dimension configurations."""
        model = NeuralSubgraphMatcher(in_dim, hidden_dim, out_dim)
        graph = Data(x=torch.randn(10, in_dim), edge_index=torch.randint(0, 10, (2, 20)), num_nodes=10)
        
        output = model(graph)
        assert output.shape == (1, out_dim)
        
    def test_cuda_compatibility(self, model, sample_graph):
        """Test CUDA compatibility if available."""
        if torch.cuda.is_available():
            model = model.cuda()
            sample_graph = sample_graph.cuda()
            
            output = model(sample_graph)
            assert output.is_cuda
            assert torch.isfinite(output).all()
            
    def test_save_load_state_dict(self, model, tmp_path):
        """Test save/load state dict."""
        save_path = tmp_path / "model.pt"
        
        torch.save(model.state_dict(), save_path)
        
        new_model = NeuralSubgraphMatcher()
        new_model.load_state_dict(torch.load(save_path))
        
        # Should produce same output
        graph = Data(x=torch.randn(5, 768), edge_index=torch.randint(0, 5, (2, 10)), num_nodes=5)
        
        with torch.no_grad():
            out1 = model(graph)
            out2 = new_model(graph)
            
        assert torch.allclose(out1, out2, atol=1e-6)


class TestGATEncoder:
    """Test GAT encoder component."""
    
    def test_basic_forward(self):
        """Test basic GAT forward pass."""
        encoder = GATEncoder(in_channels=768, hidden_channels=128, out_channels=128)
        
        x = torch.randn(10, 768)
        edge_index = torch.randint(0, 10, (2, 20))
        
        output = encoder(x, edge_index)
        
        assert output.shape == (10, 128)
        assert torch.isfinite(output).all()
        
    def test_attention_weights(self):
        """Test that attention mechanism works."""
        encoder = MultiHeadGATEncoder(
            in_channels=64,
            hidden_channels=32,
            out_channels=32,
            num_layers=2,
            heads=4
        )
        
        x = torch.randn(5, 64)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        
        output = encoder(x, edge_index)
        assert output.shape == (5, 32)


class TestDeepGINEncoder:
    """Test GIN encoder component."""
    
    def test_jk_aggregation(self):
        """Test JK-Net aggregation."""
        encoder = DeepGINEncoder(
            in_channels=128,
            hidden_channels=64,
            out_channels=64,
            num_layers=3
        )
        
        x = torch.randn(8, 128)
        edge_index = torch.randint(0, 8, (2, 16))
        
        output, layer_outputs = encoder(x, edge_index)
        
        assert output.shape == (8, 64)
        assert len(layer_outputs) == 3
        assert all(lo.shape == (8, 64) for lo in layer_outputs)
