"""
C-RAG V3 Production Neural Subgraph Matcher
Research-Grade GAT + GIN with Multi-Scale Aggregation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GINConv, GCNConv, global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MultiHeadGATEncoder(nn.Module):
    """
    Multi-layer Graph Attention Network with residual connections.
    Research-grade implementation with dropout and layer normalization.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, 
                 num_layers: int = 3, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # First layer
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, concat=True, dropout=dropout))
        self.norms.append(nn.LayerNorm(hidden_channels * heads))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True, dropout=dropout))
            self.norms.append(nn.LayerNorm(hidden_channels * heads))
            
        # Output layer
        self.convs.append(GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout))
        self.norms.append(nn.LayerNorm(out_channels))
        
        # Residual projection if dimensions mismatch
        self.residual_proj = nn.Linear(in_channels, out_channels) if in_channels != out_channels else None
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        residual = x
        
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        # Residual connection
        if self.residual_proj is not None:
            residual = self.residual_proj(residual)
        if x.shape == residual.shape:
            x = x + residual
            
        return x


class DeepGINEncoder(nn.Module):
    """
    Deep Graph Isomorphism Network with JK-Net style aggregation.
    Captures multi-scale structural patterns.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_layers: int = 5, dropout: float = 0.1):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU()
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.norms.append(nn.LayerNorm(hidden_channels))
            
        # JK-Net style: concatenate all layer outputs
        self.jk_proj = nn.Linear(hidden_channels * num_layers, out_channels)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        layer_outputs = []
        
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_outputs.append(x)
            
        # JK aggregation
        jk_out = torch.cat(layer_outputs, dim=-1)
        out = self.jk_proj(jk_out)
        
        return out, layer_outputs


class MultiScalePooling(nn.Module):
    """
    Multi-scale graph pooling combining add, mean, and max.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Three pooling strategies concatenated
        self.proj = nn.Linear(in_channels * 3, out_channels)
        
    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        add_pool = global_add_pool(x, batch)
        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        
        pooled = torch.cat([add_pool, mean_pool, max_pool], dim=-1)
        return self.proj(pooled)


class NeuralSubgraphMatcher(nn.Module):
    """
    Production Research-Grade Neural Subgraph Matcher.
    
    Architecture:
    1. Multi-Head GAT for local node attention
    2. Deep GIN for structural pattern recognition
    3. Multi-scale pooling for graph-level representation
    4. Contrastive matching head
    """
    def __init__(self, in_channels: int = 768, hidden_channels: int = 256, 
                 out_channels: int = 256, num_gat_layers: int = 2,
                 num_gin_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # GAT for local attention
        self.gat_encoder = MultiHeadGATEncoder(
            hidden_channels, hidden_channels, hidden_channels,
            num_layers=num_gat_layers, heads=4, dropout=dropout
        )
        
        # GIN for structural patterns
        self.gin_encoder = DeepGINEncoder(
            hidden_channels, hidden_channels, hidden_channels,
            num_layers=num_gin_layers, dropout=dropout
        )
        
        # Multi-scale pooling
        self.pooling = MultiScalePooling(hidden_channels, hidden_channels)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_channels, out_channels),
            nn.LayerNorm(out_channels)
        )
        
        # Matching head (for training)
        self.match_head = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, 1)
        )
        
        logger.info(f"NeuralSubgraphMatcher initialized: in={in_channels}, hidden={hidden_channels}, out={out_channels}")
        
    def encode(self, data: Data) -> torch.Tensor:
        """Encode a graph into a fixed-size vector."""
        x, edge_index = data.x, data.edge_index
        
        # Handle batch dimension
        if hasattr(data, 'batch') and data.batch is not None:
            batch = data.batch
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
        # Input projection
        x = self.input_proj(x)
        
        # GAT encoding
        x = self.gat_encoder(x, edge_index)
        
        # GIN encoding
        x, _ = self.gin_encoder(x, edge_index)
        
        # Pooling
        graph_emb = self.pooling(x, batch)
        
        # Output projection + normalize
        graph_emb = self.output_proj(graph_emb)
        graph_emb = F.normalize(graph_emb, p=2, dim=-1)
        
        return graph_emb
        
    def forward(self, data: Data) -> torch.Tensor:
        """Alias for encode."""
        return self.encode(data)
        
    def match(self, query_graph: Data, candidate_graph: Data) -> float:
        """
        Compute similarity between query and candidate graphs.
        Returns cosine similarity.
        """
        with torch.no_grad():
            q_emb = self.encode(query_graph)
            c_emb = self.encode(candidate_graph)
            
            similarity = F.cosine_similarity(q_emb, c_emb, dim=-1)
            return similarity.item()
            
    def match_batch(self, query_graph: Data, candidates: List[Data]) -> torch.Tensor:
        """
        Match query against batch of candidates.
        Returns tensor of similarity scores.
        """
        with torch.no_grad():
            q_emb = self.encode(query_graph)
            
            # Batch candidates
            batch = Batch.from_data_list(candidates)
            c_embs = self.encode(batch)
            
            # Cosine similarity
            similarities = F.cosine_similarity(q_emb.expand_as(c_embs), c_embs, dim=-1)
            return similarities
            
    def compute_contrastive_loss(self, anchor: Data, positive: Data, negatives: List[Data],
                                  temperature: float = 0.07) -> torch.Tensor:
        """
        InfoNCE contrastive loss for training.
        """
        anchor_emb = self.encode(anchor)
        pos_emb = self.encode(positive)
        
        neg_batch = Batch.from_data_list(negatives)
        neg_embs = self.encode(neg_batch)
        
        # Positive similarity
        pos_sim = F.cosine_similarity(anchor_emb, pos_emb, dim=-1) / temperature
        
        # Negative similarities
        neg_sims = F.cosine_similarity(anchor_emb.expand_as(neg_embs), neg_embs, dim=-1) / temperature
        
        # InfoNCE
        logits = torch.cat([pos_sim.unsqueeze(0), neg_sims])
        labels = torch.zeros(1, dtype=torch.long, device=anchor_emb.device)
        
        loss = F.cross_entropy(logits.unsqueeze(0), labels)
        return loss


class GATEncoder(nn.Module):
    """Simpler GAT encoder for backward compatibility."""
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, heads: int = 4):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.elu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x
