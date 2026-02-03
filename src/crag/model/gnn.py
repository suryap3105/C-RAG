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
        self.skips = nn.ModuleList() # Residual projections per layer
        
        # Dimensions check
        head_dim = hidden_channels // heads if hidden_channels % heads == 0 else hidden_channels
        
        # First layer
        self.convs.append(GATConv(in_channels, head_dim, heads=heads, concat=True, dropout=dropout))
        self.norms.append(nn.LayerNorm(head_dim * heads))
        self.skips.append(nn.Linear(in_channels, head_dim * heads) if in_channels != head_dim * heads else nn.Identity())
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(head_dim * heads, head_dim, heads=heads, concat=True, dropout=dropout))
            self.norms.append(nn.LayerNorm(head_dim * heads))
            self.skips.append(nn.Identity()) # Same dim
            
        # Output layer
        self.convs.append(GATConv(head_dim * heads, out_channels, heads=1, concat=False, dropout=dropout))
        self.norms.append(nn.LayerNorm(out_channels))
        self.skips.append(nn.Linear(head_dim * heads, out_channels) if head_dim * heads != out_channels else nn.Identity())
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        
        for i, (conv, norm, skip) in enumerate(zip(self.convs, self.norms, self.skips)):
            x_in = x
            x = conv(x, edge_index)
            x = norm(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Residual connection
            x = x + skip(x_in)
            
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
        
        # Input projection to ensure dimension match
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
        
        logger.info(f"NeuralSubgraphMatcher initialized: in={in_channels}, hidden={hidden_channels}, out={out_channels}")
        
    def encode(self, data: Data) -> torch.Tensor:
        """Encode a graph into a fixed-size vector."""
        x, edge_index = data.x, data.edge_index
        
        # Handle batch dimension
        if hasattr(data, 'batch') and data.batch is not None:
            batch = data.batch
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
        # Ensure input features are present and correct dimension
        if x is None:
             # Fallback if no features
             x = torch.zeros((batch.size(0), self.input_proj.in_features), device=data.edge_index.device)

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
        # Encode all graphs
        anchor_emb = self.encode(anchor)
        pos_emb = self.encode(positive)
        
        # Batch negatives efficiently
        neg_batch = Batch.from_data_list(negatives)
        neg_emb = self.encode(neg_batch)
        
        # Ensure dimensions match for broadcast
        # anchor: [batch, dim]
        # pos: [batch, dim]
        # neg: [batch * num_neg, dim] if singular batch, or [batch, num_neg, dim] if handled carefully
        # Here we assume standard training loop passes single items or batches.
        # But if standard training loop passes BATCHES, then anchor is (B, D).
        # We need to compute similarity properly.
        
        # Case 1: Single sample (inference-like)
        if anchor_emb.dim() == 1:
            anchor_emb = anchor_emb.unsqueeze(0)
            pos_emb = pos_emb.unsqueeze(0)

        # Batch Size
        B = anchor_emb.size(0)
        
        # Positive similarity: (B,)
        pos_sim = F.cosine_similarity(anchor_emb, pos_emb, dim=-1) / temperature
        
        # Negative similarity
        # If neg_emb is huge batch (B * K), we need to reshape or mask
        # Assuming neg_batch is B*K stacked
        K = len(negatives) // B if B > 0 else len(negatives)
        
        # Reshape negatives to (B, K, D) if perfectly aligned
        if neg_emb.size(0) == B * K:
            neg_emb = neg_emb.view(B, K, -1)
            # anchor: (B, 1, D)
            anchor_expanded = anchor_emb.unsqueeze(1)
            # sim: (B, K)
            neg_sims = F.cosine_similarity(anchor_expanded, neg_emb, dim=-1) / temperature
        else:
             # Fallback for mismatched batch sizes (shouldn't happen in strict training but safety first)
             # Compute pairwise against ALL negatives
             neg_sims = torch.mm(anchor_emb, neg_emb.t()) / temperature
        
        # Logits: (B, 1 + K)
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sims], dim=1)
        
        # Labels: 0 (positive is first)
        labels = torch.zeros(B, dtype=torch.long, device=anchor_emb.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss
