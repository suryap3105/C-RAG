"""
C-RAG V3 Production ColBERT Partition Router
Late-Interaction MaxSim Routing with Learned Token Selection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import List, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class ColBERTPartitionRouter:
    """
    Production Partition-Level ColBERT Router.
    Uses MaxSim late interaction for precise partition selection.
    """
    def __init__(self, partition_matrix_path: str = None, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.partition_embs = None  # [NumPartitions, MaxTokens, Hidden]
        self.partition_masks = None  # [NumPartitions, MaxTokens] - valid token mask
        self.encoder = None
        
        if partition_matrix_path:
            self.load(partition_matrix_path)
            
    def _get_encoder(self):
        if self.encoder is None:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer('intfloat/e5-base-v2', device=self.device)
        return self.encoder
        
    def build_partition_matrices(self, graph_engine, num_tokens_per_partition: int = 32):
        """
        Build partition token matrices from graph data.
        Uses K-means clustering on partition node embeddings to select representative tokens.
        """
        if not hasattr(graph_engine.data, 'part_id') or graph_engine.data.part_id is None:
            logger.error("Graph has no partitions. Cannot build ColBERT matrices.")
            return
            
        if graph_engine.data.x is None:
            logger.error("Graph has no embeddings. Compute embeddings first.")
            return
            
        part_ids = graph_engine.data.part_id
        embeddings = graph_engine.data.x
        num_partitions = int(part_ids.max().item()) + 1
        hidden_dim = embeddings.size(1)
        
        logger.info(f"Building ColBERT matrices for {num_partitions} partitions...")
        
        # Initialize matrices
        self.partition_embs = torch.zeros(num_partitions, num_tokens_per_partition, hidden_dim)
        self.partition_masks = torch.zeros(num_partitions, num_tokens_per_partition, dtype=torch.bool)
        
        for pid in range(num_partitions):
            mask = (part_ids == pid)
            partition_nodes = mask.nonzero(as_tuple=True)[0]
            partition_embs = embeddings[partition_nodes]
            
            num_nodes = len(partition_nodes)
            if num_nodes == 0:
                continue
                
            if num_nodes <= num_tokens_per_partition:
                # Use all nodes
                self.partition_embs[pid, :num_nodes] = partition_embs
                self.partition_masks[pid, :num_nodes] = True
            else:
                # K-means clustering to select representative tokens
                try:
                    from sklearn.cluster import KMeans
                    kmeans = KMeans(n_clusters=num_tokens_per_partition, random_state=42, n_init=10)
                    kmeans.fit(partition_embs.cpu().numpy())
                    centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
                    self.partition_embs[pid] = centers
                    self.partition_masks[pid] = True
                except ImportError:
                    # Fallback: random sampling
                    indices = torch.randperm(num_nodes)[:num_tokens_per_partition]
                    self.partition_embs[pid] = partition_embs[indices]
                    self.partition_masks[pid] = True
                    
        # Normalize embeddings
        self.partition_embs = F.normalize(self.partition_embs, p=2, dim=-1)
        self.partition_embs = self.partition_embs.to(self.device)
        self.partition_masks = self.partition_masks.to(self.device)
        
        logger.info(f"ColBERT matrices built: {self.partition_embs.shape}")
        
    def encode_query(self, query: str) -> torch.Tensor:
        """Encode query into token embeddings."""
        encoder = self._get_encoder()
        
        # Get token embeddings (not pooled)
        # For sentence-transformers, we need to access the model directly
        with torch.no_grad():
            encoded = encoder.tokenize([f"query: {query}"])
            
            # Move to device
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            # Get embeddings from transformer
            model_output = encoder[0].auto_model(**encoded)
            token_embs = model_output.last_hidden_state[0]  # [seq_len, hidden]
            
            # Remove special tokens (first and last)
            token_embs = token_embs[1:-1]
            
            # Normalize
            token_embs = F.normalize(token_embs, p=2, dim=-1)
            
        return token_embs
        
    def route(self, query: str, k: int = 5) -> Tuple[List[int], List[float]]:
        """
        Route query to top-k partitions using MaxSim.
        
        Returns:
            partition_ids: List of partition indices
            scores: List of MaxSim scores
        """
        if self.partition_embs is None:
            logger.warning("ColBERT Router not initialized. Returning random partitions.")
            return list(range(min(k, 10))), [1.0] * min(k, 10)
            
        # Encode query
        query_embs = self.encode_query(query)  # [Q, H]
        
        # Compute MaxSim
        # partition_embs: [N, T, H]
        # query_embs: [Q, H]
        # Result: [N] where each value is sum over Q of max over T
        
        # Similarity: [N, Q, T]
        sim = torch.einsum('qh,nth->nqt', query_embs, self.partition_embs)
        
        # Apply mask to ignore padding tokens
        if self.partition_masks is not None:
            mask = self.partition_masks.unsqueeze(1).expand_as(sim)  # [N, Q, T]
            sim = sim.masked_fill(~mask, float('-inf'))
            
        # Max over partition tokens (T) -> [N, Q]
        max_sim, _ = sim.max(dim=2)
        
        # Replace -inf with 0 for padded queries
        max_sim = torch.where(max_sim == float('-inf'), torch.zeros_like(max_sim), max_sim)
        
        # Sum over query tokens (Q) -> [N]
        scores = max_sim.sum(dim=1)
        
        # Top-K
        k = min(k, scores.size(0))
        topk_scores, topk_indices = torch.topk(scores, k=k)
        
        return topk_indices.cpu().tolist(), topk_scores.cpu().tolist()
        
    def save(self, path: str):
        """Save partition matrices."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'partition_embs': self.partition_embs.cpu() if self.partition_embs is not None else None,
            'partition_masks': self.partition_masks.cpu() if self.partition_masks is not None else None
        }
        torch.save(state, path)
        logger.info(f"Saved ColBERT Router to {path}")
        
    def load(self, path: str):
        """Load partition matrices."""
        state = torch.load(path, map_location=self.device)
        self.partition_embs = state['partition_embs']
        self.partition_masks = state.get('partition_masks')
        
        if self.partition_embs is not None:
            self.partition_embs = self.partition_embs.to(self.device)
        if self.partition_masks is not None:
            self.partition_masks = self.partition_masks.to(self.device)
            
        logger.info(f"Loaded ColBERT Router from {path}")
