"""
C-RAG V3 Production ColBERT Partition Router
Late-Interaction MaxSim Routing with Learned Token Selection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import List, Tuple, Optional, Any
from pathlib import Path
from .structural import StructuralAligner

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
        self.partition_metadata = [] # List[Dict] - fingerprints for each partition
        self.partition_centroids = None # [NumPartitions, Hidden]
        self.encoder = None
        self.structural_aligner = StructuralAligner()

        
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
        self.partition_centroids = torch.zeros(num_partitions, hidden_dim)
        self.partition_metadata = [{} for _ in range(num_partitions)]
        
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
                    self.partition_embs[pid] = partition_embs[indices]
                    self.partition_masks[pid] = True
            
            # Compute centroid
            self.partition_centroids[pid] = partition_embs.mean(dim=0)
            
            # Extract structural metadata from GraphEngine
            node_types_set = set()
            edge_types_set = set()
            
            try:
                # Extract node types from partition nodes
                for node_idx in partition_nodes[:100].tolist():  # Sample first 100 for efficiency
                    node_data = graph_engine.node_text_map.get(node_idx, {})
                    if 'type' in node_data:
                        node_types_set.add(node_data['type'])
                    elif 'label' in node_data:
                        node_types_set.add(node_data['label'])
                
                # Extract edge types from edges within partition
                edge_index = graph_engine.data.edge_index
                for i in range(min(edge_index.size(1), 1000)):  # Sample edges
                    src, dst = edge_index[0, i].item(), edge_index[1, i].item()
                    if src in partition_nodes and dst in partition_nodes:
                        edge_key = (src, dst)
                        edge_attrs = graph_engine.edge_attr_map.get(edge_key, {})
                        if 'relation' in edge_attrs:
                            edge_types_set.add(edge_attrs['relation'])
                        elif 'type' in edge_attrs:
                            edge_types_set.add(edge_attrs['type'])
                            
            except Exception as e:
                logger.warning(f"Failed to extract metadata for partition {pid}: {e}")
                # Failsafe: continue with empty metadata
            
            self.partition_metadata[pid] = {
                'num_nodes': num_nodes,
                'part_id': pid,
                'node_types': node_types_set,
                'edge_types': edge_types_set
            }
                    
        # Normalize embeddings
        self.partition_embs = F.normalize(self.partition_embs, p=2, dim=-1)
        self.partition_embs = self.partition_embs.to(self.device)
        self.partition_masks = self.partition_masks.to(self.device)
        self.partition_centroids = F.normalize(self.partition_centroids, p=2, dim=-1).to(self.device)

        
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
        
    def route(self, query: str, query_graph: Optional[Any] = None, k: int = 5,
              weights: Tuple[float, float, float] = (0.4, 0.3, 0.3)) -> Tuple[List[int], List[float]]:
        """
        Route query to top-k partitions using Hybrid Scoring.
        
        Args:
            query: Natural language query
            query_graph: Optional parsed QueryGraph object
            k: Top-k
            weights: (alpha_vec, beta_struct, gamma_colbert)
        
        Returns:
            partition_ids: List of partition indices
            scores: List of Hybrid scores
        """

        if self.partition_embs is None:
            logger.warning("ColBERT Router not initialized. Returning random partitions.")
            return list(range(min(k, 10))), [1.0] * min(k, 10)
        
        # Validate and normalize weights
        alpha, beta, gamma = weights
        weight_sum = alpha + beta + gamma
        if weight_sum == 0:
            logger.warning("All weights are zero. Using equal weights.")
            alpha, beta, gamma = 0.33, 0.33, 0.34
        elif abs(weight_sum - 1.0) > 0.01:
            logger.warning(f"Weights sum to {weight_sum}, normalizing to 1.0")
            alpha, beta, gamma = alpha/weight_sum, beta/weight_sum, gamma/weight_sum
        
        try:
            # 1. Encode query
            query_embs = self.encode_query(query)  # [Q, H]
            query_vec = query_embs.mean(dim=0) # [H]
            
            # 2. Vector Score: Cosine(Query, Partition Centroid)
            if self.partition_centroids is not None:
                vec_scores = torch.mv(self.partition_centroids, query_vec) # [N]
                # Clamp to valid range
                vec_scores = torch.clamp(vec_scores, -1.0, 1.0)
            else:
                logger.warning("No partition centroids available. Skipping vector score.")
                vec_scores = torch.zeros(self.partition_embs.size(0), device=self.device)
            
            # 3. ColBERT MaxSim Score
            # partition_embs: [N, T, H]
            # query_embs: [Q, H]
            
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
            colbert_scores = max_sim.sum(dim=1)
            
            # Normalize ColBERT scores to [0,1] range
            if query_embs.size(0) > 0:
                colbert_scores = colbert_scores / query_embs.size(0)
            
            # 4. Structural Score
            struct_scores = torch.zeros(self.partition_embs.size(0), device=self.device)
            if query_graph and hasattr(query_graph, 'num_nodes') and query_graph.num_nodes > 0:
                try:
                    s_scores = self.structural_aligner.batch_score(query_graph, self.partition_metadata)
                    struct_scores = torch.tensor(s_scores, device=self.device, dtype=torch.float32)
                except Exception as e:
                    logger.error(f"Structural scoring failed: {e}. Skipping structural component.")
                    
            # 5. Hybrid Combination with normalization
            # Normalize each component to [0, 1] if needed
            if vec_scores.abs().max() > 1e-6:
                vec_scores = (vec_scores - vec_scores.min()) / (vec_scores.max() - vec_scores.min() + 1e-8)
            
            final_scores = (alpha * vec_scores) + (beta * struct_scores) + (gamma * colbert_scores)
            
            # Top-K with validation
            k = min(k, final_scores.size(0))
            if k == 0:
                logger.warning("No partitions available.")
                return [], []
                
            topk_scores, topk_indices = torch.topk(final_scores, k=k)
            
            return topk_indices.cpu().tolist(), topk_scores.cpu().tolist()
            
        except Exception as e:
            logger.error(f"Hybrid routing failed: {e}. Falling back to random selection.")
            # Failsafe: return first k partitions
            num_partitions = self.partition_embs.size(0) if self.partition_embs is not None else 10
            k = min(k, num_partitions)
            return list(range(k)), [1.0] * k
        
    def save(self, path: str):
        """Save partition matrices."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'partition_embs': self.partition_embs.cpu() if self.partition_embs is not None else None,
            'partition_masks': self.partition_masks.cpu() if self.partition_masks is not None else None,
            'partition_centroids': self.partition_centroids.cpu() if self.partition_centroids is not None else None,
            'partition_metadata': self.partition_metadata
        }

        torch.save(state, path)
        logger.info(f"Saved ColBERT Router to {path}")
        
    def load(self, path: str):
        """Load partition matrices."""
        state = torch.load(path, map_location=self.device)
        self.partition_embs = state['partition_embs']
        self.partition_masks = state.get('partition_masks')
        self.partition_centroids = state.get('partition_centroids')
        self.partition_metadata = state.get('partition_metadata', [])
        
        if self.partition_embs is not None:
            self.partition_embs = self.partition_embs.to(self.device)
        if self.partition_masks is not None:
            self.partition_masks = self.partition_masks.to(self.device)
        if self.partition_centroids is not None:
            self.partition_centroids = self.partition_centroids.to(self.device)

            
        logger.info(f"Loaded ColBERT Router from {path}")
