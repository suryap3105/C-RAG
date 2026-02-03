"""
C-RAG Simple ColBERT Router
Text-only partition routing without query graphs.
Optimized for scenarios where FAISS fails due to noise or dimensionality.
"""
import torch
import torch.nn.functional as F
import logging
from typing import List, Tuple
from pathlib import Path
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class SimpleColBERTRouter:
    """
    Simplified ColBERT partition router for pure text queries.
    
    Use this when:
    - FAISS fails with muddied/noisy knowledge graphs
    - Dimensionality reduction causes accuracy loss
    - Schema information unavailable
    - Maximum speed required
    """
    
    def __init__(self, partition_matrix_path: str = None, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder_name = 'sentence-transformers/all-MiniLM-L6-v2'
        
        # Partition data
        self.partition_embs = None  # [NumPartitions, MaxTokens, Hidden]
        self.partition_masks = None  # [NumPartitions, MaxTokens]
        self.num_partitions = 0
        
        self._encoder = None
        
        if partition_matrix_path and Path(partition_matrix_path).exists():
            self.load(partition_matrix_path)
            
        logger.info(f"SimpleColBERTRouter initialized on {self.device}")
    
    def _get_encoder(self):
        """Lazy load encoder."""
        if self._encoder is None:
            self._encoder = SentenceTransformer(self.encoder_name)
            self._encoder.to(self.device)
        return self._encoder
    
    def build_partition_matrices(self, graph_engine, num_tokens_per_partition: int = 32):
        """
        Build partition matrices from graph engine.
        
        Simplified version:
        - No metadata extraction
        - Pure embedding-based
        - Fast initialization
        """
        logger.info("Building partition matrices (simplified)...")
        
        if not hasattr(graph_engine.data, 'part_id'):
            raise ValueError("Graph must be partitioned first. Run SemanticPartitioner.")
        
        part_ids = graph_engine.data.part_id
        self.num_partitions = int(part_ids.max().item()) + 1
        
        encoder = self._get_encoder()
        hidden_dim = encoder.get_sentence_embedding_dimension()
        
        # Initialize tensors
        self.partition_embs = torch.zeros(
            self.num_partitions, num_tokens_per_partition, hidden_dim,
            dtype=torch.float32
        )
        self.partition_masks = torch.zeros(
            self.num_partitions, num_tokens_per_partition,
            dtype=torch.bool
        )
        
        # Build partition representations
        for pid in range(self.num_partitions):
            partition_nodes = (part_ids == pid).nonzero(as_tuple=True)[0]
            num_nodes = len(partition_nodes)
            
            if num_nodes == 0:
                continue
            
            # Sample nodes for efficiency
            if num_nodes > 100:
                indices = torch.randperm(num_nodes)[:100]
                partition_nodes = partition_nodes[indices]
            
            # Extract node texts
            node_texts = []
            for node_idx in partition_nodes.tolist():
                if node_idx in graph_engine.node_text_map:
                    text = graph_engine.node_text_map[node_idx].get('text', '')
                    if text:
                        node_texts.append(text)
            
            if not node_texts:
                continue
            
            # Encode partition text
            partition_text = " ".join(node_texts[:10])  # Sample first 10
            
            with torch.no_grad():
                encoded = encoder.tokenize([f"passage: {partition_text}"])
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                
                model_output = encoder[0].auto_model(**encoded)
                token_embs = model_output.last_hidden_state[0]  # [seq_len, hidden]
                
                # Remove special tokens
                token_embs = token_embs[1:-1]
                token_embs = F.normalize(token_embs, p=2, dim=-1)
                
                # Store (pad or truncate)
                actual_len = min(token_embs.size(0), num_tokens_per_partition)
                self.partition_embs[pid, :actual_len] = token_embs[:actual_len].cpu()
                self.partition_masks[pid, :actual_len] = True
        
        # Move to device
        self.partition_embs = self.partition_embs.to(self.device)
        self.partition_masks = self.partition_masks.to(self.device)
        
        logger.info(f"✓ Built matrices for {self.num_partitions} partitions")
    
    def encode_query(self, query: str) -> torch.Tensor:
        """Encode query into token embeddings."""
        encoder = self._get_encoder()
        
        with torch.no_grad():
            encoded = encoder.tokenize([f"query: {query}"])
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            model_output = encoder[0].auto_model(**encoded)
            token_embs = model_output.last_hidden_state[0]
            
            # Remove special tokens
            token_embs = token_embs[1:-1]
            token_embs = F.normalize(token_embs, p=2, dim=-1)
            
        return token_embs
    
    def route(self, query: str, k: int = 5) -> Tuple[List[int], List[float]]:
        """
        Route text query to top-k partitions using ColBERT MaxSim.
        
        Args:
            query: Natural language text query (no graph structure needed)
            k: Number of partitions to return
            
        Returns:
            partition_ids: List of partition indices
            scores: List of ColBERT MaxSim scores
        """
        if self.partition_embs is None:
            logger.warning("Router not initialized. Returning default partitions.")
            return list(range(min(k, 10))), [1.0] * min(k, 10)
        
        try:
            # Encode query
            query_embs = self.encode_query(query)  # [Q, H]
            
            # Compute MaxSim: [N, Q, T]
            sim = torch.einsum('qh,nth->nqt', query_embs, self.partition_embs)
            
            # Apply mask
            if self.partition_masks is not None:
                mask = self.partition_masks.unsqueeze(1).expand_as(sim)
                sim = sim.masked_fill(~mask, float('-inf'))
            
            # Max over partition tokens (T) -> [N, Q]
            max_sim, _ = sim.max(dim=2)
            max_sim = torch.where(max_sim == float('-inf'), torch.zeros_like(max_sim), max_sim)
            
            # Sum over query tokens (Q) -> [N]
            scores = max_sim.sum(dim=1)
            
            # Normalize
            if query_embs.size(0) > 0:
                scores = scores / query_embs.size(0)
            
            # Top-K
            k = min(k, scores.size(0))
            if k == 0:
                return [], []
            
            topk_scores, topk_indices = torch.topk(scores, k=k)
            
            return topk_indices.cpu().tolist(), topk_scores.cpu().tolist()
            
        except Exception as e:
            logger.error(f"Routing failed: {e}. Returning fallback.")
            num_partitions = self.partition_embs.size(0) if self.partition_embs is not None else 10
            k = min(k, num_partitions)
            return list(range(k)), [1.0] * k
    
    def save(self, path: str):
        """Save partition matrices."""
        torch.save({
            'partition_embs': self.partition_embs.cpu() if self.partition_embs is not None else None,
            'partition_masks': self.partition_masks.cpu() if self.partition_masks is not None else None,
            'num_partitions': self.num_partitions,
            'encoder_name': self.encoder_name
        }, path)
        logger.info(f"Saved router to {path}")
    
    def load(self, path: str):
        """Load partition matrices."""
        checkpoint = torch.load(path, map_location='cpu')
        self.partition_embs = checkpoint['partition_embs'].to(self.device)
        self.partition_masks = checkpoint['partition_masks'].to(self.device)
        self.num_partitions = checkpoint['num_partitions']
        self.encoder_name = checkpoint.get('encoder_name', self.encoder_name)
        logger.info(f"Loaded router from {path}")
