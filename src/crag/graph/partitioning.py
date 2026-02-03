"""
C-RAG V3 Semantic Partitioner
Separates Knowledge Graph into coherent semantic clusters.
"""
import torch
import logging
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from typing import Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)

class SemanticPartitioner:
    """
    Partitions the Knowledge Graph into semantic communities.
    Uses NetworkX community detection (Greedy Modularity).
    """
    def __init__(self, resolution: float = 1.0):
        self.resolution = resolution
        
    def partition(self, data: Data) -> torch.Tensor:
        """
        Compute partitions for the graph with multiple failsafe strategies.
        Returns:
            part_id: Tensor of shape [num_nodes] with partition IDs.
        """
        logger.info(f"Starting semantic partitioning for {data.num_nodes} nodes...")
        
        # Validate input
        if data.num_nodes == 0:
            logger.warning("Empty graph provided, returning empty partition tensor.")
            return torch.tensor([], dtype=torch.long)
            
        if data.edge_index.size(1) == 0:
            logger.warning("Graph has no edges. Assigning single partition.")
            return torch.zeros(data.num_nodes, dtype=torch.long)
        
        try:
            # Convert to NetworkX for community detection
            G = to_networkx(data, to_undirected=True)
            
            # Check if graph is disconnected
            if not nx.is_connected(G):
                logger.info("Graph is disconnected. Using connected components as base partitions.")
                return self._partition_disconnected(G, data.num_nodes)
            
            # Check if graph is large
            if data.num_nodes > 10000:
                logger.warning("Large graph detected. Using fast approximation.")
                return self._partition_large_graph(G, data.num_nodes)
            
            # Standard partitioning
            return self._partition_standard(G, data.num_nodes)
            
        except Exception as e:
            logger.error(f"Partitioning failed: {e}. Using fallback strategy.")
            return self._fallback_partition(data)
    
    def _partition_standard(self, G: nx.Graph, num_nodes: int) -> torch.Tensor:
        """Standard partitioning using greedy modularity."""
        try:
            from networkx.algorithms.community import greedy_modularity_communities
            
            communities = greedy_modularity_communities(G, resolution=self.resolution)
            
            # Create partition tensor
            part_id = torch.zeros(num_nodes, dtype=torch.long)
            
            for pid, community in enumerate(communities):
                for node_idx in community:
                    part_id[node_idx] = pid
                    
            num_communities = len(communities)
            logger.info(f"Partitioning complete. Found {num_communities} communities.")
            
            return part_id
            
        except ImportError:
            logger.warning("Greedy modularity not available. Using label propagation.")
            return self._partition_label_propagation(G, num_nodes)
    
    def _partition_large_graph(self, G: nx.Graph, num_nodes: int) -> torch.Tensor:
        """Fast partitioning for large graphs using label propagation."""
        return self._partition_label_propagation(G, num_nodes)
    
    def _partition_label_propagation(self, G: nx.Graph, num_nodes: int) -> torch.Tensor:
        """Partition using label propagation algorithm (faster)."""
        try:
            from networkx.algorithms.community import label_propagation_communities
            
            communities = list(label_propagation_communities(G))
            
            part_id = torch.zeros(num_nodes, dtype=torch.long)
            for pid, community in enumerate(communities):
                for node_idx in community:
                    part_id[node_idx] = pid
                    
            logger.info(f"Label propagation complete. Found {len(communities)} communities.")
            return part_id
            
        except Exception as e:
            logger.error(f"Label propagation failed: {e}")
            raise
    
    def _partition_disconnected(self, G: nx.Graph, num_nodes: int) -> torch.Tensor:
        """Handle disconnected graphs by using connected components."""
        part_id = torch.zeros(num_nodes, dtype=torch.long)
        
        for pid, component in enumerate(nx.connected_components(G)):
            for node_idx in component:
                part_id[node_idx] = pid
                
        logger.info(f"Partitioned disconnected graph into {pid + 1} components.")
        return part_id
    
    def _fallback_partition(self, data: Data) -> torch.Tensor:
        """Ultimate fallback: degree-based partitioning."""
        logger.warning("Using degree-based fallback partitioning.")
        
        # Calculate node degrees
        edge_index = data.edge_index
        degrees = torch.zeros(data.num_nodes, dtype=torch.long)
        
        for i in range(edge_index.size(1)):
            degrees[edge_index[0, i]] += 1
            
        # Create 5 partitions based on degree quantiles
        if data.num_nodes < 5:
            return torch.arange(data.num_nodes, dtype=torch.long)
            
        sorted_indices = torch.argsort(degrees)
        part_id = torch.zeros(data.num_nodes, dtype=torch.long)
        
        chunk_size = data.num_nodes // 5
        for i in range(5):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < 4 else data.num_nodes
            part_id[sorted_indices[start:end]] = i
            
        logger.info("Fallback partitioning complete with 5 degree-based partitions.")
        return part_id
            
    def compute_partition_centers(self, data: Data, part_id: torch.Tensor) -> torch.Tensor:
        """
        Compute centroid embeddings for each partition.
        """
        if data.x is None:
            logger.warning("No node embeddings found. Cannot compute partition centers.")
            return None
            
        num_partitions = int(part_id.max().item()) + 1
        hidden_dim = data.x.size(1)
        
        centers = torch.zeros(num_partitions, hidden_dim, device=data.x.device)
        
        for pid in range(num_partitions):
            mask = (part_id == pid)
            if mask.sum() > 0:
                centers[pid] = data.x[mask].mean(dim=0)
                
        # Normalize
        centers = torch.nn.functional.normalize(centers, p=2, dim=-1)
        
        return centers
