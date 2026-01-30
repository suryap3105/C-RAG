"""
C-RAG V3 Production Graph Partitioner
METIS + Leiden + Spectral Partitioning
"""
import torch
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class GraphPartitioner:
    """
    Production Graph Partitioner.
    Supports METIS, Leiden, and Spectral partitioning algorithms.
    """
    def __init__(self, method: str = "metis", n_partitions: int = 10):
        self.method = method.lower()
        self.n_partitions = n_partitions
        
    def partition(self, graph_engine) -> torch.Tensor:
        """
        Partition the graph and assign partition IDs to nodes.
        Returns partition assignment tensor.
        """
        edge_index = graph_engine.data.edge_index
        num_nodes = graph_engine.data.num_nodes
        
        logger.info(f"Partitioning {num_nodes} nodes into {self.n_partitions} partitions using {self.method}")
        
        if self.method == "metis":
            part_ids = self._metis_partition(edge_index, num_nodes)
        elif self.method == "leiden":
            part_ids = self._leiden_partition(edge_index, num_nodes)
        elif self.method == "spectral":
            part_ids = self._spectral_partition(edge_index, num_nodes)
        else:
            # Random fallback
            part_ids = torch.randint(0, self.n_partitions, (num_nodes,))
            
        # Assign to graph
        graph_engine.data.part_id = part_ids
        
        # Log distribution
        unique, counts = torch.unique(part_ids, return_counts=True)
        logger.info(f"Partition distribution: {dict(zip(unique.tolist(), counts.tolist()))}")
        
        return part_ids
        
    def _metis_partition(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """METIS-based graph partitioning."""
        try:
            import pymetis
            
            # Build adjacency list
            adjacency = [[] for _ in range(num_nodes)]
            src, dst = edge_index[0].tolist(), edge_index[1].tolist()
            
            for s, d in zip(src, dst):
                if s != d:  # No self-loops
                    adjacency[s].append(d)
                    
            # Handle isolated nodes
            for i in range(num_nodes):
                if not adjacency[i]:
                    # Connect to next node
                    if i < num_nodes - 1:
                        adjacency[i].append(i + 1)
                        adjacency[i + 1].append(i)
                        
            # Run METIS
            n_cuts, membership = pymetis.part_graph(self.n_partitions, adjacency=adjacency)
            logger.info(f"METIS: {n_cuts} edge cuts")
            
            return torch.tensor(membership, dtype=torch.long)
            
        except ImportError:
            logger.warning("pymetis not installed. Falling back to random.")
            return torch.randint(0, self.n_partitions, (num_nodes,))
        except Exception as e:
            logger.error(f"METIS error: {e}. Falling back to random.")
            return torch.randint(0, self.n_partitions, (num_nodes,))
            
    def _leiden_partition(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Leiden community detection."""
        try:
            import igraph as ig
            import leidenalg
            
            # Build igraph
            edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))
            g = ig.Graph(n=num_nodes, edges=edges, directed=False)
            
            # Run Leiden
            partition = leidenalg.find_partition(
                g, 
                leidenalg.ModularityVertexPartition,
                n_iterations=10
            )
            
            membership = partition.membership
            
            # Map to fixed number of partitions
            unique_parts = list(set(membership))
            if len(unique_parts) > self.n_partitions:
                # Merge small partitions
                part_map = {p: i % self.n_partitions for i, p in enumerate(unique_parts)}
                membership = [part_map[m] for m in membership]
                
            return torch.tensor(membership, dtype=torch.long)
            
        except ImportError:
            logger.warning("leidenalg not installed. Falling back to spectral.")
            return self._spectral_partition(edge_index, num_nodes)
            
    def _spectral_partition(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Spectral clustering partition."""
        try:
            from sklearn.cluster import SpectralClustering
            from scipy.sparse import csr_matrix
            
            # Build sparse adjacency
            src, dst = edge_index[0].numpy(), edge_index[1].numpy()
            data = np.ones(len(src))
            adj = csr_matrix((data, (src, dst)), shape=(num_nodes, num_nodes))
            adj = adj + adj.T  # Symmetrize
            adj = (adj > 0).astype(float)
            
            # Run spectral clustering
            clustering = SpectralClustering(
                n_clusters=self.n_partitions,
                affinity='precomputed',
                random_state=42,
                n_init=10
            )
            labels = clustering.fit_predict(adj.toarray())
            
            return torch.tensor(labels, dtype=torch.long)
            
        except Exception as e:
            logger.warning(f"Spectral clustering failed: {e}. Using random.")
            return torch.randint(0, self.n_partitions, (num_nodes,))
            
    def compute_partition_quality(self, graph_engine) -> Dict[str, float]:
        """
        Compute partition quality metrics.
        """
        if not hasattr(graph_engine.data, 'part_id'):
            return {}
            
        part_ids = graph_engine.data.part_id
        edge_index = graph_engine.data.edge_index
        
        # Edge cut ratio
        src_parts = part_ids[edge_index[0]]
        dst_parts = part_ids[edge_index[1]]
        cut_edges = (src_parts != dst_parts).sum().item()
        total_edges = edge_index.size(1)
        cut_ratio = cut_edges / total_edges if total_edges > 0 else 0
        
        # Balance (std of partition sizes)
        unique, counts = torch.unique(part_ids, return_counts=True)
        balance_std = counts.float().std().item()
        balance_mean = counts.float().mean().item()
        balance_score = 1 - (balance_std / balance_mean) if balance_mean > 0 else 0
        
        return {
            'cut_ratio': cut_ratio,
            'balance_score': balance_score,
            'num_partitions': len(unique)
        }
