"""
C-RAG V3 Production Robustness Module
Systematic Noise Injection for Adversarial Testing
"""
import torch
import numpy as np
import copy
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from ..graph.engine import GraphEngine

logger = logging.getLogger(__name__)


@dataclass
class NoiseConfig:
    """Configuration for noise injection."""
    phantom_nodes_ratio: float = 0.0  # Ratio of phantom nodes to add
    bridge_noise_ratio: float = 0.0   # Ratio of random edges to add
    attribute_noise_std: float = 0.0  # Std for Gaussian noise on embeddings
    edge_drop_ratio: float = 0.0      # Ratio of edges to drop
    label_flip_ratio: float = 0.0     # Ratio of node labels to flip
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'NoiseConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__annotations__})


class GraphMuddier:
    """
    Production Robustness Testing Module.
    Injects systematic noise to test retrieval system resilience.
    """
    def __init__(self, graph_engine: GraphEngine, noise_config: Dict):
        self.original_ge = graph_engine
        self.cfg = NoiseConfig.from_dict(noise_config)
        self.stats: Dict[str, int] = {}
        
    def apply(self) -> GraphEngine:
        """
        Apply all noise transformations and return new GraphEngine.
        """
        logger.info(f"[GraphMuddier] Applying noise: {self.cfg}")
        
        if self.original_ge.data is None or self.original_ge.data.num_nodes == 0:
            logger.warning("Empty graph, returning as-is")
            return self.original_ge
            
        # Deep copy
        new_ge = GraphEngine()
        new_ge.data = self._clone_data(self.original_ge.data)
        new_ge.node_text_map = copy.deepcopy(self.original_ge.node_text_map)
        new_ge.edge_attr_map = copy.deepcopy(self.original_ge.edge_attr_map)
        
        # Apply transformations
        if self.cfg.phantom_nodes_ratio > 0:
            new_ge.data = self._inject_phantoms(new_ge.data)
            
        if self.cfg.bridge_noise_ratio > 0:
            new_ge.data = self._inject_bridges(new_ge.data)
            
        if self.cfg.attribute_noise_std > 0:
            new_ge.data = self._perturb_attributes(new_ge.data)
            
        if self.cfg.edge_drop_ratio > 0:
            new_ge.data = self._drop_edges(new_ge.data)
            
        # Update text map for phantom nodes
        self._extend_text_map(new_ge)
        
        # Log stats
        logger.info(f"[GraphMuddier] Stats: {self.stats}")
        
        return new_ge
        
    def _clone_data(self, data) -> 'torch_geometric.data.Data':
        """Deep clone PyG Data object."""
        from torch_geometric.data import Data
        
        new_data = Data()
        for key, value in data:
            if torch.is_tensor(value):
                setattr(new_data, key, value.clone())
            else:
                setattr(new_data, key, copy.deepcopy(value))
        return new_data
        
    def _inject_phantoms(self, data) -> 'torch_geometric.data.Data':
        """
        Inject phantom nodes with realistic but irrelevant embeddings.
        """
        num_original = data.num_nodes
        num_phantoms = int(num_original * self.cfg.phantom_nodes_ratio)
        
        if num_phantoms == 0:
            return data
            
        logger.info(f"[GraphMuddier] Injecting {num_phantoms} phantom nodes")
        self.stats['phantom_nodes'] = num_phantoms
        
        if data.x is not None:
            # Create phantom embeddings similar to real distribution
            mean = data.x.mean(dim=0)
            std = data.x.std(dim=0) * 1.5  # Slightly higher variance
            
            phantom_x = torch.normal(
                mean=mean.unsqueeze(0).expand(num_phantoms, -1),
                std=std.unsqueeze(0).expand(num_phantoms, -1)
            )
            
            # Normalize to match original distribution
            phantom_x = phantom_x / (phantom_x.norm(dim=1, keepdim=True) + 1e-8)
            phantom_x = phantom_x * data.x.norm(dim=1).mean()
            
            data.x = torch.cat([data.x, phantom_x], dim=0)
            
        # Update partition IDs
        if hasattr(data, 'part_id') and data.part_id is not None:
            num_parts = data.part_id.max().item() + 1
            phantom_parts = torch.randint(0, num_parts, (num_phantoms,))
            data.part_id = torch.cat([data.part_id, phantom_parts])
            
        # Connect phantoms sparsely (to make them harder to detect)
        num_phantom_edges = num_phantoms * 2
        src_phantoms = torch.randint(num_original, num_original + num_phantoms, (num_phantom_edges,))
        dst_phantoms = torch.randint(0, num_original + num_phantoms, (num_phantom_edges,))
        phantom_edges = torch.stack([src_phantoms, dst_phantoms], dim=0)
        
        data.edge_index = torch.cat([data.edge_index, phantom_edges], dim=1)
        data.num_nodes = num_original + num_phantoms
        
        return data
        
    def _inject_bridges(self, data) -> 'torch_geometric.data.Data':
        """
        Inject random bridge edges between unrelated nodes.
        """
        num_edges = data.edge_index.size(1)
        num_noise = int(num_edges * self.cfg.bridge_noise_ratio)
        
        if num_noise == 0:
            return data
            
        logger.info(f"[GraphMuddier] Injecting {num_noise} bridge edges")
        self.stats['bridge_edges'] = num_noise
        
        num_nodes = data.num_nodes
        
        # Generate random edges
        src = torch.randint(0, num_nodes, (num_noise,))
        dst = torch.randint(0, num_nodes, (num_noise,))
        
        # Filter self-loops
        mask = src != dst
        src, dst = src[mask], dst[mask]
        
        noise_edges = torch.stack([src, dst], dim=0)
        data.edge_index = torch.cat([data.edge_index, noise_edges], dim=1)
        
        return data
        
    def _perturb_attributes(self, data) -> 'torch_geometric.data.Data':
        """
        Add Gaussian noise to node embeddings.
        """
        if data.x is None:
            return data
            
        logger.info(f"[GraphMuddier] Perturbing attributes with std={self.cfg.attribute_noise_std}")
        
        noise = torch.randn_like(data.x) * self.cfg.attribute_noise_std
        data.x = data.x + noise
        
        # Re-normalize
        data.x = data.x / (data.x.norm(dim=1, keepdim=True) + 1e-8)
        
        self.stats['perturbed_nodes'] = data.num_nodes
        
        return data
        
    def _drop_edges(self, data) -> 'torch_geometric.data.Data':
        """
        Randomly drop edges.
        """
        num_edges = data.edge_index.size(1)
        num_drop = int(num_edges * self.cfg.edge_drop_ratio)
        
        if num_drop == 0:
            return data
            
        logger.info(f"[GraphMuddier] Dropping {num_drop} edges")
        self.stats['dropped_edges'] = num_drop
        
        # Random mask
        keep_mask = torch.ones(num_edges, dtype=torch.bool)
        drop_indices = torch.randperm(num_edges)[:num_drop]
        keep_mask[drop_indices] = False
        
        data.edge_index = data.edge_index[:, keep_mask]
        
        return data
        
    def _extend_text_map(self, ge: GraphEngine):
        """
        Add text entries for phantom nodes.
        """
        if not ge.node_text_map:
            return
            
        current_max = max(ge.node_text_map.keys()) if ge.node_text_map else -1
        total_nodes = ge.data.num_nodes
        
        for i in range(current_max + 1, total_nodes):
            ge.node_text_map[i] = {
                'id': i,
                'name': f'Phantom_Entity_{i}',
                'text': f'A phantom entity with no real semantic content.',
                'is_phantom': True
            }
            
    def get_noise_statistics(self) -> Dict[str, Any]:
        """Get statistics about applied noise."""
        return {
            'config': {
                'phantom_nodes_ratio': self.cfg.phantom_nodes_ratio,
                'bridge_noise_ratio': self.cfg.bridge_noise_ratio,
                'attribute_noise_std': self.cfg.attribute_noise_std,
                'edge_drop_ratio': self.cfg.edge_drop_ratio
            },
            'applied': self.stats
        }


def create_degradation_curve(retriever, dataset: List[Dict], 
                            graph_engine: GraphEngine,
                            noise_levels: List[float] = None) -> Dict[float, float]:
    """
    Create degradation curve showing performance vs noise level.
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]
        
    from .experiment_manager import ExperimentManager
    
    exp_mgr = ExperimentManager()
    results = exp_mgr.run_robustness_experiment(
        retriever, dataset, graph_engine, GraphMuddier, noise_levels
    )
    
    curve = {noise: metrics.mrr for noise, metrics in results.items()}
    return curve
