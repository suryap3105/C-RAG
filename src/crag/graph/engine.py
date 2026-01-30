"""
C-RAG V3 Production Graph Engine
Complete PyG-based Graph Management with Subgraph Extraction
"""
import torch
import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, subgraph

logger = logging.getLogger(__name__)

class GraphEngine:
    """
    Production Graph Engine for C-RAG V3.
    Manages Knowledge Graph as PyG Data with partitioning and subgraph extraction.
    """
    def __init__(self):
        self.data = Data()
        self.node_text_map: Dict[int, Dict] = {}
        self.edge_attr_map: Dict[Tuple[int, int], Dict] = {}
        self.encoder = None
        
    def _get_encoder(self):
        if self.encoder is None:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer('intfloat/e5-base-v2')
        return self.encoder
        
    def load_graph(self, edges_path: str, nodes_path: str):
        """Load graph from JSONL files."""
        logger.info(f"Loading graph from {edges_path}, {nodes_path}")
        
        # Load Nodes
        self.node_text_map = {}
        node_list = []
        with open(nodes_path, 'r', encoding='utf-8') as f:
            for line in f:
                node = json.loads(line.strip())
                node_list.append(node)
                self.node_text_map[node['id']] = node
                
        num_nodes = len(node_list)
        
        # Load Edges
        src_list, dst_list = [], []
        with open(edges_path, 'r', encoding='utf-8') as f:
            for line in f:
                edge = json.loads(line.strip())
                src_list.append(edge['src'])
                dst_list.append(edge['dst'])
                # Store edge attributes
                self.edge_attr_map[(edge['src'], edge['dst'])] = edge.get('attr', {})
                
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        
        self.data = Data(edge_index=edge_index, num_nodes=num_nodes)
        logger.info(f"Graph loaded: {num_nodes} nodes, {len(src_list)} edges")
        
    def compute_embeddings(self, batch_size: int = 32):
        """Compute node embeddings using text."""
        encoder = self._get_encoder()
        
        texts = []
        for i in range(self.data.num_nodes):
            node_info = self.node_text_map.get(i, {})
            text = node_info.get('text', node_info.get('name', f'Entity_{i}'))
            texts.append(f"passage: {text}")
            
        logger.info(f"Computing embeddings for {len(texts)} nodes...")
        embeddings = encoder.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
        self.data.x = torch.tensor(embeddings, dtype=torch.float32)
        logger.info(f"Embeddings computed: {self.data.x.shape}")
        
    def extract_subgraph(self, center_nodes: List[int], num_hops: int = 2, max_nodes: int = 100) -> Data:
        """
        Extract k-hop subgraph around center nodes.
        Returns a new Data object with remapped indices.
        """
        center_tensor = torch.tensor(center_nodes, dtype=torch.long)
        
        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx=center_tensor,
            num_hops=num_hops,
            edge_index=self.data.edge_index,
            relabel_nodes=True,
            num_nodes=self.data.num_nodes
        )
        
        # Limit subgraph size
        if len(subset) > max_nodes:
            subset = subset[:max_nodes]
            mask = (edge_index[0] < max_nodes) & (edge_index[1] < max_nodes)
            edge_index = edge_index[:, mask]
            
        # Build subgraph Data
        sub_data = Data(
            edge_index=edge_index,
            num_nodes=len(subset)
        )
        
        # Copy node features if available
        if self.data.x is not None:
            sub_data.x = self.data.x[subset]
        else:
            sub_data.x = torch.randn(len(subset), 768)
            
        # Store original node IDs for later lookup
        sub_data.original_ids = subset
        
        return sub_data
        
    def get_partition_subgraph(self, partition_id: int) -> Data:
        """Get all nodes in a partition as a subgraph."""
        if not hasattr(self.data, 'part_id') or self.data.part_id is None:
            logger.warning("No partition info. Returning full graph.")
            return self.data
            
        mask = (self.data.part_id == partition_id)
        node_indices = mask.nonzero(as_tuple=True)[0]
        
        # Extract subgraph
        sub_edge_index, _ = subgraph(node_indices, self.data.edge_index, relabel_nodes=True)
        
        sub_data = Data(
            edge_index=sub_edge_index,
            num_nodes=len(node_indices)
        )
        
        if self.data.x is not None:
            sub_data.x = self.data.x[node_indices]
        else:
            sub_data.x = torch.randn(len(node_indices), 768)
            
        sub_data.original_ids = node_indices
        return sub_data
        
    def get_neighbors(self, node_id: int) -> torch.Tensor:
        """Get neighbor node IDs."""
        edge_index = self.data.edge_index
        mask = edge_index[0] == node_id
        return edge_index[1, mask]
        
    def save(self, path: str):
        """Save graph state."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'data': self.data,
            'node_text_map': self.node_text_map,
            'edge_attr_map': self.edge_attr_map
        }
        torch.save(state, path)
        logger.info(f"Saved GraphEngine to {path}")
        
    def load(self, path: str):
        """Load graph state."""
        state = torch.load(path, map_location='cpu')
        self.data = state['data']
        self.node_text_map = state.get('node_text_map', {})
        self.edge_attr_map = state.get('edge_attr_map', {})
        logger.info(f"Loaded GraphEngine from {path}: {self.data.num_nodes} nodes")
