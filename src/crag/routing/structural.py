"""
C-RAG V3 Structural Alignment
Computes structural similarity between Query Graphs and Knowledge Graph Partitions.
"""
import torch
import logging
from typing import Dict, Any, List
from torch_geometric.data import Data

logger = logging.getLogger(__name__)

class StructuralAligner:
    """
    Evaluates how well a Query Graph fits into a Graph Partition.
    Uses structural fingerprints (Type histograms, Relation sets).
    """
    def __init__(self, use_embeddings: bool = True):
        self.use_embeddings = use_embeddings
        
    def compute_fingerprint(self, data: Data) -> Dict[str, Any]:
        """
        Extract structural fingerprint from a subgraph/partition.
        """
        fingerprint = {
            'node_types': set(),
            'edge_types': set(),
            'num_nodes': data.num_nodes,
        }
        
        # Check metadata from QueryGraph extraction or GraphEngine loading
        if hasattr(data, 'node_types'):
            fingerprint['node_types'].update(data.node_types)
            
        if hasattr(data, 'edge_types'):
            fingerprint['edge_types'].update(data.edge_types)
            
        # If not explicit, we can't guess types without schema, 
        # but we can rely on what's available.
        return fingerprint
        
    def score(self, query_graph: Data, partition_fp: Dict[str, Any]) -> float:
        """
        Compute alignment score between query graph and partition fingerprint.
        Score in [0, 1].
        """
        # Validate inputs
        if query_graph is None or partition_fp is None:
            logger.warning("Null input to structural aligner. Returning neutral score.")
            return 0.5
            
        if not hasattr(query_graph, 'num_nodes') or query_graph.num_nodes == 0:
            logger.warning("Invalid query graph. Returning neutral score.")
            return 0.5
        
        try:
            q_fp = self.compute_fingerprint(query_graph)
            
            if not q_fp['node_types'] and not q_fp['edge_types']:
                return 0.5  # Neutral if no structural info
                
            score = 0.0
            components = 0
            
            # 1. Node Type Overlap
            if q_fp['node_types']:
                q_types = q_fp['node_types']
                p_types = partition_fp.get('node_types', set())
                
                # Handle empty partition types
                if not p_types:
                    logger.debug("Partition has no node type information.")
                else:
                    # Intersection / Query Size (Recall)
                    overlap = len(q_types.intersection(p_types))
                    type_score = overlap / len(q_types)
                    
                    score += type_score
                    components += 1
                
            # 2. Edge Type Overlap
            if q_fp['edge_types']:
                q_rels = q_fp['edge_types']
                p_rels = partition_fp.get('edge_types', set())
                
                if not p_rels:
                    logger.debug("Partition has no edge type information.")
                else:
                    overlap = len(q_rels.intersection(p_rels))
                    rel_score = overlap / len(q_rels)
                    
                    score += rel_score
                    components += 1
                    
            # If no components matched, return neutral
            if components == 0:
                return 0.5
                
            return score / components
            
        except Exception as e:
            logger.error(f"Error in structural scoring: {e}")
            return 0.5  # Failsafe: neutral score

    def batch_score(self, query_graph: Data, partition_fps: List[Dict[str, Any]]) -> List[float]:
        """Score multiple partitions with validation."""
        if not partition_fps:
            logger.warning("Empty partition list provided to batch_score.")
            return []
            
        try:
            return [self.score(query_graph, fp) for fp in partition_fps]
        except Exception as e:
            logger.error(f"Batch scoring failed: {e}")
            # Failsafe: return neutral scores
            return [0.5] * len(partition_fps)
