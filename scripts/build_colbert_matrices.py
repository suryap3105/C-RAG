#!/usr/bin/env python
"""
Build ColBERT Partition Matrices
"""
import argparse
import logging
from pathlib import Path

from crag.graph.engine import GraphEngine
from crag.routing.colbert import ColBERTPartitionRouter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--tokens_per_partition', type=int, default=32)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    
    # Load graph
    logger.info(f"Loading graph from {args.graph_path}")
    ge = GraphEngine()
    ge.load(args.graph_path)
    
    if not hasattr(ge.data, 'part_id') or ge.data.part_id is None:
        logger.error("Graph has no partitions! Run partitioning first.")
        return
        
    # Build matrices
    router = ColBERTPartitionRouter(device=args.device)
    router.build_partition_matrices(ge, num_tokens_per_partition=args.tokens_per_partition)
    
    # Save
    router.save(args.output_path)
    logger.info(f"Saved ColBERT matrices to {args.output_path}")


if __name__ == '__main__':
    main()
