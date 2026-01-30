#!/usr/bin/env python
"""
Preprocess and index knowledge graph
"""
import argparse
import logging
from pathlib import Path

from crag.graph.engine import GraphEngine
from crag.graph.partitioning import GraphPartitioner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes', type=str, required=True, help='Path to nodes.jsonl')
    parser.add_argument('--edges', type=str, required=True, help='Path to edges.jsonl')
    parser.add_argument('--output', type=str, required=True, help='Output path for processed graph')
    parser.add_argument('--partition_method', type=str, default='metis', choices=['metis', 'leiden', 'spectral'])
    parser.add_argument('--n_partitions', type=int, default=10)
    args = parser.parse_args()
    
    # Load graph
    logger.info("Loading graph...")
    ge = GraphEngine()
    ge.load_graph(args.edges, args.nodes)
    
    # Compute embeddings
    logger.info("Computing node embeddings...")
    ge.compute_embeddings()
    
    # Partition
    logger.info(f"Partitioning with {args.partition_method}...")
    partitioner = GraphPartitioner(method=args.partition_method, n_partitions=args.n_partitions)
    partitioner.partition(ge)
    
    # Quality
    quality = partitioner.compute_partition_quality(ge)
    logger.info(f"Partition quality: {quality}")
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ge.save(args.output)
    
    logger.info(f"Saved processed graph to {args.output}")


if __name__ == '__main__':
    main()
