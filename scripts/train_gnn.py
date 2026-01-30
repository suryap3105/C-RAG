#!/usr/bin/env python
"""
Train Neural Subgraph Matcher
"""
import argparse
import logging
import torch
from pathlib import Path

from crag.graph.engine import GraphEngine
from crag.model.gnn import NeuralSubgraphMatcher
from crag.training.gnn_trainer import GNNTrainer, ContrastiveGraphDataset
from torch.utils.data import DataLoader, random_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_path', type=str, required=True, help='Path to saved GraphEngine')
    parser.add_argument('--queries_path', type=str, required=True, help='Path to queries JSON')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()
    
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on {device}")
    
    # Load graph
    logger.info(f"Loading graph from {args.graph_path}")
    graph_engine = GraphEngine()
    graph_engine.load(args.graph_path)
    
    # Load queries
    import json
    with open(args.queries_path, 'r') as f:
        queries = json.load(f)
    logger.info(f"Loaded {len(queries)} queries")
    
    # Create dataset
    dataset = ContrastiveGraphDataset(graph_engine, queries)
    
    # Train/val split
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Initialize model
    model = NeuralSubgraphMatcher(
        in_channels=768,
        hidden_channels=args.hidden_dim,
        out_channels=args.hidden_dim
    )
    
    # Train
    trainer = GNNTrainer(model, device=device)
    trainer.train(train_loader, val_loader, epochs=args.epochs, save_dir=args.output_dir)
    
    logger.info("Training complete!")


if __name__ == '__main__':
    main()
