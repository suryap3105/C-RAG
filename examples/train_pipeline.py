"""
Example: End-to-End Training Pipeline
"""
import torch
from torch.utils.data import DataLoader
from crag.graph.engine import GraphEngine
from crag.model.gnn import NeuralSubgraphMatcher
from crag.training.gnn_trainer import GNNTrainer, ContrastiveGraphDataset

# Load preprocessed graph
graph_engine = GraphEngine()
graph_engine.load("data/preprocessed_graph.pt")

# Load training queries
import json
with open("data/train_queries.json") as f:
    queries = json.load(f)

# Create dataset
dataset = ContrastiveGraphDataset(
    graph_engine, 
    queries, 
    num_negatives=5,
    num_hops=2
)

# Split
from torch.utils.data import random_split
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

# Dataloaders
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)

# Initialize model
model = NeuralSubgraphMatcher(
    in_channels=768,
    hidden_channels=256,
    out_channels=256,
    num_gat_layers=2,
    num_gin_layers=4
)

# Train
trainer = GNNTrainer(model, device='cuda')
history = trainer.train(
    train_loader, 
    val_loader, 
    epochs=100,
    save_dir="checkpoints"
)

print("Training complete!")
print(f"Best validation accuracy: {max([h['accuracy'] for h in history['val']]):.4f}")
