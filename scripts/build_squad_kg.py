#!/usr/bin/env python3
"""
Script to build SQuAD Knowledge Graph.
"""
import sys
import os
sys.path.insert(0, 'src')

from crag.data.unified_loader import UnifiedDatasetLoader
from crag.graph.builder import SimpleGraphBuilder

def main():
    print("Building Knowledge Graph from SQuAD...")
    
    # Load Data
    squad_path = "data/squad_train_v2.json"
    if not os.path.exists(squad_path):
        print("SQuAD data not found.")
        return

    # Load first 2000 contexts for demo graph
    # 2000 is enough to get a dense graph without exploding memory for this demo
    data = UnifiedDatasetLoader.load_squad(squad_path)[:2000]
    print(f"Loaded {len(data)} documents")
    
    # Build Graph
    builder = SimpleGraphBuilder(min_freq=2)
    builder.process_documents(data)
    
    # Save
    os.makedirs("data/graphs", exist_ok=True)
    builder.save_graph("data/graphs/squad_kg.json")

if __name__ == "__main__":
    main()
