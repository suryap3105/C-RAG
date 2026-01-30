#!/bin/bash
# Verify Robustness of C-RAG System

echo "=== C-RAG Robustness Verification ==="

# We need a Python driver script since I haven't implemented run_exp.py yet.
# Creating a temp test script
cat <<EOF > test_muddy.py
import torch
import os
from crag.graph.engine import GraphEngine
from crag.evaluation.robustness import GraphMuddier

# Create Dummy Graph
print("Creating Dummy Graph...")
if not os.path.exists("dummy_data"):
    os.makedirs("dummy_data")

with open("dummy_data/nodes.jsonl", "w") as f:
    for i in range(100):
        f.write(f'{{"id": {i}, "name": "Entity_{i}"}}\\n')

with open("dummy_data/edges.jsonl", "w") as f:
    # Circle graph
    for i in range(100):
        src = i
        dst = (i + 1) % 100
        f.write(f'{{"src": {src}, "dst": {dst}}}\\n')

ge = GraphEngine()
ge.load_graph("dummy_data/edges.jsonl", "dummy_data/nodes.jsonl")
print(f"Original: {ge.data.num_nodes} nodes, {ge.data.edge_index.size(1)} edges")

# Inject Noise
cfg = {"phantom_nodes_ratio": 0.5, "bridge_noise_ratio": 0.1}
muddier = GraphMuddier(ge, cfg)
mud_ge = muddier.apply()

print(f"Muddied: {mud_ge.data.num_nodes} nodes, {mud_ge.data.edge_index.size(1)} edges")

assert mud_ge.data.num_nodes == 150, "Expected 150 nodes"
# Edges: 100 original + 10% of 100 = 10? No, edge_index has 2 dims.
# Original edges = 100. Bridge ratio 0.1 => 10 noise edges. Total 110.
assert mud_ge.data.edge_index.size(1) == 110, f"Expected 110 edges, got {mud_ge.data.edge_index.size(1)}"
print("Robustness Test Passed!")
EOF

python test_muddy.py
rm test_muddy.py
rm -rf dummy_data
