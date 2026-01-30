import torch
import os
import shutil
import logging
from crag.graph.engine import GraphEngine
from crag.evaluation.robustness import GraphMuddier

logging.basicConfig(level=logging.INFO)

# Setup Dummy Data
if not os.path.exists("dummy_data"):
    os.makedirs("dummy_data")

try:
    with open("dummy_data/nodes.jsonl", "w") as f:
        for i in range(100):
            f.write(f'{{"id": {i}, "name": "Entity_{i}"}}\\n')

    with open("dummy_data/edges.jsonl", "w") as f:
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
    assert mud_ge.data.edge_index.size(1) == 110, f"Expected 110 edges, got {mud_ge.data.edge_index.size(1)}"
    print("SUCCESS: Robustness Test Passed!")

finally:
    if os.path.exists("dummy_data"):
        shutil.rmtree("dummy_data")
