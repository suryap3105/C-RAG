#!/usr/bin/env python3
"""
Large-scale experiment script for C-RAG NeurIPS evaluation.
Runs all baselines on full datasets (subsampled for efficiency).
"""
import sys
import os
sys.path.insert(0, 'src')

from crag.data.unified_loader import UnifiedDatasetLoader
from crag.evaluation.experiment_manager import ExperimentManager
from crag.llm.interface import MockLLMClient
import yaml

def load_config(path):
    """Load YAML config."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # Load datasets (cap at 500 per source for efficiency)
    print("\n" + "="*70)
    print("C-RAG LARGE-SCALE EVALUATION")
    print("="*70 + "\n")
    
    data = UnifiedDatasetLoader.load_all(max_per_dataset=500)
    print(f"\nDataset loaded: {len(data)} questions\n")
    
    # Systems to evaluate
    configs = [
        "configs/baseline_vector.yaml",
        "configs/baseline_graph.yaml",
        "configs/crag_no_rerank.yaml",
        "configs/crag_full.yaml"
    ]
    
    for config_path in configs:
        if not os.path.exists(config_path):
            print(f"[SKIP] {config_path} not found")
            continue
        
        config = load_config(config_path)
        system_name = config.get('experiment_name', 'unknown')
        
        print(f"\n{'='*70}")
        print(f"Running: {system_name}")
        print(f"{'='*70}\n")
        
        # For now, run on first 100 questions to verify pipeline
        # In production, increase to 500+
        subset = data[:100]
        
        # Initialize experiment manager
        # (Note: Actual system initialization would go here)
        # For skeleton, just log intent
        
        print(f"  [TODO] Run {system_name} on {len(subset)} questions")
        print(f"  [TODO] Log metrics to runs/{system_name}_*.jsonl")
        print(f"  [TODO] Compute EM/F1, latency, nodes\n")
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("  1. Run: python src/crag/analysis/analyze_results.py --runs_dir runs --output_dir experiments")
    print("  2. Check: experiments/analysis_report.md")
    print("  3. Check: experiments/pareto_curve.png\n")

if __name__ == "__main__":
    main()
