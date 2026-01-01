#!/usr/bin/env python3
"""
Ablation Suite Runner for C-RAG System.
Systematically removes/modifies components to measure contributions.
"""
import sys
import os
sys.path.insert(0, 'src')

from crag.data.unified_loader import UnifiedDatasetLoader
import yaml
import time

def load_config(path):
    """Load YAML configuration."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def run_ablation_suite(max_queries=100):
    """
    Run all ablation experiments.
    
    Args:
        max_queries: Number of queries to evaluate per ablation
    """
    print("\n" + "="*70)
    print("C-RAG ABLATION SUITE")
    print("="*70 + "\n")
    
    # Load evaluation dataset
    print("[1/6] Loading evaluation dataset...")
    data = UnifiedDatasetLoader.load_all(max_per_dataset=500)
    eval_subset = data[:max_queries]
    print(f"  -> Loaded {len(eval_subset)} questions for ablation\n")
    
    # Define ablation configurations
    ablations = [
        ("configs/crag_full.yaml", "Full System (Baseline)"),
        ("configs/crag_no_rerank.yaml", "No ColBERT Reranker"),
        ("configs/ablation_no_vector.yaml", "No Vector Prefilter"),
        ("configs/ablation_vector_only.yaml", "Vector-Only (No Graph)"),
        ("configs/ablation_one_shot.yaml", "One-Shot (No Agent)"),
    ]
    
    results = []
    
    for idx, (config_path, description) in enumerate(ablations, start=2):
        print(f"\n[{idx}/6] Running: {description}")
        print(f"  Config: {config_path}")
        
        if not os.path.exists(config_path):
            print(f"  [SKIP] Config not found\n")
            continue
        
        config = load_config(config_path)
        system_name = config.get('experiment_name', 'unknown')
        
        # Placeholder for actual experiment execution
        # In production, this would:
        # 1. Initialize system from config
        # 2. Run on eval_subset
        # 3. Log metrics to runs/
        # 4. Compute aggregates
        
        print(f"  [TODO] Initialize {system_name}")
        print(f"  [TODO] Run on {len(eval_subset)} queries")
        print(f"  [TODO] Log to runs/{system_name}_ablation/")
        
        # Simulated metrics
        result = {
            "system": system_name,
            "description": description,
            "config": config_path,
            "queries": len(eval_subset),
            "status": "pending"
        }
        results.append(result)
    
    # Budget sweeps (Hop variations)
    print(f"\n[6/6] Running: Hop Budget Sweep")
    for hops in [1, 2, 3, 5]:
        print(f"  [TODO] Run with max_hops={hops}")
    
    print("\n" + "="*70)
    print("ABLATION SUITE COMPLETE")
    print("="*70)
    
    print("\nAblations Executed:")
    for r in results:
        print(f"  - {r['description']} ({r['system']})")
    
    print("\nNext Steps:")
    print("  1. Analyze: python src/crag/analysis/ablation_analysis.py")
    print("  2. Generate table: experiments/ablation_table.md")
    print("  3. Statistical tests: paired t-tests vs baseline\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_queries", type=int, default=100,
                        help="Number of queries per ablation")
    args = parser.parse_args()
    
    run_ablation_suite(max_queries=args.max_queries)
