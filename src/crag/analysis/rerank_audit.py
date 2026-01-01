"""
Reranker Impact Audit Script
Analyzes pre/post reranking to determine if reranker helps or harms accuracy.
"""

import json
import os
from collections import defaultdict
from typing import List, Dict
import pandas as pd

class RerankerAuditor:
    """Audit reranker impact on candidate selection."""
    
    def __init__(self, runs_dir: str = "runs"):
        self.runs_dir = runs_dir
    
    def load_metrics(self, system_name: str) -> List[Dict]:
        """Load metrics for a specific system."""
        metrics = []
        if not os.path.exists(self.runs_dir):
            return metrics
            
        for run_dir in os.listdir(self.runs_dir):
            run_path = os.path.join(self.runs_dir, run_dir)
            if not os.path.isdir(run_path): continue
            
            metrics_file = os.path.join(run_path, "metrics.jsonl")
            
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            if data.get('system') == system_name:
                                metrics.append(data)
                        except: continue
        return metrics
    
    def compare_systems(self, with_rerank: str, without_rerank: str):
        """Compare system with/without reranker."""
        with_metrics = self.load_metrics(with_rerank)
        without_metrics = self.load_metrics(without_rerank)
        
        # Create qid-indexed dicts
        with_dict = {m['qid']: m for m in with_metrics if 'qid' in m}
        without_dict = {m['qid']: m for m in without_metrics if 'qid' in m}
        
        # Find common qids
        common_qids = set(with_dict.keys()) & set(without_dict.keys())
        
        results = []
        for qid in common_qids:
            with_result = with_dict[qid]
            without_result = without_dict[qid]
            
            results.append({
                'qid': qid,
                'question': with_result['question'],
                'with_rerank_correct': with_result['is_correct'],
                'without_rerank_correct': without_result['is_correct'],
                'with_rerank_latency': with_result['latency'],
                'without_rerank_latency': without_result['latency'],
                'with_rerank_nodes': with_result['nodes_expanded'],
                'without_rerank_nodes': without_result['nodes_expanded'],
            })
        
        return pd.DataFrame(results)
    
    def analyze_impact(self, df: pd.DataFrame):
        """Analyze reranker impact."""
        if df.empty:
            print("[WARN] No data to analyze")
            return df
            
        # Categorize impact
        df['improved'] = (~df['without_rerank_correct']) & df['with_rerank_correct']
        df['harmed'] = df['without_rerank_correct'] & (~df['with_rerank_correct'])
        df['unchanged'] = df['without_rerank_correct'] == df['with_rerank_correct']
        
        print("\n=== Reranker Impact Analysis ===")
        print(f"Total queries: {len(df)}")
        print(f"Improved (wrong -> correct): {df['improved'].sum()}")
        print(f"Harmed (correct -> wrong): {df['harmed'].sum()}")
        print(f"Unchanged: {df['unchanged'].sum()}")
        print(f"\nNet impact: {df['improved'].sum() - df['harmed'].sum()}")
        
        # Accuracy comparison
        with_acc = df['with_rerank_correct'].mean()
        without_acc = df['without_rerank_correct'].mean()
        print(f"\nWith reranker: {with_acc:.2%}")
        print(f"Without reranker: {without_acc:.2%}")
        print(f"Delta: {(with_acc - without_acc):.2%}")
        
        # Latency comparison
        avg_latency_with = df['with_rerank_latency'].mean()
        avg_latency_without = df['without_rerank_latency'].mean()
        print(f"\nAvg latency with reranker: {avg_latency_with:.2f}s")
        print(f"Avg latency without: {avg_latency_without:.2f}s")
        print(f"Overhead: {(avg_latency_with - avg_latency_without):.2f}s")
        
        return df
    
    def save_report(self, df: pd.DataFrame, output_path: str = "experiments/rerank_impact.csv"):
        """Save detailed report."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\nDetailed report saved: {output_path}")

def main():
    auditor = RerankerAuditor()
    
    # Compare systems
    # Using crag_no_rerank as baseline for full
    df = auditor.compare_systems(
        with_rerank="crag_full_v2",
        without_rerank="crag_no_rerank"
    )
    
    if len(df) == 0:
        print("[WARN] No common queries found between systems")
        print("[HINT] Run both systems on same dataset first")
        return
    
    df = auditor.analyze_impact(df)
    auditor.save_report(df)

if __name__ == "__main__":
    main()
