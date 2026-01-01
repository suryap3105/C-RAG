"""
Termination Reason Analysis
Aggregates and analyzes why queries fail/succeed.
"""

import json
import os
from collections import Counter
from typing import List, Dict
import matplotlib.pyplot as plt

class TerminationAnalyzer:
    """Analyze termination reasons from experiment runs."""
    
    def __init__(self, runs_dir: str = "runs"):
        self.runs_dir = runs_dir
    
    def load_all_metrics(self) -> List[Dict]:
        """Load all metrics from all runs."""
        all_metrics = []
        for run_dir in os.listdir(self.runs_dir):
            run_path = os.path.join(self.runs_dir, run_dir)
            metrics_file = os.path.join(run_path, "metrics.jsonl")
            
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    for line in f:
                        data = json.loads(line)
                        all_metrics.append(data)
        return all_metrics
    
    def aggregate_by_system(self, metrics: List[Dict]):
        """Aggregate termination reasons by system."""
        by_system = {}
        
        for m in metrics:
            system = m.get('system', 'unknown')
            if system not in by_system:
                by_system[system] = {
                    'total': 0,
                    'correct': 0,
                    'incorrect': 0,
                    'termination_reasons': Counter(),
                    'failure_reasons': Counter()
                }
            
            by_system[system]['total'] += 1
            
            if m.get('is_correct'):
                by_system[system]['correct'] += 1
            else:
                by_system[system]['incorrect'] += 1
                reason = m.get('termination_reason', 'unknown')
                by_system[system]['failure_reasons'][reason] += 1
            
            reason = m.get('termination_reason', 'unknown')
            by_system[system]['termination_reasons'][reason] += 1
        
        return by_system
    
    def print_report(self, by_system: Dict):
        """Print termination analysis report."""
        print("\n" + "="*60)
        print("TERMINATION REASON ANALYSIS")
        print("="*60)
        
        for system, data in sorted(by_system.items()):
            print(f"\n[{system}]")
            print(f"  Total: {data['total']}")
            print(f"  Correct: {data['correct']} ({data['correct']/data['total']:.1%})")
            print(f"  Incorrect: {data['incorrect']} ({data['incorrect']/data['total']:.1%})")
            
            print(f"\n  All Termination Reasons:")
            for reason, count in data['termination_reasons'].most_common():
                print(f"    {reason}: {count} ({count/data['total']:.1%})")
            
            if data['incorrect'] > 0:
                print(f"\n  Failure-Only Termination Reasons:")
                for reason, count in data['failure_reasons'].most_common():
                    print(f"    {reason}: {count} ({count/data['incorrect']:.1%})")
    
    def plot_termination_reasons(self, by_system: Dict, output_path: str = "experiments/termination_reasons.png"):
        """Plot termination reasons."""
        fig, axes = plt.subplots(1, len(by_system), figsize=(5*len(by_system), 4))
        
        if len(by_system) == 1:
            axes = [axes]
        
        for idx, (system, data) in enumerate(sorted(by_system.items())):
            ax = axes[idx]
            
            reasons = data['termination_reasons']
            labels = list(reasons.keys())
            counts = list(reasons.values())
            
            ax.bar(labels, counts)
            ax.set_title(f"{system}\n(n={data['total']})")
            ax.set_xlabel("Termination Reason")
            ax.set_ylabel("Count")
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved: {output_path}")

def main():
    analyzer = TerminationAnalyzer()
    metrics = analyzer.load_all_metrics()
    
    if not metrics:
        print("[WARN] No metrics found in runs/")
        return
    
    by_system = analyzer.aggregate_by_system(metrics)
    analyzer.print_report(by_system)
    analyzer.plot_termination_reasons(by_system)
    
    print("\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)
    
    for system, data in by_system.items():
        if data['incorrect'] > 0:
            top_failure = data['failure_reasons'].most_common(1)[0]
            print(f"[{system}] Top failure reason: {top_failure[0]} ({top_failure[1]}/{data['incorrect']})")

if __name__ == "__main__":
    main()
