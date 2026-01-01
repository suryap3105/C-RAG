#!/usr/bin/env python3
"""
Ablation Analysis Script.
Compares ablation results against full system baseline.
"""
import json
import os
import glob
import pandas as pd
from pathlib import Path

def load_ablation_results(runs_dir="runs"):
    """Load all ablation experiment results."""
    files = glob.glob(os.path.join(runs_dir, "*ablation*/metrics.jsonl"))
    files += glob.glob(os.path.join(runs_dir, "crag_*/metrics.jsonl"))
    
    if not files:
        print(f"[WARN] No ablation results found in {runs_dir}")
        return pd.DataFrame()
    
    all_data = []
    for fpath in files:
        with open(fpath, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    all_data.append(entry)
                except json.JSONDecodeError:
                    continue
    
    return pd.DataFrame(all_data)

def compute_ablation_table(df):
    """Generate ablation comparison table."""
    if df.empty:
        print("[ERROR] No data to analyze")
        return
    
    # Aggregate by system
    stats = df.groupby('system').agg({
        'is_correct': ['mean', 'count'],
        'latency': 'mean',
        'hops': 'mean',
        'nodes_expanded': 'mean'
    }).reset_index()
    
    stats.columns = ['System', 'Accuracy', 'N', 'Latency_s', 'Avg_Hops', 'Avg_Nodes']
    
    # Find baseline (full system)
    baseline_idx = stats[stats['System'].str.contains('full', case=False)].index
    if len(baseline_idx) == 0:
        print("[WARN] No baseline (full) system found")
        baseline_acc = None
    else:
        baseline_acc = stats.loc[baseline_idx[0], 'Accuracy']
    
    # Compute deltas
    if baseline_acc is not None:
        stats['Delta_Acc'] = stats['Accuracy'] - baseline_acc
    else:
        stats['Delta_Acc'] = 0.0
    
    return stats

def generate_report(stats, output_path="experiments/ablation_table.md"):
    """Generate markdown report."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    report = "# Ablation Study Results\n\n"
    report += "## Component Contribution Analysis\n\n"
    report += stats.to_markdown(index=False, floatfmt=".3f")
    report += "\n\n## Key Findings\n\n"
    
    # Auto-generate insights
    if not stats.empty:
        best_system = stats.loc[stats['Accuracy'].idxmax()]
        fastest_system = stats.loc[stats['Latency_s'].idxmin()]
        
        report += f"- **Best Accuracy**: {best_system['System']} ({best_system['Accuracy']:.1%})\n"
        report += f"- **Fastest**: {fastest_system['System']} ({fastest_system['Latency_s']:.2f}s)\n"
        
        # Component impacts
        if baseline_acc := stats[stats['System'].str.contains('full', case=False)]['Accuracy'].values:
            baseline_acc = baseline_acc[0]
            report += f"\n### Component Impacts (vs Full System: {baseline_acc:.1%})\n\n"
            
            for _, row in stats.iterrows():
                if 'full' in row['System'].lower():
                    continue
                delta = row['Delta_Acc']
                impact = "Critical" if abs(delta) > 0.1 else "Moderate" if abs(delta) > 0.05 else "Minor"
                report += f"- **{row['System']}**: {delta:+.1%} ({impact})\n"
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"[OK] Report saved: {output_path}")

def main():
    print("\n" + "="*60)
    print("ABLATION ANALYSIS")
    print("="*60 + "\n")
    
    df = load_ablation_results()
    if df.empty:
        return
    
    print(f"Loaded {len(df)} experiment records\n")
    
    stats = compute_ablation_table(df)
    print("\n" + stats.to_string(index=False) + "\n")
    
    generate_report(stats)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
