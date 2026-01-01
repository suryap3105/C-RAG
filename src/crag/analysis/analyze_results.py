
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import glob
import argparse
from typing import List, Dict

def load_metrics(runs_dir: str) -> pd.DataFrame:
    """
    Loads all metrics.jsonl files from runs directory into a DataFrame.
    """
    all_runs = []
    
    # improved glob pattern to find all metrics.jsonl files recursively or in immediate subdirs
    files = glob.glob(os.path.join(runs_dir, "*", "metrics.jsonl"))
    
    if not files:
        print(f"[WARN] No metrics.jsonl files found in {runs_dir}")
        return pd.DataFrame()

    print(f"Found {len(files)} run files.")
    
    for fpath in files:
        with open(fpath, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    all_runs.append(entry)
                except json.JSONDecodeError:
                    continue
                    
    df = pd.DataFrame(all_runs)
    return df

def generate_pareto_plot(df: pd.DataFrame, output_path: str):
    """
    Generates Average Accuracy vs Average Latency plot.
    """
    if df.empty:
        print("[WARN] DataFrame empty, skipping plots.")
        return

    # Aggregate by system
    stats = df.groupby('system').agg({
        'is_correct': 'mean',
        'latency': 'mean',
        'hops': 'mean', # Proxy for "steps" or "compute"
        'nodes_expanded': 'mean' 
    }).reset_index()
    
    print("\n=== Aggregate Stats ===")
    print(stats)
    
    plt.figure(figsize=(10, 6))
    
    for _, row in stats.iterrows():
        plt.scatter(row['latency'], row['is_correct'], s=100, label=row['system'])
        plt.text(row['latency'], row['is_correct'], f"  {row['system']}", fontsize=9)
        
    plt.title('Pareto Frontier: Accuracy vs Latency (Compute)')
    plt.xlabel('Average Latency (s)')
    plt.ylabel('Accuracy (Exatch Match)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    
    plt.savefig(output_path)
    print(f"Pareto plot saved to {output_path}")

def generate_report(df: pd.DataFrame, output_path: str):
    """
    Generates a markdown analysis report.
    """
    if df.empty:
        return
        
    stats = df.groupby('system').agg({
        'is_correct': ['mean', 'count'],
        'latency': 'mean',
        'nodes_expanded': 'mean'
    }).reset_index()
    
    # Flatten columns
    stats.columns = ['System', 'Accuracy', 'Samples', 'Avg_Latency', 'Avg_Nodes_Expanded']
    
    report = "# Experiment Analysis Report\n\n"
    report += "## Aggregate Performance\n"
    report += stats.to_markdown(index=False)
    
    report += "\n\n## Failure Modes\n"
    failures = df[df['is_correct'] == False]
    report += f"Total Failures: {len(failures)}\n"
    
    # Group by system
    metrics = failures.groupby('system').size().reset_index(name='failures')
    report += metrics.to_markdown(index=False)
    
    with open(output_path, 'w') as f:
        f.write(report)
    print(f"Report saved to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", type=str, default="runs")
    parser.add_argument("--output_dir", type=str, default="experiments")
    args = parser.parse_args()
    
    df = load_metrics(args.runs_dir)
    
    generate_pareto_plot(df, os.path.join(args.output_dir, "pareto_curve.png"))
    generate_report(df, os.path.join(args.output_dir, "analysis_report.md"))

if __name__ == "__main__":
    main()
