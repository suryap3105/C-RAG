
import os
import json
import glob
import argparse
from typing import List, Dict

def extract_interesting_cases(runs_dir: str):
    """
    Finds:
    1. A complex success case (multi-hop).
    2. A failure case.
    """
    # Find full C-RAG runs
    files = glob.glob(os.path.join(runs_dir, "*", "metrics.jsonl"))
    
    full_crag_entries = []
    
    for fpath in files:
        with open(fpath, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if "full" in entry['system'].lower():
                        full_crag_entries.append(entry)
                except:
                    continue
                    
    if not full_crag_entries:
        print("No full C-RAG runs found.")
        return

    # 1. Success Case (Longest Path)
    successes = [e for e in full_crag_entries if e['is_correct']]
    if successes:
        # Sort by path length desc
        successes.sort(key=lambda x: x['path_length'], reverse=True)
        best_case = successes[0]
        
        print("\n=== SUCCESS CASE (Paper Example) ===")
        print_case(best_case)
    else:
        print("No successful cases found for C-RAG.")

    # 2. Failure Case
    failures = [e for e in full_crag_entries if not e['is_correct']]
    if failures:
        # Just take the first one
        fail_case = failures[0]
        
        print("\n=== FAILURE CASE (Error Analysis) ===")
        print_case(fail_case)
    else:
        print("No failure cases found for C-RAG.")

def print_case(entry: Dict):
    print(f"Query: {entry['query']}")
    print(f"System: {entry['system']}")
    print(f"Gold: {entry['gold_answers']}")
    print(f"Prediction: {entry['prediction']}")
    print(f"Latency: {entry['latency']:.4f}s")
    print(f"Steps: {entry['path_length']}")
    print("-" * 30)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", type=str, default="runs")
    args = parser.parse_args()
    
    extract_interesting_cases(args.runs_dir)

if __name__ == "__main__":
    main()
