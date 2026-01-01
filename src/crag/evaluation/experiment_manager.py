# src/crag/evaluation/experiment_manager.py
import pandas as pd
import time
import json
import os
import uuid
import numpy as np
from datetime import datetime
from typing import List, Dict, Callable

class ExperimentManager:
    """
    Research-Grade Experiment Manager.
    - Logs every query trace to metrics.jsonl
    - Computes aggregate stats
    - Maintains reproducibility
    """
    def __init__(self, output_dir: str = "runs"):
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
        self.output_dir = os.path.join(output_dir, self.run_id)
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.metrics_file = os.path.join(self.output_dir, "metrics.jsonl")
        self.results = []
        print(f"[ExpManager] Initialized run: {self.run_id}")
        print(f"[ExpManager] Output dir: {self.output_dir}")

    def run_experiment(self, system_name: str, dataset: List[Dict], model_func: Callable):
        """
        Runs a full pass over the dataset.
        """
        print(f"Running Experiment: {system_name}")
        scores = []
        latencies = []
        
        with open(self.metrics_file, 'a') as f:
            for item in dataset:
                start = time.time()
                result = {}
                try:
                    # model_func(query) -> Dict with 'answer', 'path', 'final_answer'
                    output = model_func(item['query'])
                    duration = time.time() - start
                    
                    # Basic Evaluation (Placeholder for EM/F1 func)
                    # In real usage, we'd use a robust metric function
                    gold = item.get('answers', [])
                    pred = output.get('answer', "")
                    
                    # Naive exact match for now
                    is_correct = any(g.lower() in pred.lower() for g in gold) if gold else False
                    score = 1.0 if is_correct else 0.0
                    
                    scores.append(score)
                    latencies.append(duration)
                    
                    # Granular Metric Log
                    # Extract termination reason from output if it's a ReasoningState-like object
                    termination_reason = "unknown"
                    if hasattr(output, 'termination_reason'):
                        termination_reason = output.termination_reason or "unknown"
                    elif isinstance(output, dict):
                        termination_reason = output.get('termination_reason', 'unknown')
                    
                    log_entry = {
                        "run_id": self.run_id,
                        "system": system_name,
                        "qid": item.get('id', 'unknown'),
                        "question": item['query'],
                        "gold_answers": gold,
                        "prediction": pred,
                        "is_correct": is_correct,
                        "latency": duration,
                        "hops": len(output.get('path', [])),
                        "nodes_expanded": sum(s.candidates_count for s in output.get('path', []) if hasattr(s, 'candidates_count')),
                        "termination_reason": termination_reason,
                        "timestamp": time.time()
                    }
                    
                    f.write(json.dumps(log_entry) + "\n")
                    
                except Exception as e:
                    print(f"Error on {item['query']}: {e}")
                    f.write(json.dumps({"error": str(e), "query": item['query']}) + "\n")
                    scores.append(0.0)
                    latencies.append(0.0)

        avg_score = np.mean(scores) if scores else 0.0
        avg_latency = np.mean(latencies) if latencies else 0.0
        
        summary = {
            "System": system_name,
            "Accuracy (EM)": f"{avg_score:.2f}",
            "Latency (s)": f"{avg_latency:.2f}",
            "Samples": len(dataset)
        }
        self.results.append(summary)
        
        # Save summary JSON
        with open(os.path.join(self.output_dir, "summary.json"), 'w') as f:
            json.dump(self.results, f, indent=2)

    def generate_latex_table(self, filename: str = None):
        """
        Outputs results as a NeurIPS-style LaTeX table.
        """
        if not filename:
             filename = os.path.join(self.output_dir, "results.tex")
             
        df = pd.DataFrame(self.results)
        latex = df.to_latex(index=False, caption="Performance Metrics", label="tab:results")
        
        with open(filename, 'w') as f:
            f.write(latex)
        print(f"LaTeX table saved to {filename}")
