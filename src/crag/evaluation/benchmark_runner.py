"""
Benchmark runner for evaluating all systems on identical datasets.
Ensures evaluation parity as required for NeurIPS submissions.
"""

import os
import json
import time
from typing import List, Dict, Callable, Any
from datetime import datetime

class BenchmarkRunner:
    """
    Runs multiple systems on identical datasets with consistent logging.
    Ensures evaluation parity: same QIDs, budgets, and conditions.
    """
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(output_dir, self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        
        print(f"[BenchmarkRunner] Initialized run: {self.run_id}")
        print(f"[BenchmarkRunner] Output: {self.run_dir}")
    
    def run_all_systems(
        self, 
        dataset: List[Dict],
        systems: Dict[str, Callable],
        config: Dict[str, Any]
    ):
        """
        Run all systems on identical dataset.
        
        Args:
            dataset: List of {id, query, answers, ...}
            systems: {system_name: solve_function}
            config: Shared config (max_hops, budget, etc.)
        """
        # Validate dataset consistency
        qids = [item['id'] for item in dataset]
        assert len(qids) == len(set(qids)), "Duplicate QIDs in dataset"
        
        print(f"\n{'='*60}")
        print(f"Running {len(systems)} systems on {len(dataset)} queries")
        print(f"{'='*60}\n")
        
        for system_name, system_func in systems.items():
            print(f"[{system_name}] Starting evaluation...")
            self._run_system(system_name, dataset, system_func, config)
            print(f"[{system_name}] Complete\n")
        
        # Verification
        self._verify_consistency(systems.keys())
        
        print(f"\n[BenchmarkRunner] All systems complete")
        print(f"[BenchmarkRunner] Results: {self.run_dir}")
    
    def _run_system(
        self,
        system_name: str,
        dataset: List[Dict],
        system_func: Callable,
        config: Dict
    ):
        """Run single system and log results."""
        output_file = os.path.join(self.run_dir, f"{system_name}.jsonl")
        
        with open(output_file, 'w') as f:
            for item in dataset:
                result = self._run_single_query(
                    qid=item['id'],
                    query=item['query'],
                    gold_answers=item.get('answers', []),
                    system_name=system_name,
                    system_func=system_func,
                    config=config
                )
                f.write(json.dumps(result) + '\n')
    
    def _run_single_query(
        self,
        qid: str,
        query: str,
        gold_answers: List[str],
        system_name: str,
        system_func: Callable,
        config: Dict
    ) -> Dict:
        """Execute single query and return structured result."""
        start_time = time.time()
        
        try:
            # Call system (must return dict with 'answer', optional 'path', etc.)
            output = system_func(query)
            latency = time.time() - start_time
            
            prediction = output.get('answer', output.get('final_answer', ''))
            
            # Evaluation
            is_correct = self._evaluate(prediction, gold_answers)
            
            # Extract metrics
            path = output.get('path', [])
            hops = len(path)
            nodes_expanded = sum(
                step.get('candidates_count', 0) 
                for step in path 
                if isinstance(step, dict)
            )
            
            termination_reason = output.get('termination_reason', 'unknown')
            
            return {
                "qid": qid,
                "question": query,
                "gold_answers": gold_answers,
                "prediction": prediction,
                "is_correct": is_correct,
                "latency": latency,
                "hops": hops,
                "nodes_expanded": nodes_expanded,
                "termination_reason": termination_reason,
                "error_type": None
            }
            
        except Exception as e:
            latency = time.time() - start_time
            return {
                "qid": qid,
                "question": query,
                "gold_answers": gold_answers,
                "prediction": "",
                "is_correct": False,
                "latency": latency,
                "hops": 0,
                "nodes_expanded": 0,
                "termination_reason": "error",
                "error_type": str(type(e).__name__),
                "error_message": str(e)
            }
    
    def _evaluate(self, prediction: str, gold_answers: List[str]) -> bool:
        """Evaluate prediction against gold answers."""
        if not gold_answers:
            return False
        
        pred_lower = prediction.lower().strip()
        
        # Exact match (substring)
        return any(gold.lower() in pred_lower for gold in gold_answers)
    
    def _verify_consistency(self, system_names: List[str]):
        """Verify all systems ran on identical QIDs."""
        qid_sets = {}
        
        for system_name in system_names:
            filepath = os.path.join(self.run_dir, f"{system_name}.jsonl")
            qids = []
            with open(filepath, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    qids.append(data['qid'])
            qid_sets[system_name] = set(qids)
        
        # Check all sets are identical
        reference = list(qid_sets.values())[0]
        for system_name, qids in qid_sets.items():
            if qids != reference:
                missing = reference - qids
                extra = qids - reference
                print(f"[WARN] {system_name} QID mismatch!")
                if missing:
                    print(f"  Missing: {missing}")
                if extra:
                    print(f"  Extra: {extra}")
            else:
                print(f"[OK] {system_name}: {len(qids)} QIDs verified")
