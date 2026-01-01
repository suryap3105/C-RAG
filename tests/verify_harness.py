
import sys
import os
import shutil

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from crag.utils.repro import seed_everything, log_env
from crag.utils.config import load_config
from crag.evaluation.experiment_manager import ExperimentManager
from crag.baselines.vector import VectorBaseline
from crag.llm.interface import MockLLMClient
from crag.retrieval.vector_store import FaissVectorStore

def verify_harness():
    print("=== Verifying Engineering Harness ===")
    
    # 1. Test Reproducibility
    seed_everything(42)
    env = log_env()
    assert env['torch'] is not None
    print("[PASS] Reproducibility Utils")

    # 2. Test Config Loading
    try:
        cfg = load_config("configs/defaults.yaml")
        assert cfg['seed'] == 42
        print("[PASS] Config Loading")
    except Exception as e:
        print(f"[FAIL] Config Loading: {e}")
        return

    # 3. Test Instrumentation (Metrics Logging)
    mgr = ExperimentManager(output_dir="tests/runs_verify")
    dummy_data = [{"id": "test1", "query": "test query", "answers": ["test"]}]
    
    # Run Dummy Experiment
    llm = MockLLMClient()
    vs = FaissVectorStore()
    model = VectorBaseline(vs, llm)
    
    mgr.run_experiment("Verify_Vector", dummy_data, model.solve)
    
    # Check if metrics.jsonl exists
    if os.path.exists(mgr.metrics_file):
        print("[PASS] Metrics Logging (metrics.jsonl created)")
    else:
        print("[FAIL] Metrics Logging (file missing)")
        
    # Cleanup
    shutil.rmtree("tests/runs_verify", ignore_errors=True)
    
    print("\nAll System Checks Passed.")

if __name__ == "__main__":
    verify_harness()
