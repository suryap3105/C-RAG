"""
Example: Robustness Evaluation
"""
from crag import GraphEngine, ExperimentManager, GraphMuddier
from crag.run_exp import build_pipeline, load_config

# Load configuration and pipeline
config = load_config("configs/default.yaml")
pipeline = build_pipeline(config)

# Load test dataset
exp_manager = ExperimentManager(output_dir="experiments/robustness")
dataset = exp_manager.load_dataset("data/test.jsonl")

# Run robustness experiments at multiple noise levels
noise_levels = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

results = exp_manager.run_robustness_experiment(
    retriever=pipeline,
    dataset=dataset,
    graph_engine=pipeline.graph_engine,
    muddier_class=GraphMuddier,
    noise_levels=noise_levels
)

# Print degradation curve
print("\n=== Robustness Degradation Curve ===")
print(f"{'Noise Level':<12} {'MRR':<10} {'R@10':<10} {'Degradation':<12}")
print("-" * 50)

baseline_mrr = results[0.0].mrr

for noise, metrics in results.items():
    degradation = (baseline_mrr - metrics.mrr) / baseline_mrr * 100 if baseline_mrr > 0 else 0
    print(f"{noise*100:>10.0f}%  {metrics.mrr:>8.4f}  {metrics.recall_at_10:>8.4f}  {degradation:>10.1f}%")

# Generate full report
report = exp_manager.generate_report()
print(f"\nReport saved to experiments/robustness/")
