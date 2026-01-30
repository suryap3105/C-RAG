"""
C-RAG V3 Production Experiment Manager
Complete Evaluation Pipeline with Metrics
"""
import json
import logging
import time
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics."""
    # Retrieval Metrics
    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr: float = 0.0  # Mean Reciprocal Rank
    ndcg_at_10: float = 0.0
    
    # QA Metrics
    exact_match: float = 0.0
    f1_score: float = 0.0
    
    # Speed Metrics
    avg_latency_ms: float = 0.0
    throughput_qps: float = 0.0
    
    # Robustness Metrics
    degradation_at_20_noise: float = 0.0
    degradation_at_50_noise: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


class MetricsComputer:
    """Compute retrieval and QA metrics."""
    
    @staticmethod
    def compute_recall_at_k(retrieved: List[Any], relevant: List[Any], k: int) -> float:
        """Compute Recall@K."""
        retrieved_k = set(retrieved[:k])
        relevant_set = set(relevant)
        if not relevant_set:
            return 0.0
        return len(retrieved_k & relevant_set) / len(relevant_set)
        
    @staticmethod
    def compute_mrr(retrieved: List[Any], relevant: List[Any]) -> float:
        """Compute Mean Reciprocal Rank."""
        relevant_set = set(relevant)
        for i, item in enumerate(retrieved):
            if item in relevant_set:
                return 1.0 / (i + 1)
        return 0.0
        
    @staticmethod
    def compute_ndcg(retrieved: List[Any], relevant: List[Any], k: int = 10) -> float:
        """Compute NDCG@K."""
        def dcg(scores, k):
            scores = scores[:k]
            return sum(s / np.log2(i + 2) for i, s in enumerate(scores))
            
        relevant_set = set(relevant)
        relevance = [1.0 if r in relevant_set else 0.0 for r in retrieved[:k]]
        
        ideal = sorted(relevance, reverse=True)
        
        dcg_val = dcg(relevance, k)
        idcg_val = dcg(ideal, k)
        
        return dcg_val / idcg_val if idcg_val > 0 else 0.0
        
    @staticmethod
    def compute_f1(pred: str, gold: str) -> float:
        """Compute token-level F1 score."""
        pred_tokens = set(pred.lower().split())
        gold_tokens = set(gold.lower().split())
        
        if not pred_tokens or not gold_tokens:
            return 0.0
            
        common = pred_tokens & gold_tokens
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gold_tokens)
        
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
        
    @staticmethod
    def compute_exact_match(pred: str, gold: str) -> float:
        """Compute exact match."""
        return 1.0 if pred.strip().lower() == gold.strip().lower() else 0.0


class ExperimentManager:
    """
    Production Experiment Manager.
    Handles data loading, evaluation, and result reporting.
    """
    def __init__(self, output_dir: str = "experiments"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results: Dict[str, Any] = {}
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def load_dataset(self, path: str, name: str = "metaqa") -> List[Dict]:
        """
        Load evaluation dataset.
        Expected format: JSONL with {query, answers, relevant_ids}
        """
        data = []
        path = Path(path)
        
        if path.suffix == '.jsonl':
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
        elif path.suffix == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict) and 'data' in data:
                    data = data['data']
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")
            
        logger.info(f"Loaded {len(data)} samples from {path}")
        return data
        
    def evaluate_retrieval(self, retriever, dataset: List[Dict], 
                          k_values: List[int] = [1, 5, 10]) -> EvaluationMetrics:
        """
        Evaluate retrieval performance.
        """
        metrics_computer = MetricsComputer()
        
        recalls = {k: [] for k in k_values}
        mrrs = []
        ndcgs = []
        latencies = []
        
        logger.info(f"Evaluating retrieval on {len(dataset)} samples...")
        
        for i, sample in enumerate(dataset):
            query = sample.get('query', sample.get('question', ''))
            relevant = sample.get('relevant_ids', sample.get('answer_ids', []))
            
            if not query:
                continue
                
            # Retrieve
            start = time.time()
            results = retriever.retrieve(query, k=max(k_values))
            latency = (time.time() - start) * 1000
            latencies.append(latency)
            
            # Get retrieved IDs
            retrieved = [r.get('id', r.get('metadata', {}).get('id')) for r in results]
            
            # Compute metrics
            for k in k_values:
                recalls[k].append(metrics_computer.compute_recall_at_k(retrieved, relevant, k))
                
            mrrs.append(metrics_computer.compute_mrr(retrieved, relevant))
            ndcgs.append(metrics_computer.compute_ndcg(retrieved, relevant, k=10))
            
            if (i + 1) % 100 == 0:
                logger.info(f"Evaluated {i + 1}/{len(dataset)}")
                
        # Aggregate
        metrics = EvaluationMetrics(
            recall_at_1=np.mean(recalls[1]) if 1 in recalls else 0,
            recall_at_5=np.mean(recalls[5]) if 5 in recalls else 0,
            recall_at_10=np.mean(recalls[10]) if 10 in recalls else 0,
            mrr=np.mean(mrrs),
            ndcg_at_10=np.mean(ndcgs),
            avg_latency_ms=np.mean(latencies),
            throughput_qps=1000 / np.mean(latencies) if latencies else 0
        )
        
        return metrics
        
    def evaluate_qa(self, qa_pipeline, dataset: List[Dict]) -> EvaluationMetrics:
        """
        Evaluate end-to-end QA performance.
        """
        metrics_computer = MetricsComputer()
        
        ems = []
        f1s = []
        latencies = []
        
        logger.info(f"Evaluating QA on {len(dataset)} samples...")
        
        for i, sample in enumerate(dataset):
            query = sample.get('query', sample.get('question', ''))
            answers = sample.get('answers', sample.get('answer', []))
            
            if isinstance(answers, str):
                answers = [answers]
                
            if not query or not answers:
                continue
                
            # Get answer
            start = time.time()
            pred = qa_pipeline.answer(query)
            latency = (time.time() - start) * 1000
            latencies.append(latency)
            
            # Best match against any gold answer
            best_em = max(metrics_computer.compute_exact_match(pred, a) for a in answers)
            best_f1 = max(metrics_computer.compute_f1(pred, a) for a in answers)
            
            ems.append(best_em)
            f1s.append(best_f1)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Evaluated {i + 1}/{len(dataset)}")
                
        metrics = EvaluationMetrics(
            exact_match=np.mean(ems),
            f1_score=np.mean(f1s),
            avg_latency_ms=np.mean(latencies)
        )
        
        return metrics
        
    def run_robustness_experiment(self, retriever, dataset: List[Dict],
                                  graph_engine, muddier_class,
                                  noise_levels: List[float] = [0.0, 0.2, 0.5]) -> Dict[float, EvaluationMetrics]:
        """
        Run robustness experiments at different noise levels.
        """
        results = {}
        baseline_mrr = None
        
        for noise in noise_levels:
            logger.info(f"Running robustness experiment at {noise*100:.0f}% noise...")
            
            if noise > 0:
                # Apply noise
                noise_config = {
                    'phantom_nodes_ratio': noise * 0.5,
                    'bridge_noise_ratio': noise * 0.3
                }
                muddier = muddier_class(graph_engine, noise_config)
                noisy_ge = muddier.apply()
                
                # Rebuild retriever with noisy graph
                # This requires the retriever to support graph replacement
                if hasattr(retriever, 'graph_engine'):
                    original_ge = retriever.graph_engine
                    retriever.graph_engine = noisy_ge
                    
            metrics = self.evaluate_retrieval(retriever, dataset)
            results[noise] = metrics
            
            if noise == 0:
                baseline_mrr = metrics.mrr
            elif baseline_mrr and baseline_mrr > 0:
                degradation = (baseline_mrr - metrics.mrr) / baseline_mrr
                if noise == 0.2:
                    metrics.degradation_at_20_noise = degradation
                elif noise == 0.5:
                    metrics.degradation_at_50_noise = degradation
                    
            # Restore original graph
            if noise > 0 and hasattr(retriever, 'graph_engine'):
                retriever.graph_engine = original_ge
                
        return results
        
    def save_results(self, name: str, metrics: EvaluationMetrics, 
                    config: Dict[str, Any] = None):
        """Save experiment results."""
        result = {
            'name': name,
            'run_id': self.run_id,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics.to_dict(),
            'config': config or {}
        }
        
        self.results[name] = result
        
        # Save to file
        output_path = self.output_dir / f"{self.run_id}_{name}.json"
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
            
        logger.info(f"Saved results to {output_path}")
        
    def generate_report(self) -> str:
        """Generate markdown report of all experiments."""
        lines = [
            "# C-RAG V3 Experiment Report",
            f"Run ID: {self.run_id}",
            f"Generated: {datetime.now().isoformat()}",
            "",
            "## Results Summary",
            ""
        ]
        
        for name, result in self.results.items():
            metrics = result['metrics']
            lines.append(f"### {name}")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            
            for k, v in metrics.items():
                if isinstance(v, float):
                    lines.append(f"| {k} | {v:.4f} |")
                else:
                    lines.append(f"| {k} | {v} |")
                    
            lines.append("")
            
        report = "\n".join(lines)
        
        # Save report
        report_path = self.output_dir / f"{self.run_id}_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
            
        return report
