import argparse
import sys
import os

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from crag.data.loaders import WebQSPLoader, MetaQALoader
from crag.evaluation.experiment_manager import ExperimentManager
from crag.agent.cra import CognitiveRetrievalAgent
from crag.retrieval.hybrid import HybridRetrievalModule
from crag.retrieval.vector_store import FaissVectorStore
from crag.graph.wikidata import WikidataKG
from crag.model.colbert import ColBERTReranker
from crag.llm.interface import MockLLMClient

def main():
    parser = argparse.ArgumentParser(description="Run C-RAG Benchmarks")
    parser.add_argument("--dataset", type=str, default="webqsp", choices=["webqsp", "metaqa"], help="Dataset to evaluate on")
    parser.add_argument("--output", type=str, default="results.tex", help="Output LaTeX table file")
    args = parser.parse_args()

    print(f"=== Running C-RAG Benchmark on {args.dataset.upper()} ===")
    
    # 1. Load Data
    if args.dataset == "webqsp":
        loader = WebQSPLoader()
    else:
        loader = MetaQALoader()
    data = loader.load()
    print(f"Loaded {len(data)} test samples.")

    # 2. Setup System Components
    kg = WikidataKG()
    vs = FaissVectorStore() # Mock
    hrm = HybridRetrievalModule(kg, vs)
    reranker = ColBERTReranker()
    llm = MockLLMClient() # Use Mock LLM for benchmark speed unless configured
    
    agent = CognitiveRetrievalAgent(hrm, reranker, llm_client=llm)

    # 3. Define Model Function using Agent
    # 3. Define Model Function using Agent
    def run_crag_full(query):
        # Full System
        agent = CognitiveRetrievalAgent(hrm, reranker, llm_client=llm, use_reranker=True)
        state = agent.solve(query)
        # Extract answer from ReasoningState
        return {
            "answer": state.final_answer if state.final_answer else "No answer found",
            "path": state.trajectory
        }

    # 4. Setup Baselines
    def run_crag_no_rerank(query):
        # Ablation: No ColBERT
        agent = CognitiveRetrievalAgent(hrm, reranker, llm_client=llm, use_reranker=False)
        state = agent.solve(query)
        return {
            "answer": state.final_answer if state.final_answer else "No answer found",
            "path": state.trajectory
        }

    # 5. Run Experiments
    manager = ExperimentManager()
    
    # Run C-RAG (Full)
    manager.run_experiment("C-RAG (Full)", data, run_crag_full)
    
    # Run C-RAG (No Rerank)
    manager.run_experiment("C-RAG (No Rerank)", data, run_crag_no_rerank)
    
    # 6. Generate Report
    manager.generate_latex_table(args.output)

if __name__ == "__main__":
    main()
