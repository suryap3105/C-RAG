
import argparse
import sys
import os

from crag.utils.config import load_config
from crag.utils.repro import seed_everything, log_env
from crag.data.loaders import WebQSPLoader, MetaQALoader
from crag.evaluation.experiment_manager import ExperimentManager
from crag.llm.interface import MockLLMClient, OllamaClient
from crag.retrieval.vector_store import FaissVectorStore
from crag.graph.wikidata import WikidataKG
from crag.retrieval.hybrid import HybridRetrievalModule
from crag.model.colbert import ColBERTReranker

# Models
from crag.agent.cra import CognitiveRetrievalAgent
from crag.baselines.vector import VectorBaseline
from crag.baselines.static_graph import StaticGraphBaseline

def main():
    parser = argparse.ArgumentParser(description="Run C-RAG Experiments")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    args = parser.parse_args()
    
    # 1. Load Config & Seed
    cfg = load_config(args.config)
    seed_everything(cfg.get('seed', 42))
    log_env()
    
    print(f"=== Starting Experiment: {cfg.get('experiment_name', 'Unnamed')} ===")
    
    # 2. Setup Shared Components
    # LLM - Default to Ollama for production
    llm_cfg = cfg.get('llm', {})
    provider = llm_cfg.get('provider', 'ollama')  # Default to ollama
    
    # Healthcheck if configured
    if cfg.get('healthcheck_on_startup', True) and provider == 'ollama':
        from crag.llm.healthcheck import LLMHealthCheck
        base_url = llm_cfg.get('base_url', 'http://localhost:11434')
        checker = LLMHealthCheck(backend='ollama', base_url=base_url)
        if not checker.run():
            print("[FATAL] Ollama healthcheck failed. Use --config crag_mock.yaml for testing.")
            sys.exit(1)
    
    if provider == 'mock':
        print("[WARN] Using MockLLM for testing. Not suitable for production.")
        llm = MockLLMClient()
    else:
        # Production: Use Ollama
        model_name = llm_cfg.get('model', 'llama3')
        base_url = llm_cfg.get('base_url', 'http://localhost:11434')
        print(f"[LLM] Connecting to Ollama: {model_name} @ {base_url}")
        llm = OllamaClient(model_name=model_name, base_url=base_url)
        
    # KG & Vector Store
    kg = WikidataKG()
    vs = FaissVectorStore()
    
    # Try to load pre-built vectorstore
    dataset_name = cfg.get('dataset', 'metaqa')
    vectorstore_path = f"data/{dataset_name}/vectorstore"
    if os.path.exists(vectorstore_path):
        vs.load(vectorstore_path)
    else:
        print(f"[WARN] No vectorstore found at {vectorstore_path}")
        print(f"[WARN] Run: python scripts/build_vectorstore.py --dataset {dataset_name}")
    
    
    # Retrieval Module
    hrm = HybridRetrievalModule(kg, vs)
    
    # Reranker
    reranker = ColBERTReranker()

    # 3. Instantiate System based on Config
    system_type = cfg.get('experiment_name', '').lower()
    
    model_func = None
    system_label = cfg.get('experiment_name')
    
    if "baseline_vector" in system_type:
        print("-> Mode: Vector Baseline")
        model = VectorBaseline(vs, llm, k=cfg['retrieval']['k'])
        model_func = model.solve
    
    elif "baseline_graph" in system_type or "static" in system_type:
        print("-> Mode: Static Graph Baseline")
        model = StaticGraphBaseline(hrm, llm)
        model_func = model.solve
        
    else:
        print(f"-> Mode: C-RAG Agent (Full/Ablated)")
        # Parse Agent Config
        agent_cfg = cfg.get('agent', {})
        retrieval_cfg = cfg.get('retrieval', {})
        
        agent = CognitiveRetrievalAgent(
            hrm, 
            reranker, 
            llm_client=llm,
            use_reranker=retrieval_cfg.get('use_reranker', True)
        )
        # Apply other agent config settings if class supports them
        agent.max_steps = agent_cfg.get('max_hops', 3)
        agent.max_expansions = agent_cfg.get('max_expansions', 3)
        
        # Wrapper to extract 'answer' from ReasoningState
        def agent_wrapper(query):
            state = agent.solve(query)
            return {
                "answer": state.final_answer if state.final_answer else "No answer found",
                "path": state.trajectory,
                "final_answer": state.final_answer,
                "termination_reason": state.termination_reason
            }
        model_func = agent_wrapper

    # 4. Load Data
    dataset_name = cfg.get('dataset', 'metaqa')
    if dataset_name == 'webqsp':
        loader = WebQSPLoader()
    else:
        loader = MetaQALoader()
        
    data = loader.load()
    print(f"-> Loaded {len(data)} samples from {dataset_name}")

    # 5. Run Experiment
    manager = ExperimentManager(output_dir=cfg.get('output_dir', 'runs'))
    manager.run_experiment(system_label, data, model_func)
    
    # 6. Final Report
    manager.generate_latex_table()
    print("=== Experiment Completed ===")

if __name__ == "__main__":
    main()
