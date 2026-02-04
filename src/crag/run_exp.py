"""
C-RAG V3 Production Main Entry Point
Complete Pipeline Orchestration
"""
import argparse
import logging
import sys
import json
import torch
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Local imports
from crag.graph.engine import GraphEngine
from crag.graph.partitioning import GraphPartitioner
from crag.retrieval.vector_store import FaissVectorStore
from crag.retrieval.neural_hybrid import NeuroHybridRetrievalModule
from crag.routing.colbert import ColBERTPartitionRouter
from crag.model.gnn import NeuralSubgraphMatcher
from crag.model.query_graph import QueryGraphGenerator
from crag.model.cross_encoder import ColBERTReranker
from crag.llm.interface import create_llm_client
from crag.evaluation.experiment_manager import ExperimentManager
from crag.evaluation.robustness import GraphMuddier


def load_config(config_path: str) -> dict:
    """Load YAML/JSON configuration."""
    path = Path(config_path)
    
    if not path.exists():
        logger.warning(f"Config not found: {config_path}. Using defaults.")
        return {}
        
    with open(path, 'r') as f:
        if path.suffix in ['.yaml', '.yml']:
            import yaml
            return yaml.safe_load(f)
        else:
            return json.load(f)


def build_pipeline(config: dict, device: str = None) -> NeuroHybridRetrievalModule:
    """
    Build the complete C-RAG V3 retrieval pipeline.
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Building pipeline on device: {device}")
    
    # 1. Graph Engine
    graph_engine = GraphEngine()
    
    graph_config = config.get('graph', {})
    if graph_config.get('nodes_path') and graph_config.get('edges_path'):
        graph_engine.load_graph(
            graph_config['edges_path'],
            graph_config['nodes_path']
        )
        
        # Compute embeddings if not already present
        if graph_engine.data.x is None:
            graph_engine.compute_embeddings()
    else:
        # Initialize with dummy data for testing
        logger.warning("No graph data paths. Initializing dummy graph.")
        graph_engine.data.num_nodes = 100
        graph_engine.data.edge_index = torch.randint(0, 100, (2, 500))
        graph_engine.data.x = torch.randn(100, 768)
        
    # 2. Partitioning
    partition_config = config.get('partitioning', {})
    if partition_config.get('enabled', True):
        partitioner = GraphPartitioner(
            method=partition_config.get('method', 'metis'),
            n_partitions=partition_config.get('n_partitions', 10)
        )
        partitioner.partition(graph_engine)
        
    # 3. Vector Store
    vs_config = config.get('vector_store', {})
    vector_store = FaissVectorStore(
        embedding_dim=vs_config.get('embedding_dim', 768),
        index_type=vs_config.get('index_type', 'Flat')
    )
    
    if vs_config.get('load_path'):
        vector_store.load(vs_config['load_path'])
    elif graph_engine.node_text_map:
        # Build from graph nodes
        documents = [
            {'id': nid, 'text': info.get('text', info.get('name', '')), 'metadata': {'id': nid}}
            for nid, info in graph_engine.node_text_map.items()
        ]
        if documents:
            vector_store.add_documents(documents)
            
    # 4. ColBERT Router
    router_config = config.get('colbert_router', {})
    colbert_router = ColBERTPartitionRouter(device=device)
    
    if router_config.get('load_path'):
        colbert_router.load(router_config['load_path'])
    elif hasattr(graph_engine.data, 'part_id'):
        colbert_router.build_partition_matrices(
            graph_engine,
            num_tokens_per_partition=router_config.get('tokens_per_partition', 32)
        )
        
    # 5. Neural Matcher
    gnn_config = config.get('gnn', {})
    neural_matcher = NeuralSubgraphMatcher(
        in_channels=gnn_config.get('in_channels', 768),
        hidden_channels=gnn_config.get('hidden_channels', 256),
        out_channels=gnn_config.get('out_channels', 256)
    )
    
    if gnn_config.get('checkpoint_path'):
        checkpoint = torch.load(gnn_config['checkpoint_path'], map_location=device)
        neural_matcher.load_state_dict(checkpoint['model_state_dict'])
        
    # 6. Query Graph Generator
    llm_config = config.get('llm', {})
    llm_client = create_llm_client(
        provider=llm_config.get('provider', 'mock'),
        model=llm_config.get('model', 'llama3.2'),
        base_url=llm_config.get('base_url', 'http://localhost:11434')
    )
    query_gen = QueryGraphGenerator(llm_client)
    
    # 7. Reranker
    reranker = ColBERTReranker(device=device)
    
    # 8. Assemble Pipeline
    pipeline = NeuroHybridRetrievalModule(
        vector_store=vector_store,
        graph_engine=graph_engine,
        query_gen=query_gen,
        neural_matcher=neural_matcher,
        colbert_router=colbert_router,
        reranker=reranker,
        use_adaptive_gating=config.get('use_adaptive_gating', True),
        device=device
    )
    
    logger.info("Pipeline built successfully")
    return pipeline


def run_experiment(args):
    """Run evaluation experiment."""
    config = load_config(args.config)
    
    # Build pipeline
    pipeline = build_pipeline(config, args.device)
    
    # Load dataset
    exp_manager = ExperimentManager(output_dir=args.output_dir)
    dataset = exp_manager.load_dataset(args.dataset)
    
    # Apply noise if specified
    if args.noise_level > 0:
        logger.info(f"Applying noise level: {args.noise_level}")
        noise_config = {
            'phantom_nodes_ratio': args.noise_level * 0.5,
            'bridge_noise_ratio': args.noise_level * 0.3,
            'attribute_noise_std': args.noise_level * 0.1
        }
        muddier = GraphMuddier(pipeline.graph_engine, noise_config)
        pipeline.graph_engine = muddier.apply()
        
    # Evaluate
    metrics = exp_manager.evaluate_retrieval(pipeline, dataset)
    
    # Save results
    exp_manager.save_results(
        name=f"v3_noise_{args.noise_level:.2f}",
        metrics=metrics,
        config={'noise_level': args.noise_level, **config}
    )
    
    # Generate report
    report = exp_manager.generate_report()
    print("\n" + report)
    
    return metrics


def run_interactive(args):
    """Run interactive query mode."""
    config = load_config(args.config)
    pipeline = build_pipeline(config, args.device)
    
    print("\n=== C-RAG V3 Interactive Mode ===")
    print("Type 'quit' to exit.\n")
    
    while True:
        query = input("Query> ").strip()
        if query.lower() in ['quit', 'exit', 'q']:
            break
            
        if not query:
            continue
            
        results = pipeline.retrieve(query, k=args.top_k)
        
        print(f"\nTop {len(results)} Results:")
        for i, r in enumerate(results, 1):
            print(f"  [{i}] (score={r.get('score', 0):.4f}) {r.get('text', '')[:100]}...")
        print()


def run_robustness(args):
    """Run robustness experiment."""
    config = load_config(args.config)
    pipeline = build_pipeline(config, args.device)
    
    exp_manager = ExperimentManager(output_dir=args.output_dir)
    dataset = exp_manager.load_dataset(args.dataset)
    
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.5]
    
    results = exp_manager.run_robustness_experiment(
        pipeline, dataset, 
        pipeline.graph_engine, GraphMuddier,
        noise_levels=noise_levels
    )
    
    print("\n=== Robustness Results ===")
    for noise, metrics in results.items():
        print(f"Noise {noise*100:.0f}%: MRR={metrics.mrr:.4f}, R@10={metrics.recall_at_10:.4f}")
        
    report = exp_manager.generate_report()
    print("\n" + report)


def main():
    parser = argparse.ArgumentParser(
        description="C-RAG V3: Neuro-Symbolic Knowledge Graph Retrieval"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Experiment command
    exp_parser = subparsers.add_parser('experiment', help='Run evaluation experiment')
    exp_parser.add_argument('--config', type=str, default='configs/default.yaml')
    exp_parser.add_argument('--dataset', type=str, required=True)
    exp_parser.add_argument('--noise_level', type=float, default=0.0)
    exp_parser.add_argument('--output_dir', type=str, default='experiments')
    exp_parser.add_argument('--device', type=str, default=None)
    
    # Interactive command
    int_parser = subparsers.add_parser('interactive', help='Interactive query mode')
    int_parser.add_argument('--config', type=str, default='configs/default.yaml')
    int_parser.add_argument('--top_k', type=int, default=10)
    int_parser.add_argument('--device', type=str, default=None)
    
    # Robustness command
    rob_parser = subparsers.add_parser('robustness', help='Run robustness experiment')
    rob_parser.add_argument('--config', type=str, default='configs/default.yaml')
    rob_parser.add_argument('--dataset', type=str, required=True)
    rob_parser.add_argument('--output_dir', type=str, default='experiments')
    rob_parser.add_argument('--device', type=str, default=None)
    
    args = parser.parse_args()
    
    if args.command == 'experiment':
        run_experiment(args)
    elif args.command == 'interactive':
        run_interactive(args)
    elif args.command == 'robustness':
        run_robustness(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
