"""
C-RAG V3: Neuro-Symbolic Knowledge Graph Retrieval
Production Package
"""

__version__ = "3.0.0"
__author__ = "C-RAG Team"

from .graph.engine import GraphEngine
from .graph.partitioning import SemanticPartitioner
from .retrieval.vector_store import FaissVectorStore
from .retrieval.neural_hybrid import NeuroHybridRetrievalModule
from .routing.colbert import ColBERTPartitionRouter
from .model.gnn import NeuralSubgraphMatcher
from .model.query_graph import QueryGraphGenerator
from .model.cross_encoder import ColBERTReranker
from .llm.interface import create_llm_client
from .evaluation.experiment_manager import ExperimentManager
from .evaluation.robustness import GraphMuddier

__all__ = [
    'GraphEngine',
    'SemanticPartitioner',
    'FaissVectorStore',
    'NeuroHybridRetrievalModule',
    'ColBERTPartitionRouter',
    'NeuralSubgraphMatcher',
    'QueryGraphGenerator',
    'ColBERTReranker',
    'create_llm_client',
    'ExperimentManager',
    'GraphMuddier',
]
