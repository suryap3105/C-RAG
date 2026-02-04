"""
C-RAG V3 Production Hybrid Retrieval Module
Complete Neuro-Symbolic Fusion with Adaptive Gating
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from .vector_store import FaissVectorStore
from ..model.gnn import NeuralSubgraphMatcher
from ..model.cross_encoder import ColBERTReranker
from ..routing.colbert import ColBERTPartitionRouter
from ..graph.engine import GraphEngine
from ..model.query_graph import QueryGraphGenerator

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Structured retrieval result."""
    id: Any
    text: str
    score: float
    source: str  # 'vector', 'graph', 'fused'
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'text': self.text,
            'score': self.score,
            'source': self.source,
            'metadata': self.metadata or {}
        }


class AdaptiveGatingNetwork(nn.Module):
    """
    Learned gating network to adaptively blend vector and graph results.
    Predicts alpha in [0, 1] where 0=pure graph, 1=pure vector.
    """
    def __init__(self, input_dim: int = 768):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, query_embedding: torch.Tensor) -> torch.Tensor:
        """Predict fusion weight alpha."""
        return self.network(query_embedding)


class NeuroHybridRetrievalModule:
    """
    Production Neuro-Symbolic Hybrid Retrieval Module.
    
    Features:
    - Parallel vector and graph retrieval
    - Adaptive gating for result fusion
    - Weighted Reciprocal Rank Fusion
    - Neural subgraph matching
    - ColBERT reranking
    """
    def __init__(self, 
                 vector_store: FaissVectorStore,
                 graph_engine: GraphEngine,
                 query_gen: QueryGraphGenerator,
                 neural_matcher: NeuralSubgraphMatcher,
                 colbert_router: ColBERTPartitionRouter,
                 reranker: Optional[ColBERTReranker] = None,
                 use_adaptive_gating: bool = True,
                 device: str = None):
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.vector_store = vector_store
        self.graph_engine = graph_engine
        self.query_gen = query_gen
        self.neural_matcher = neural_matcher.to(self.device)
        self.colbert_router = colbert_router
        self.reranker = reranker or ColBERTReranker(device=self.device)
        
        # Adaptive gating
        self.use_adaptive_gating = use_adaptive_gating
        self.gating_network = AdaptiveGatingNetwork().to(self.device) if use_adaptive_gating else None
        
        # Query encoder for gating
        self.query_encoder = None
        
        # Default alpha when gating is disabled
        self.default_alpha = 0.5
        
        logger.info(f"NeuroHybridRetrievalModule initialized. Adaptive gating: {use_adaptive_gating}")
        
    def _get_query_encoder(self):
        if self.query_encoder is None:
            from sentence_transformers import SentenceTransformer
            self.query_encoder = SentenceTransformer('intfloat/e5-base-v2', device=self.device)
        return self.query_encoder
        
    def _predict_alpha(self, query: str) -> float:
        """Predict fusion alpha using gating network or heuristics."""
        if not self.use_adaptive_gating or self.gating_network is None:
            # Heuristic fallback
            query_lower = query.lower()
            
            # Graph-heavy patterns (structural queries)
            graph_patterns = ['connected', 'related', 'path', 'between', 'neighbor', 
                             'linked', 'relationship', 'directed', 'produced']
            
            # Vector-heavy patterns (semantic queries)
            vector_patterns = ['what is', 'who is', 'define', 'meaning', 'describe',
                              'explain', 'when', 'where']
            
            graph_score = sum(1 for p in graph_patterns if p in query_lower)
            vector_score = sum(1 for p in vector_patterns if p in query_lower)
            
            if graph_score > vector_score:
                return 0.3  # Favor graph
            elif vector_score > graph_score:
                return 0.7  # Favor vector
            return 0.5
            
        # Use learned gating
        encoder = self._get_query_encoder()
        with torch.no_grad():
            query_emb = encoder.encode(f"query: {query}", convert_to_tensor=True)
            query_emb = query_emb.to(self.device)
            alpha = self.gating_network(query_emb).item()
            
        return alpha
        
    def _vector_search(self, query: str, k: int) -> List[RetrievalResult]:
        """Execute vector search."""
        results = self.vector_store.search(query, k=k)
        
        return [
            RetrievalResult(
                id=r['metadata'].get('id', i),
                text=r.get('text', ''),
                score=r.get('score', 0.0),
                source='vector',
                metadata=r.get('metadata', {})
            )
            for i, r in enumerate(results)
        ]
        
    def _graph_search(self, query: str, k: int) -> List[RetrievalResult]:
        """Execute neural subgraph search."""
        results = []
        
        try:
            # 1. Parse query into graph
            query_graph = self.query_gen.parse(query)
            if query_graph.num_nodes == 0:
                logger.warning("Query graph is empty. Skipping graph search.")
                return []
                
            query_graph = query_graph.to(self.device)
            
            # 2. Route to relevant partitions
            partition_ids, partition_scores = self.colbert_router.route(query, query_graph=query_graph, k=5)
            
            # 3. Extract subgraphs from partitions
            candidates = []
            for pid in partition_ids:
                subgraph = self.graph_engine.get_partition_subgraph(pid)
                if subgraph.num_nodes > 0:
                    subgraph = subgraph.to(self.device)
                    candidates.append((pid, subgraph))
                    
            if not candidates:
                return []
                
            # 4. Score with neural matcher
            scored = []
            for pid, subgraph in candidates:
                score = self.neural_matcher.match(query_graph, subgraph)
                
                # Get representative nodes from subgraph
                if hasattr(subgraph, 'original_ids'):
                    for node_id in subgraph.original_ids[:k].tolist():
                        node_info = self.graph_engine.node_text_map.get(node_id, {})
                        scored.append((node_id, node_info, score, pid))
                        
            # Sort by score
            scored.sort(key=lambda x: x[2], reverse=True)
            
            # Convert to results
            for node_id, node_info, score, pid in scored[:k]:
                results.append(RetrievalResult(
                    id=node_id,
                    text=node_info.get('text', node_info.get('name', f'Node_{node_id}')),
                    score=score,
                    source='graph',
                    metadata={'partition_id': pid, 'name': node_info.get('name', '')}
                ))
                
        except Exception as e:
            logger.error(f"Graph search error: {e}")
            
        return results
        
    def _weighted_rrf(self, vec_results: List[RetrievalResult], 
                      graph_results: List[RetrievalResult],
                      k: int, alpha: float, rrf_k: int = 60) -> List[RetrievalResult]:
        """
        Weighted Reciprocal Rank Fusion.
        
        Score = alpha * (1 / (k + rank_vec)) + (1-alpha) * (1 / (k + rank_graph))
        """
        scores: Dict[Any, Dict] = {}
        
        # Process vector results
        for rank, result in enumerate(vec_results):
            rid = result.id
            vec_score = alpha * (1.0 / (rrf_k + rank + 1))
            
            if rid not in scores:
                scores[rid] = {'vec_score': 0, 'graph_score': 0, 'result': result}
            scores[rid]['vec_score'] = vec_score
            scores[rid]['result'] = result  # Prefer vector result for text
            
        # Process graph results
        for rank, result in enumerate(graph_results):
            rid = result.id
            graph_score = (1 - alpha) * (1.0 / (rrf_k + rank + 1))
            
            if rid not in scores:
                scores[rid] = {'vec_score': 0, 'graph_score': 0, 'result': result}
            scores[rid]['graph_score'] = graph_score
            
            # Merge metadata
            if scores[rid]['result'].source == 'vector':
                # Append graph metadata
                scores[rid]['result'].metadata.update(result.metadata)
                
        # Compute final scores and sort
        final_results = []
        for rid, data in scores.items():
            final_score = data['vec_score'] + data['graph_score']
            result = data['result']
            result.score = final_score
            result.source = 'fused'
            final_results.append(result)
            
        final_results.sort(key=lambda x: x.score, reverse=True)
        return final_results[:k]
        
    def retrieve(self, query: str, k: int = 10, 
                 use_reranking: bool = True) -> List[Dict[str, Any]]:
        """
        Main retrieval method.
        
        Args:
            query: User query
            k: Number of results to return
            use_reranking: Whether to apply reranking
            
        Returns:
            List of result dictionaries
        """
        logger.info(f"[NeuroHRM] Processing: {query}")
        
        # Predict fusion alpha
        alpha = self._predict_alpha(query)
        logger.info(f"[NeuroHRM] Fusion alpha: {alpha:.3f}")
        
        # Parallel retrieval
        vec_results = []
        graph_results = []
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(self._vector_search, query, k * 2): 'vector',
                executor.submit(self._graph_search, query, k * 2): 'graph'
            }
            
            for future in as_completed(futures):
                source = futures[future]
                try:
                    if source == 'vector':
                        vec_results = future.result()
                    else:
                        graph_results = future.result()
                except Exception as e:
                    logger.error(f"{source} retrieval error: {e}")
                    
        logger.info(f"[NeuroHRM] Retrieved {len(vec_results)} vector, {len(graph_results)} graph results")
        
        # Fusion
        fused_results = self._weighted_rrf(vec_results, graph_results, k * 2, alpha)
        
        # Convert to dicts
        results = [r.to_dict() for r in fused_results]
        
        # Rerank
        if use_reranking and results:
            results = self.reranker.rerank(query, results, top_k=k)
        else:
            results = results[:k]
            
        logger.info(f"[NeuroHRM] Final results: {len(results)}")
        return results
        
    def train_gating_network(self, training_data: List[Dict], epochs: int = 10):
        """
        Train the adaptive gating network.
        training_data: List of {query, vec_better: bool}
        """
        if self.gating_network is None:
            logger.warning("Gating network not initialized.")
            return
            
        encoder = self._get_query_encoder()
        optimizer = torch.optim.Adam(self.gating_network.parameters(), lr=1e-4)
        criterion = nn.BCELoss()
        
        self.gating_network.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for item in training_data:
                query = item['query']
                target = 1.0 if item['vec_better'] else 0.0
                
                with torch.no_grad():
                    query_emb = encoder.encode(f"query: {query}", convert_to_tensor=True)
                    query_emb = query_emb.to(self.device)
                    
                pred = self.gating_network(query_emb)
                loss = criterion(pred, torch.tensor([target], device=self.device))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(training_data):.4f}")
            
        self.gating_network.eval()
