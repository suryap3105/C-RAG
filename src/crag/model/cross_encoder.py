"""
C-RAG V3 Production Cross-Encoder Reranker
Transformer-based Reranking with ColBERT Fallback
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import List, Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Production Cross-Encoder Reranker.
    Uses pretrained cross-encoder for high-precision reranking.
    """
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", 
                 device: str = None, max_length: int = 512):
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = max_length
        self._model = None
        
    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"Loading CrossEncoder: {self.model_name}")
                self._model = CrossEncoder(self.model_name, device=self.device, max_length=self.max_length)
            except ImportError:
                logger.warning("sentence-transformers not found. Using fallback scoring.")
                self._model = "FALLBACK"
                
    def score(self, query: str, documents: List[str]) -> List[float]:
        """
        Score query-document pairs.
        Returns list of relevance scores.
        """
        self._load_model()
        
        if not documents:
            return []
            
        if self._model == "FALLBACK":
            # Length-based heuristic fallback
            return [len(doc) / 1000.0 for doc in documents]
            
        pairs = [[query, doc] for doc in documents]
        scores = self._model.predict(pairs, show_progress_bar=False)
        
        if not isinstance(scores, list):
            scores = scores.tolist()
            
        return scores
        
    def rerank(self, query: str, candidates: List[Dict[str, Any]], 
               top_k: int = 10, text_key: str = 'text') -> List[Dict[str, Any]]:
        """
        Rerank candidates by cross-encoder score.
        
        Args:
            query: Query string
            candidates: List of candidate dicts with 'text' field
            top_k: Number of top results to return
            text_key: Key for text in candidate dict
            
        Returns:
            Reranked candidates with added 'rerank_score' field
        """
        if not candidates:
            return []
            
        documents = [c.get(text_key, '') for c in candidates]
        scores = self.score(query, documents)
        
        # Pair with original candidates
        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Add scores and return top-k
        results = []
        for cand, score in scored[:top_k]:
            cand = cand.copy()
            cand['rerank_score'] = float(score)
            results.append(cand)
            
        return results


class ColBERTReranker(nn.Module):
    """
    ColBERT-style late interaction reranker.
    More efficient than cross-encoder for large candidate sets.
    """
    def __init__(self, model_name: str = "intfloat/e5-base-v2", 
                 device: str = None, max_length: int = 256):
        super().__init__()
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = max_length
        self.encoder = None
        
    def _get_encoder(self):
        if self.encoder is None:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(self.model_name, device=self.device)
        return self.encoder
        
    def _encode_tokens(self, texts: List[str], prefix: str = "") -> torch.Tensor:
        """Get token-level embeddings."""
        encoder = self._get_encoder()
        
        with torch.no_grad():
            all_embs = []
            for text in texts:
                full_text = f"{prefix}{text}"
                encoded = encoder.tokenize([full_text])
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                
                output = encoder[0].auto_model(**encoded)
                embs = output.last_hidden_state[0]
                embs = F.normalize(embs, p=2, dim=-1)
                all_embs.append(embs)
                
        return all_embs
        
    def score_maxsim(self, query: str, documents: List[str]) -> List[float]:
        """
        Score using MaxSim late interaction.
        """
        if not documents:
            return []
            
        # Encode query tokens
        query_embs = self._encode_tokens([query], prefix="query: ")[0]  # [Q, H]
        
        # Encode document tokens
        doc_embs = self._encode_tokens(documents, prefix="passage: ")
        
        scores = []
        for doc_emb in doc_embs:  # [D, H]
            # MaxSim: sum_q max_d (q . d)
            sim = torch.matmul(query_embs, doc_emb.T)  # [Q, D]
            max_sim = sim.max(dim=1)[0]  # [Q]
            score = max_sim.sum().item()
            scores.append(score)
            
        return scores
        
    def rerank(self, query: str, candidates: List[Dict[str, Any]], 
               top_k: int = 10, text_key: str = 'text') -> List[Dict[str, Any]]:
        """Rerank using MaxSim."""
        if not candidates:
            return []
            
        documents = [c.get(text_key, '') for c in candidates]
        scores = self.score_maxsim(query, documents)
        
        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for cand, score in scored[:top_k]:
            cand = cand.copy()
            cand['rerank_score'] = float(score)
            results.append(cand)
            
        return results


class HybridReranker:
    """
    Combines Cross-Encoder and ColBERT for adaptive reranking.
    Uses ColBERT for initial filtering, Cross-Encoder for final precision.
    """
    def __init__(self, cross_encoder_cutoff: int = 20, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.cross_encoder_cutoff = cross_encoder_cutoff
        
        self.colbert = ColBERTReranker(device=self.device)
        self.cross_encoder = CrossEncoderReranker(device=self.device)
        
    def rerank(self, query: str, candidates: List[Dict[str, Any]], 
               top_k: int = 10, text_key: str = 'text') -> List[Dict[str, Any]]:
        """
        Two-stage reranking:
        1. ColBERT filters to top cross_encoder_cutoff
        2. Cross-encoder reranks for final top_k
        """
        if len(candidates) <= self.cross_encoder_cutoff:
            # Small enough for direct cross-encoder
            return self.cross_encoder.rerank(query, candidates, top_k, text_key)
            
        # Stage 1: ColBERT filtering
        filtered = self.colbert.rerank(query, candidates, self.cross_encoder_cutoff, text_key)
        
        # Stage 2: Cross-encoder precision
        final = self.cross_encoder.rerank(query, filtered, top_k, text_key)
        
        return final
