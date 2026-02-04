"""
C-RAG V3 Production ColBERT Reranker
ColBERT-style late interaction reranker.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


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

# Alias for backwards compatibility if needed, though we prefer direct usage
HybridReranker = ColBERTReranker
