"""
C-RAG V3 Production Vector Store
FAISS-backed Dense Retrieval with Sentence Transformers
"""
import torch
import numpy as np
import json
import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class FaissVectorStore:
    """
    Production-grade FAISS Vector Store.
    Supports: Add, Search, Save, Load with metadata preservation.
    """
    def __init__(self, embedding_dim: int = 768, index_type: str = "IVF"):
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.index = None
        self.documents = []  # List of {id, text, metadata}
        self.encoder = None
        self._init_index()
        
    def _init_index(self):
        try:
            import faiss
            if self.index_type == "Flat":
                self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product for cosine
            else:
                # IVF with 100 centroids, requires training
                quantizer = faiss.IndexFlatIP(self.embedding_dim)
                self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, 100, faiss.METRIC_INNER_PRODUCT)
            logger.info(f"FAISS Index initialized: {self.index_type}, dim={self.embedding_dim}")
        except ImportError:
            logger.warning("FAISS not available. Using numpy fallback.")
            self.index = None
            self._embeddings_matrix = None
            
    def _get_encoder(self):
        if self.encoder is None:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer('intfloat/e5-base-v2')
        return self.encoder
        
    def add_documents(self, documents: List[Dict[str, Any]], batch_size: int = 64):
        """
        Add documents to the index.
        Each document: {id, text, metadata (optional)}
        """
        encoder = self._get_encoder()
        
        texts = [f"passage: {doc.get('text', '')}" for doc in documents]
        
        logger.info(f"Encoding {len(texts)} documents...")
        embeddings = encoder.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
        
        # Normalize for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings.astype('float32')
        
        if self.index is not None:
            import faiss
            if isinstance(self.index, faiss.IndexIVFFlat) and not self.index.is_trained:
                logger.info("Training IVF index...")
                self.index.train(embeddings)
            self.index.add(embeddings)
        else:
            # Numpy fallback
            if self._embeddings_matrix is None:
                self._embeddings_matrix = embeddings
            else:
                self._embeddings_matrix = np.vstack([self._embeddings_matrix, embeddings])
                
        # Store documents for retrieval
        start_idx = len(self.documents)
        for i, doc in enumerate(documents):
            doc['_faiss_id'] = start_idx + i
            self.documents.append(doc)
            
        logger.info(f"Added {len(documents)} documents. Total: {len(self.documents)}")
        
    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        """
        encoder = self._get_encoder()
        
        query_emb = encoder.encode(f"query: {query}", convert_to_numpy=True)
        query_emb = query_emb / np.linalg.norm(query_emb)
        query_emb = query_emb.astype('float32').reshape(1, -1)
        
        if self.index is not None and self.index.ntotal > 0:
            scores, indices = self.index.search(query_emb, min(k, self.index.ntotal))
            scores = scores[0]
            indices = indices[0]
        elif self._embeddings_matrix is not None:
            # Numpy fallback
            scores = np.dot(self._embeddings_matrix, query_emb.T).flatten()
            top_k = min(k, len(scores))
            indices = np.argsort(scores)[::-1][:top_k]
            scores = scores[indices]
        else:
            return []
            
        results = []
        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= len(self.documents):
                continue
            doc = self.documents[idx].copy()
            doc['score'] = float(score)
            doc['metadata'] = doc.get('metadata', {'id': doc.get('id', idx)})
            if 'id' not in doc['metadata']:
                doc['metadata']['id'] = doc.get('id', idx)
            results.append(doc)
            
        return results
        
    def save(self, path: str):
        """Save index and documents."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        if self.index is not None:
            import faiss
            faiss.write_index(self.index, str(path / "index.faiss"))
            
        with open(path / "documents.json", 'w', encoding='utf-8') as f:
            json.dump(self.documents, f)
            
        logger.info(f"Saved VectorStore to {path}")
        
    def load(self, path: str):
        """Load index and documents."""
        path = Path(path)
        
        if (path / "index.faiss").exists():
            import faiss
            self.index = faiss.read_index(str(path / "index.faiss"))
            
        if (path / "documents.json").exists():
            with open(path / "documents.json", 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
                
        logger.info(f"Loaded VectorStore from {path}: {len(self.documents)} documents")


# Backwards-compatible alias
VectorStore = FaissVectorStore
