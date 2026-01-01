from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np
import os

class VectorStore(ABC):
    @abstractmethod
    def add_texts(self, texts: List[str], metadatas: List[Dict] = None):
        pass

    @abstractmethod
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        pass

class FaissVectorStore(VectorStore):
    """
    Production-ready FAISS Vector Store.
    Uses 'sentence-transformers/all-MiniLM-L6-v2' for real embeddings.
    """
    def __init__(self, embedding_dim: int = 384): # 384 is dimension for all-MiniLM-L6-v2
        import faiss
        from sentence_transformers import SentenceTransformer
        
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        
        self.documents = [] # Metadata store
        
    def add_texts(self, texts: List[str], metadatas: List[Dict] = None):
        if not metadatas:
            metadatas = [{}] * len(texts)
        
        # Batch encoding for efficiency
        embeddings = self.encoder.encode(texts, convert_to_numpy=True)
        
        import faiss
        self.index.add(embeddings)
        
        for i, text in enumerate(texts):
            self.documents.append({"text": text, "metadata": metadatas[i]})
            
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        query_vec = self.encoder.encode([query], convert_to_numpy=True)
        
        D, I = self.index.search(query_vec, k)
        results = []
        for idx in I[0]:
            if idx < len(self.documents) and idx >= 0:
                results.append(self.documents[idx])
        return results

    def save_local(self, path: str):
        import faiss
        import pickle
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "docs.pkl"), "wb") as f:
            pickle.dump(self.documents, f)

    def load_local(self, path: str):
        import faiss
        import pickle
        self.index = faiss.read_index(os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "docs.pkl"), "rb") as f:
            self.documents = pickle.load(f)
    
    def load(self, vectorstore_dir: str):
        """
        Load vectorstore built by build_vectorstore.py.
        Expected format:
          - faiss.index
          - id_map.json
          - version.json
        """
        import faiss
        import json
        
        index_path = os.path.join(vectorstore_dir, "faiss.index")
        metadata_path = os.path.join(vectorstore_dir, "id_map.json")
        
        if not os.path.exists(index_path):
            print(f"[WARN] VectorStore not found: {index_path}")
            return
        
        # Load index
        self.index = faiss.read_index(index_path)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Convert to internal format
        self.documents = []
        for meta in metadata:
            text = f"{meta['name']} - {meta.get('description', '')}"
            self.documents.append({
                "text": text,
                "metadata": meta
            })
        
        print(f"[VectorStore] Loaded {len(self.documents)} documents from {vectorstore_dir}")

