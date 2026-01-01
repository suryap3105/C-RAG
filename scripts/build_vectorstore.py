"""
VectorStore Population Pipeline
Embeds KG entities and builds FAISS index for vector baseline + HRM prefiltering.
"""

import os
import json
import argparse
import hashlib
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

class VectorStoreBuilder:
    """Builds FAISS vector store from KG entities."""
    
    def __init__(self, encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.encoder_name = encoder_name
        print(f"[VectorStoreBuilder] Loading encoder: {encoder_name}")
        self.encoder = SentenceTransformer(encoder_name)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        print(f"[VectorStoreBuilder] Embedding dimension: {self.embedding_dim}")
    
    def build_corpus(self, kg_file: str) -> tuple[List[str], List[Dict]]:
        """
        Extract text corpus from KG.
        
        Returns:
            texts: List of text snippets to embed
            metadata: List of metadata dicts (id, name, description)
        """
        # For demo: use hardcoded entities
        # In production, load from actual KG file
        entities = [
            {"id": "Q190050", "name": "Christopher Nolan", "description": "British-American film director"},
            {"id": "Q25188", "name": "Inception", "description": "2010 science fiction film"},
            {"id": "Q2263", "name": "Leonardo DiCaprio", "description": "American actor"},
            {"id": "Q2263_movie1", "name": "Titanic", "description": "1997 romantic drama film"},
            {"id": "Q2263_movie2", "name": "The Wolf of Wall Street", "description": "2013 biographical black comedy crime film"},
            {"id": "Q2263_movie3", "name": "The Revenant", "description": "2015 survival drama film"},
            {"id": "Q2263_movie4", "name": "Catch Me If You Can", "description": "2002 biographical crime film"},
            {"id": "Q2263_movie5", "name": "Shutter Island", "description": "2010 psychological thriller film"},
            {"id": "Q37079", "name": "Tom Hanks", "description": "American actor and filmmaker"},
            {"id": "Q134773", "name": "Forrest Gump", "description": "1994 American comedy-drama film"},
            {"id": "Q105825", "name": "Saving Private Ryan", "description": "1998 American war film"},
            {"id": "Q153882", "name": "The Green Mile", "description": "1999 American fantasy drama film"},
            {"id": "Q105825_dir", "name": "Steven Spielberg", "description": "American film director"},
        ]
        
        texts = []
        metadata = []
        
        for entity in entities:
            # Format: "name - description"
            text = f"{entity['name']} - {entity.get('description', '')}"
            texts.append(text)
            metadata.append(entity)
        
        print(f"[VectorStoreBuilder] Built corpus: {len(texts)} entities")
        return texts, metadata
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed texts using SentenceTransformer."""
        print(f"[VectorStoreBuilder] Embedding {len(texts)} texts...")
        embeddings = self.encoder.encode(texts, show_progress_bar=True)
        return np.array(embeddings).astype('float32')
    
    def build_index(self, embeddings: np.ndarray, use_ivf: bool = False) -> faiss.Index:
        """Build FAISS index."""
        n, d = embeddings.shape
        print(f"[VectorStoreBuilder] Building index: {n} vectors, dim={d}")
        
        if use_ivf and n > 1000:
            # IVF for larger datasets
            nlist = min(int(np.sqrt(n)), 100)
            quantizer = faiss.IndexFlatL2(d)
            index = faiss.IndexIVFFlat(quantizer, d, nlist)
            index.train(embeddings)
        else:
            # Flat index for small datasets
            index = faiss.IndexFlatL2(d)
        
        index.add(embeddings)
        print(f"[VectorStoreBuilder] Index built: {index.ntotal} vectors")
        return index
    
    def compute_checksum(self, texts: List[str]) -> str:
        """Compute checksum for versioning."""
        content = json.dumps({"encoder": self.encoder_name, "texts": texts}, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()
    
    def save(self, index: faiss.Index, metadata: List[Dict], output_dir: str, checksum: str):
        """Save index and metadata."""
        os.makedirs(output_dir, exist_ok=True)
        
        index_path = os.path.join(output_dir, "faiss.index")
        metadata_path = os.path.join(output_dir, "id_map.json")
        version_path = os.path.join(output_dir, "version.json")
        
        # Save FAISS index
        faiss.write_index(index, index_path)
        print(f"[VectorStoreBuilder] Saved index: {index_path}")
        
        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"[VectorStoreBuilder] Saved metadata: {metadata_path}")
        
        # Save version info
        version_info = {
            "encoder": self.encoder_name,
            "embedding_dim": self.embedding_dim,
            "num_vectors": index.ntotal,
            "checksum": checksum
        }
        with open(version_path, 'w') as f:
            json.dump(version_info, f, indent=2)
        print(f"[VectorStoreBuilder] Saved version: {version_path}")

def main():
    parser = argparse.ArgumentParser(description="Build VectorStore from KG")
    parser.add_argument("--dataset", type=str, default="metaqa", help="Dataset name")
    parser.add_argument("--encoder", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--kg-file", type=str, default=None, help="KG file (optional)")
    args = parser.parse_args()
    
    output_dir = args.output or f"data/{args.dataset}/vectorstore"
    
    builder = VectorStoreBuilder(encoder_name=args.encoder)
    
    # Build corpus
    texts, metadata = builder.build_corpus(args.kg_file)
    
    # Embed
    embeddings = builder.embed(texts)
    
    # Build index
    index = builder.build_index(embeddings)
    
    # Checksum
    checksum = builder.compute_checksum(texts)
    
    # Save
    builder.save(index, metadata, output_dir, checksum)
    
    print(f"\n[SUCCESS] VectorStore ready: {output_dir}")
    print(f"[SUCCESS] Use in FaissVectorStore.load('{output_dir}')")

if __name__ == "__main__":
    main()
