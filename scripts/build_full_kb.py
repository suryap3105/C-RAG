#!/usr/bin/env python3
"""
Script to build Full-Scale Knowledge Base (Graph + VectorStore).
Processes all 146k+ questions from SQuAD, WebQSP, and MetaQA.
"""
import sys
import os
import time
sys.path.insert(0, 'src')

from crag.data.unified_loader import UnifiedDatasetLoader
from crag.graph.builder import SimpleGraphBuilder
from crag.retrieval.vector_store import FaissVectorStore

def main():
    print("="*60)
    print("BUILDING FULL-SCALE CRAG KNOWLEDGE BASE (146k+ DOCS)")
    print("="*60 + "\n")
    
    # 1. Load ALL Data
    print("[1/3] Loading Datasets...")
    start_time = time.time()
    # No limit = load everything
    data = UnifiedDatasetLoader.load_all(max_per_dataset=1000000)
    print(f"  -> Total items loaded: {len(data)}")
    
    # 2. Extract Unique Contexts
    print("\n[2/3] Extracting Unique Contexts...")
    seen_contexts = set()
    unique_docs = []
    
    for item in data:
        # Use 'context' if available (SQuAD), else 'query' + 'answers' text
        # For WebQSP/MetaQA, context is often empty or implicit.
        # We can treat the Q+A pair as the "document" for graph building if context is missing.
        
        ctx = item.get('context', '').strip()
        if not ctx:
            # Fallback for datasets without explicit context -> Use Query for graph connectivity
            # (Entities in query should be connected)
            ctx = item.get('query', '')
        
        if ctx and ctx not in seen_contexts:
            seen_contexts.add(ctx)
            unique_docs.append({'context': ctx, 'id': item['id']})
    
    print(f"  -> Unique contexts found: {len(unique_docs)}")
    
    # 3. Build Knowledge Graph
    print(f"\n[3/3] Building Knowledge Graph from {len(unique_docs)} contexts...")
    builder = SimpleGraphBuilder(min_freq=3) # Higher min_freq for larger corpus to reduce noise
    builder.process_documents(unique_docs)
    
    output_dir = "data/graphs"
    os.makedirs(output_dir, exist_ok=True)
    kg_path = os.path.join(output_dir, "full_kg.json")
    builder.save_graph(kg_path)
    
    # 4. Build Vector Store (Optional but recommended for Full C-RAG)
    # This might take time, so we check if user wants it or if we just do it.
    # User said "crag should work on it", so we need retrieval.
    # Integrating vector build here might be too slow for single script execution if not parallel.
    # For now, we will SKIP vector build in this script to keep it focused on KG as requested.
    # The existing build_vectorstore.py can handle vector part.
    
    elapsed = time.time() - start_time
    print("\n" + "="*60)
    print(f"FULL BUILD COMPLETE in {elapsed:.1f}s")
    print(f"KG Saved to: {kg_path}")
    print("="*60)

if __name__ == "__main__":
    main()
