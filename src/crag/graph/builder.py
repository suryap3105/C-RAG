import json
import re
import networkx as nx
from typing import List, Dict, Set, Tuple, Any
from collections import Counter
import logging

class SimpleGraphBuilder:
    """
    Heuristic Knowledge Graph Builder.
    Constructs a co-occurrence graph from text without heavy NLP dependencies.
    """
    def __init__(self, min_freq: int = 2, window_size: int = 1):
        self.min_freq = min_freq
        self.window_size = window_size
        self.graph = nx.Graph()
        self.entity_counts = Counter()
        
    def extract_entities(self, text: str) -> List[str]:
        """
        Extract entities using capitalization heuristic.
        Finds sequences of capitalized words.
        """
        # Remove common stopwords/noise (very basic)
        ignore = {'The', 'A', 'An', 'This', 'That', 'These', 'Those', 'It', 'He', 'She', 'They'}
        
        entities = []
        # Regex for capitalized phrases: (Capitalized Word) + (maybe more capitalized words)
        # Excludes words at start of sentence if they are simple stopwords
        matches = re.finditer(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        
        for m in matches:
            ent = m.group()
            if len(ent) > 2 and ent not in ignore:
                entities.append(ent)
        
        return entities

    def process_documents(self, documents: List[Dict[str, Any]]):
        """
        Process list of documents (dicts with 'context' or 'text').
        1. Count entity frequencies
        2. Filter rare entities
        3. Build edges
        """
        print(f"[GraphBuilder] Processing {len(documents)} documents...")
        
        # Pass 1: Entity Counting
        doc_entities = []
        for doc in documents:
            text = doc.get('context', '') or doc.get('text', '')
            if not text: continue
            
            ents = self.extract_entities(text)
            self.entity_counts.update(ents)
            doc_entities.append(ents)
            
        print(f"[GraphBuilder] Found {len(self.entity_counts)} unique raw entities")
        
        # Filter rare entities (noise reduction)
        valid_entities = {e for e, c in self.entity_counts.items() if c >= self.min_freq}
        print(f"[GraphBuilder] Retained {len(valid_entities)} entities (min_freq={self.min_freq})")
        
        # Pass 2: Graph Construction
        edge_counts = Counter()
        
        for ents in doc_entities:
            # Filter to valid only
            filtered = [e for e in ents if e in valid_entities]
            
            # Create cliques within document (Co-occurrence)
            # Simple approach: Connect all entities in same context
            # Or sliding window. For short contexts (SQuAD), full clique is okay but dense.
            # Let's use sliding window of 3 for density control
            
            for i in range(len(filtered)):
                for j in range(i + 1, min(i + 1 + 5, len(filtered))):
                    u, v = sorted((filtered[i], filtered[j]))
                    edge_counts[(u, v)] += 1

        # Add nodes and edges
        for ent in valid_entities:
            self.graph.add_node(ent, count=self.entity_counts[ent])
            
        for (u, v), weight in edge_counts.items():
            self.graph.add_edge(u, v, weight=weight)
            
        print(f"[GraphBuilder] Built graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

    def save_graph(self, path: str):
        """Save as adjacency list JSON."""
        data = nx.adjacency_data(self.graph)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        print(f"[GraphBuilder] Saved to {path}")

if __name__ == "__main__":
    import sys
    # Example usage for testing
    text = "Barack Obama was born in Hawaii. He served as President of the United States."
    builder = SimpleGraphBuilder(min_freq=1)
    builder.process_documents([{'context': text}, {'context': "Hawaii is a state in the Western United States."}])
    print("Nodes:", builder.graph.nodes())
