"""
C-RAG V3 Query Graph Generator
LLM-based Query Parsing with Validation
"""
import json
import logging
import torch
from typing import Dict, List, Optional, Tuple
from torch_geometric.data import Data
from dataclasses import dataclass

from ..llm.interface import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class QueryGraphSchema:
    """Schema defining valid node types and relations."""
    node_types: List[str]
    edge_types: List[str]
    
    @classmethod
    def default(cls) -> 'QueryGraphSchema':
        return cls(
            node_types=['Person', 'Movie', 'Actor', 'Director', 'Company', 
                       'Location', 'Date', 'Concept', 'Entity', '?'],
            edge_types=['DIRECTED', 'ACTED_IN', 'PRODUCED', 'LOCATED_IN',
                       'FOUNDED', 'WORKS_FOR', 'RELATED_TO', 'CONNECTED_TO',
                       'PART_OF', 'HAS_PROPERTY']
        )


class QueryGraphGenerator:
    """
    Production Query Graph Generator.
    Converts natural language queries into PyG graph structures.
    """
    def __init__(self, llm_client: LLMClient, schema: Optional[QueryGraphSchema] = None,
                 node_encoder_name: str = "intfloat/e5-base-v2"):
        self.llm = llm_client
        self.schema = schema or QueryGraphSchema.default()
        self.node_encoder_name = node_encoder_name
        self._encoder = None
        
        self.prompt_template = """You are a Query Graph Parser that extracts structured knowledge from questions.

TASK: Convert the question into a knowledge graph query structure.

VALID NODE TYPES: {node_types}
VALID EDGE TYPES: {edge_types}

OUTPUT FORMAT (JSON only, no explanation):
{{
    "nodes": [
        {{"id": "entity_name", "type": "TYPE"}},
        {{"id": "?", "type": "TARGET_TYPE"}}
    ],
    "edges": [
        {{"src": "entity_name", "dst": "?", "relation": "RELATION_TYPE"}}
    ]
}}

RULES:
1. Use "?" for unknown/target entities
2. Extract all entities and relationships mentioned
3. Use only valid types from the lists above
4. If unsure, use "Entity" and "RELATED_TO"

QUESTION: {query}

JSON OUTPUT:"""

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.node_encoder_name)
        return self._encoder
        
    def parse(self, query: str) -> Data:
        """
        Parse natural language query into PyG Data.
        """
        prompt = self.prompt_template.format(
            node_types=', '.join(self.schema.node_types),
            edge_types=', '.join(self.schema.edge_types),
            query=query
        )
        
        try:
            response = self.llm.generate(prompt)
            
            # Clean response
            response = response.strip()
            if response.startswith('```'):
                response = response.split('```')[1]
                if response.startswith('json'):
                    response = response[4:]
            response = response.strip()
            
            graph_json = json.loads(response)
            
            return self._json_to_pyg(graph_json, query)
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}. Using fallback parsing.")
            return self._fallback_parse(query)
        except Exception as e:
            logger.error(f"Query parsing error: {e}")
            return self._empty_graph()
            
    def _json_to_pyg(self, graph_json: Dict, query: str) -> Data:
        """Convert JSON graph to PyG Data."""
        nodes = graph_json.get('nodes', [])
        edges = graph_json.get('edges', [])
        
        if not nodes:
            return self._fallback_parse(query)
            
        # Build node mapping
        node_map = {}
        node_texts = []
        node_types = []
        
        for idx, node in enumerate(nodes):
            node_id = node.get('id', f'node_{idx}')
            node_map[node_id] = idx
            node_texts.append(node_id if node_id != '?' else 'unknown target')
            node_types.append(node.get('type', 'Entity'))
            
        # Get node embeddings
        encoder = self._get_encoder()
        embeddings = encoder.encode(
            [f"passage: {t}" for t in node_texts], 
            convert_to_numpy=True
        )
        x = torch.tensor(embeddings, dtype=torch.float32)
        
        # Build edge index
        src_list, dst_list = [], []
        edge_types = []
        
        for edge in edges:
            src_id = edge.get('src')
            dst_id = edge.get('dst')
            
            if src_id in node_map and dst_id in node_map:
                src_list.append(node_map[src_id])
                dst_list.append(node_map[dst_id])
                edge_types.append(edge.get('relation', 'RELATED_TO'))
                
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        
        # Build Data object
        data = Data(x=x, edge_index=edge_index, num_nodes=len(nodes))
        
        # Store metadata
        data.node_map = node_map
        data.node_types = node_types
        data.node_texts = node_texts
        data.edge_types = edge_types
        data.raw_json = graph_json
        
        logger.info(f"Parsed query graph: {len(nodes)} nodes, {len(edges)} edges")
        return data
        
    def _fallback_parse(self, query: str) -> Data:
        """
        Simple NER-based fallback when LLM parsing fails.
        """
        logger.info("Using fallback NER parsing")
        
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(query)
            
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            
            if not entities:
                # Extract nouns as entities
                entities = [(token.text, 'Entity') for token in doc if token.pos_ == 'NOUN']
                
            if not entities:
                return self._empty_graph()
                
            # Add target node
            entities.append(('?', 'Target'))
            
            # Get embeddings
            encoder = self._get_encoder()
            texts = [e[0] for e in entities]
            embeddings = encoder.encode([f"passage: {t}" for t in texts], convert_to_numpy=True)
            x = torch.tensor(embeddings, dtype=torch.float32)
            
            # Connect all entities to target
            target_idx = len(entities) - 1
            src_list = list(range(target_idx))
            dst_list = [target_idx] * target_idx
            
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
            
            data = Data(x=x, edge_index=edge_index, num_nodes=len(entities))
            data.node_texts = texts
            data.node_types = [e[1] for e in entities]
            
            return data
            
        except Exception as e:
            logger.error(f"Fallback parsing failed: {e}")
            return self._empty_graph()
            
    def _empty_graph(self) -> Data:
        """Return empty graph."""
        return Data(
            x=torch.zeros((1, 768)),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            num_nodes=1
        )
        
    def validate_graph(self, data: Data) -> Tuple[bool, List[str]]:
        """
        Validate parsed graph against schema.
        """
        errors = []
        
        if not hasattr(data, 'node_types'):
            return True, []
            
        for nt in data.node_types:
            if nt not in self.schema.node_types:
                errors.append(f"Invalid node type: {nt}")
                
        if hasattr(data, 'edge_types'):
            for et in data.edge_types:
                if et not in self.schema.edge_types:
                    errors.append(f"Invalid edge type: {et}")
                    
        return len(errors) == 0, errors
