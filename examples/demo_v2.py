
import sys
import os

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from crag.agent.cra import CognitiveRetrievalAgent
from crag.retrieval.hybrid import HybridRetrievalModule
from crag.retrieval.vector_store import FaissVectorStore
from crag.graph.wikidata import WikidataKG
from crag.model.colbert import ColBERTReranker
from crag.llm.interface import MockLLMClient
from crag.agent.state import ReasoningState

def print_separator(title=""):
    print(f"\n{'='*20} {title} {'='*20}")

def run_demo():
    print_separator("INITIALIZING C-RAG v2")
    
    # 1. Setup Components
    kg = WikidataKG()
    vs = FaissVectorStore() # Mock
    hrm = HybridRetrievalModule(kg, vs)
    reranker = ColBERTReranker() # Will use Mock if transformers missing
    llm = MockLLMClient()
    
    # Enable Full Architecture (Reranker + Reasoning)
    agent = CognitiveRetrievalAgent(hrm, reranker, llm_client=llm, use_reranker=True)
    
    query = "Who directed the movie Inception?"
    print(f"QUERY: {query}\n")

    # 2. Run Solve
    print_separator("STARTING AGENT LOOP")
    state: ReasoningState = agent.solve(query)
    
    # 3. Inspect Trajectory
    print_separator("REASONING TRAJECTORY")
    
    for step in state.trajectory:
        print(f"\n[STEP {step.step_n}] ACTION: {step.action.upper()}")
        print(f"  > Thought: {step.thought.strip()}")
        print(f"  > Candidates retrieved: {step.candidates_count}")
        
    print_separator("FINAL RESULT")
    print(f"TERMINATION: {state.termination_reason}")
    print(f"ANSWER:      {state.final_answer}")
    
    print_separator("INTROSPECTION DATA (Visualizing internal scores)")
    if state.knowledge_graph_context:
        print("Top Context Nodes:")
        for node in state.knowledge_graph_context[:3]:
            source = node.get('retrieval_source', 'N/A')
            score = node.get('score', 0.0)
            print(f"  - {node.get('name')} | Source: {source} | Score: {score:.4f}")

if __name__ == "__main__":
    run_demo()
