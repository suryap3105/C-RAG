"""
Example: Basic Retrieval
"""
from crag import (
    GraphEngine, 
    FaissVectorStore, 
    NeuroHybridRetrievalModule,
    QueryGraphGenerator,
    NeuralSubgraphMatcher,
    ColBERTPartitionRouter,
    create_llm_client
)

# Initialize components
graph_engine = GraphEngine()
graph_engine.load("data/preprocessed_graph.pt")

vector_store = FaissVectorStore()
vector_store.load("data/vector_index")

llm = create_llm_client(provider='mock')
query_gen = QueryGraphGenerator(llm)
matcher = NeuralSubgraphMatcher()
router = ColBERTPartitionRouter()
router.load("data/colbert_matrices.pt")

# Build pipeline
pipeline = NeuroHybridRetrievalModule(
    vector_store=vector_store,
    graph_engine=graph_engine,
    query_gen=query_gen,
    neural_matcher=matcher,
    colbert_router=router
)

# Query
results = pipeline.retrieve("What movies did Christopher Nolan direct?", k=10)

for i, result in enumerate(results, 1):
    print(f"{i}. {result['text']} (score: {result['score']:.4f})")
