# Central Claim: Cognitive Retrieval Augmented Generation (C-RAG)

## Core Hypothesis
Current RAG systems are "stateless": they retrieve once and generate. They fail on multi-hop reasoning where the necessary context is not immediately available.

**C-RAG** introduces an *Epistemic Reasoner* that maintains a `ReasoningState`. It uses a "Think-Act-Observe" loop to dynamically expand the retrieval frontier in a Knowledge Graph until it has sufficient information to answer.

## Algorithm Definition
1. **Initial Retrieval**: Hybrid (Vector + KG).
2. **Cognitive Loop**:
    - **Think**: LLM analyzes current context vs query.
    - **Hypothesize**: Formulate missing information.
    - **Act**: Expand specific nodes in the KG.
    - **Observe**: Rank new candidates and update context.
3. **Termination**: When LLM outputs `ANSWER_FOUND` or budget exceeded.

## Measured Success
C-RAG achieves higher accuracy on multi-hop benchmarks (MetaQA-3hop) compared to static baselines/Vector RAG, efficiently trading off latency for reasoning depth (Pareto optimality).
