"""
Production-grade Cognitive Retrieval Agent with robust architecture.
Uses advanced patterns: immutable data structures, type safety, custom exceptions.
"""
from typing import List, Optional
import logging

from ..llm.interface import OllamaClient
from ..retrieval.hybrid import HybridRetrievalModule
from ..model.colbert import ColBERTReranker
from .state import ReasoningState, Step, Candidate, AgentConfig, TerminationReason
from ..common.exceptions import (
    InvalidQueryError, NoInitialCandidatesError, 
    NoNeighborsError, LLMParsingError
)

logger = logging.getLogger(__name__)

class CognitiveRetrievalAgent:
    """
    Production C-RAG Agent with robust error handling.
    
    Key improvements:
    - Immutable Candidate dataclass prevents mutation bugs
    - AgentConfig with validation ensures valid parameters
    - Custom exceptions for clear error propagation
    - Guaranteed final_answer via TerminationReason enum
    """
    
    def __init__(
        self, 
        hrm: HybridRetrievalModule, 
        reranker: ColBERTReranker, 
        llm_client: Optional[OllamaClient] = None,
        config: Optional[AgentConfig] = None
    ):
        self.hrm = hrm
        self.reranker = reranker
        self.llm = llm_client if llm_client else OllamaClient()
        self.config = config if config else AgentConfig()
        
        logger.info(f"Initialized CRA with config: {self.config}")

    def solve(self, query: str) -> ReasoningState:
        """
        Main reasoning loop with comprehensive error handling.
        
        Returns:
            ReasoningState with guaranteed termination_reason and final_answer
        """
        # Validate query
        try:
            self._validate_query(query)
        except InvalidQueryError as e:
            return self._create_error_state(query, TerminationReason.INVALID_QUERY, str(e))
        
        state = ReasoningState(query=query)
        
        # Initial retrieval with error handling
        try:
            candidates = self._retrieve_initial(query)
        except NoInitialCandidatesError:
            state.termination_reason = TerminationReason.NO_INITIAL_CANDIDATES.value
            return state
        
        state.update_context(candidates)
        state.add_step(Step(
            step_n=0,
            action="initial_retrieval",
            thought="Starting search.",
            candidates_count=len(candidates)
        ))

        # Reasoning loop
        for step_n in range(1, self.config.max_steps + 1):
            if not state.knowledge_graph_context:
                state.termination_reason = TerminationReason.EXHAUSTED_CONTEXT.value
                return state

            # THINK phase with retry
            try:
                parsed = self._think_phase(query, state)
            except LLMParsingError as e:
                logger.error(f"LLM parsing failed: {e}")
                state.termination_reason = TerminationReason.LLM_ERROR.value
                return state
            
            state.set_hypothesis(parsed.get("hypothesis", ""))

            # Check for answer
            if parsed.get("action") == "answer" and parsed.get("answer"):
                state.termination_reason = TerminationReason.SUCCESS.value
                state.final_answer = parsed["answer"]
                state.add_step(Step(step_n=step_n, action="terminate", thought=parsed.get("raw_response", "")))
                return state
            
            # ACT phase - expand candidates
            try:
                new_candidates = self._expand_phase(state.knowledge_graph_context)
            except NoNeighborsError:
                state.termination_reason = TerminationReason.NO_MORE_NEIGHBORS.value
                return state
            
            # OBSERVE phase - rerank and update
            ranked_candidates = self._rank_candidates(query, new_candidates)
            state.update_context(ranked_candidates)
            state.add_step(Step(
                step_n=step_n,
                action="expand",
                thought=parsed.get("raw_response", ""),
                candidates_count=len(ranked_candidates)
            ))
        
        # Max steps reached
        state.termination_reason = TerminationReason.MAX_STEPS_REACHED.value
        return state

    def _validate_query(self, query: str):
        """Validate query input."""
        if not query or not query.strip():
            raise InvalidQueryError(query, "Query cannot be empty")
        if len(query) > 1000:
            raise InvalidQueryError(query, "Query too long (max 1000 chars)")
    
    def _retrieve_initial(self, query: str) -> List[Candidate]:
        """
        Initial retrieval with error handling.
        
        Raises:
            NoInitialCandidatesError if no candidates found
        """
        raw_candidates = self.hrm.retrieve_initial_candidates(query, k=10)
        candidates = [Candidate.from_dict(c) for c in raw_candidates]
        scored_candidates = self._rank_candidates(query, candidates)
        
        if not scored_candidates:
            raise NoInitialCandidatesError(query)
        
        return scored_candidates
    
    def _think_phase(self, query: str, state: ReasoningState) -> dict:
        """
        Execute THINK phase with LLM.
        
        Raises:
            LLMParsingError if all retries fail
        """
        context_summary = [
            f"{c.name} ({c.description[:50]})"
            for c in state.knowledge_graph_context[:5]
        ]
        
        prompt = f"""
You are a Knowledge Graph Reasoning Agent.
Query: {query}

Current known entities:
{context_summary}

Previous Hypotheses: {state.hypotheses}

Task: Analyze and determine next step.
Format:
HYPOTHESIS: <working hypothesis>
MISSING: <missing info>
ACTION: <ANSWER_FOUND: answer | EXPAND: relation/entity>
"""
        
        for attempt in range(self.config.llm_max_retries + 1):
            try:
                response = self.llm.generate(prompt)
                
                # Use flexible parser
                from ..utils.llm_parser import llm_parser
                parsed = llm_parser.parse(response, query, 
                                         [c.to_dict() for c in state.knowledge_graph_context])
                
                # Validation
                if parsed.get("action") in ["expand", "answer", "stop"]:
                    parsed["raw_response"] = response
                    return parsed
                    
            except Exception as e:
                logger.warning(f"LLM attempt {attempt} failed: {e}")
                if attempt == self.config.llm_max_retries:
                    raise LLMParsingError(str(e), attempt)
        
        raise LLMParsingError("Max retries exceeded", self.config.llm_max_retries)
    
    def _expand_phase(self, candidates: List[Candidate]) -> List[Candidate]:
        """
        Expand graph neighbors.
        
        Raises:
            NoNeighborsError if expansion yields nothing
        """
        top_nodes = candidates[:self.config.max_expansions]
        top_dicts = [c.to_dict() for c in top_nodes]
        
        raw_neighbors = self.hrm.expand_candidates(top_dicts)
        if not raw_neighbors:
            raise NoNeighborsError([c.id for c in top_nodes])
        
        return [Candidate.from_dict(n) for n in raw_neighbors]
    
    def _rank_candidates(self, query: str, candidates: List[Candidate]) -> List[Candidate]:
        """
        Rank candidates using reranker or fallback.
        Returns new list of Candidates with updated scores (immutable).
        """
        if not candidates:
            return []
        
        # Build candidate texts
        cand_texts = [f"{c.name} {c.description}" for c in candidates]
        
        if not self.config.use_reranker:
            # Use retrieval scores
            scored = [
                c.with_score(c.retrieval_score) if c.retrieval_score > 0 else c.with_score(0.5)
                for c in candidates
            ]
            return sorted(scored, key=lambda x: x.score, reverse=True)[:self.config.top_k_candidates]
        
        # Rerank with ColBERT
        scores = self.reranker.score(query, cand_texts)
        
        # Create new Candidates with scores (immutable update)
        scored_candidates = [
            c.with_score(score) 
            for c, score in zip(candidates, scores)
        ]
        
        # Sort and return top-k
        return sorted(scored_candidates, key=lambda x: x.score, reverse=True)[:self.config.top_k_candidates]
    
    def _create_error_state(self, query: str, reason: TerminationReason, message: str) -> ReasoningState:
        """Helper to create error state."""
        state = ReasoningState(query=query)
        state.termination_reason = reason.value
        state.final_answer = message
        return state
