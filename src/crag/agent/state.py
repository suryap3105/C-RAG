"""
Robust data structures for C-RAG system.
Uses dataclasses for type safety and immutability.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

class TerminationReason(Enum):
    """Enum for reasoning termination states."""
    SUCCESS = "success"
    MAX_STEPS_REACHED = "max_steps_reached"
    EXHAUSTED_CONTEXT = "exhausted_context"
    NO_INITIAL_CANDIDATES = "no_initial_candidates"
    NO_MORE_NEIGHBORS = "no_more_neighbors"
    LLM_ERROR = "llm_error"
    INVALID_QUERY = "invalid_query"

@dataclass(frozen=True)
class Candidate:
    """
    Immutable candidate representation.
    Frozen dataclass prevents accidental mutations.
    """
    id: str
    name: str
    description: str
    score: float = 0.0
    retrieval_score: float = 0.0
    source: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Candidate':
        """
        Factory method to create Candidate from dict.
        Handles missing fields gracefully.
        """
        metadata = data.get('metadata', {})
        return cls(
            id=data.get('id', data.get('node_id', 'unknown')),
            name=data.get('name') or metadata.get('name', 'Unknown'),
            description=data.get('description') or metadata.get('description', ''),
            score=data.get('score', 0.0),
            retrieval_score=data.get('retrieval_score', 0.0),
            source=data.get('source', 'unknown'),
            metadata=metadata.copy()  # Defensive copy
        )
    
    def with_score(self, score: float) -> 'Candidate':
        """Return new Candidate with updated score (immutable update)."""
        return Candidate(
            id=self.id,
            name=self.name,
            description=self.description,
            score=score,
            retrieval_score=self.retrieval_score,
            source=self.source,
            metadata=self.metadata
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'score': self.score,
            'retrieval_score': self.retrieval_score,
            'source': self.source,
            'metadata': self.metadata
        }

@dataclass
class AgentConfig:
    """
    Configuration with validation.
    """
    max_steps: int = 3
    max_expansions: int = 5
    top_k_candidates: int = 5
    use_reranker: bool = True
    llm_max_retries: int = 2
    
    def __post_init__(self):
        """Validate configuration on initialization."""
        if self.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {self.max_steps}")
        if self.max_expansions < 1:
            raise ValueError(f"max_expansions must be >= 1, got {self.max_expansions}")
        if self.top_k_candidates < 1:
            raise ValueError(f"top_k_candidates must be >= 1, got {self.top_k_candidates}")
        
        # Clamp to reasonable ranges
        self.max_steps = min(self.max_steps, 10)
        self.max_expansions = min(self.max_expansions, 20)
        self.top_k_candidates = min(self.top_k_candidates, 50)

@dataclass
class Step:
    """Reasoning step with validation."""
    step_n: int
    action: str
    thought: str
    candidates_count: int = 0
    
    def __post_init__(self):
        if self.step_n < 0:
            raise ValueError(f"step_n must be non-negative, got {self.step_n}")
        if self.candidates_count < 0:
            raise ValueError(f"candidates_count must be non-negative")

class ReasoningState:
    """
    Mutable state container for agent reasoning.
    Ensures final_answer is always set when termination_reason is set.
    """
    def __init__(self, query: str):
        self.query = query
        self.hypotheses: List[str] = []
        self.knowledge_graph_context: List[Candidate] = []
        self.path: List[Step] = []
        self._termination_reason: Optional[TerminationReason] = None
        self._final_answer: str = ""
    
    @property
    def termination_reason(self) -> Optional[str]:
        """Get termination reason as string."""
        return self._termination_reason.value if self._termination_reason else None
    
    @termination_reason.setter
    def termination_reason(self, reason: str):
        """Set termination reason with automatic answer assignment."""
        try:
            self._termination_reason = TerminationReason(reason)
        except ValueError:
            raise ValueError(f"Invalid termination reason: {reason}")
        
        # Automatically provide default answer if not already set
        if not self._final_answer:
            self._final_answer = self._get_default_answer(self._termination_reason)
    
    @property
    def final_answer(self) -> str:
        return self._final_answer
    
    @final_answer.setter
    def final_answer(self, answer: str):
        self._final_answer = answer
    
    def _get_default_answer(self, reason: TerminationReason) -> str:
        """Get default answer based on termination reason."""
        defaults = {
            TerminationReason.SUCCESS: "",  # Explicit answer required
            TerminationReason.MAX_STEPS_REACHED: "Answer inference incomplete. Please refine query.",
            TerminationReason.EXHAUSTED_CONTEXT: "No relevant context found. Unable to answer.",
            TerminationReason.NO_INITIAL_CANDIDATES: "No relevant information found.",
            TerminationReason.NO_MORE_NEIGHBORS: "No further information available. Current evidence insufficient.",
            TerminationReason.LLM_ERROR: "LLM processing failed. Unable to complete reasoning.",
            TerminationReason.INVALID_QUERY: "Query cannot be empty."
        }
        return defaults.get(reason, "Unable to determine answer.")
    
    def update_context(self, candidates: List[Candidate]):
        """Update context with new candidates."""
        self.knowledge_graph_context = candidates
    
    def add_step(self, step: Step):
        """Add reasoning step."""
        self.path.append(step)
    
    def set_hypothesis(self, hypothesis: str):
        """Add hypothesis to history."""
        if hypothesis:
            self.hypotheses.append(hypothesis)
