"""
Custom exceptions for C-RAG system.
Provides clear error hierarchy and context.
"""

class CRAGException(Exception):
    """Base exception for all C-RAG errors."""
    pass

class RetrievalException(CRAGException):
    """Raised when retrieval fails."""
    pass

class NoInitialCandidatesError(RetrievalException):
    """Raised when initial retrieval returns no candidates."""
    
    def __init__(self, query: str):
        self.query = query
        super().__init__(f"No initial candidates found for query: {query[:50]}...")

class NoNeighborsError(RetrievalException):
    """Raised when graph expansion returns no neighbors."""
    
    def __init__(self, node_ids: list):
        self.node_ids = node_ids
        super().__init__(f"No neighbors found for nodes: {node_ids[:3]}...")

class LLMException(CRAGException):
    """Raised when LLM processing fails."""
    pass

class LLMParsingError(LLMException):
    """Raised when LLM response cannot be parsed."""
    
    def __init__(self, response: str, attempt: int):
        self.response = response
        self.attempt = attempt
        super().__init__(f"Failed to parse LLM response (attempt {attempt}): {response[:100]}...")

class LLMTimeoutError(LLMException):
    """Raised when LLM takes too long to respond."""
    
    def __init__(self, timeout: float):
        self.timeout = timeout
        super().__init__(f"LLM timeout after {timeout}s")

class ValidationException(CRAGException):
    """Raised when data validation fails."""
    pass

class InvalidQueryError(ValidationException):
    """Raised when query is invalid."""
    
    def __init__(self, query: str, reason: str):
        self.query = query
        self.reason = reason
        super().__init__(f"Invalid query: {reason}")

class ConfigurationError(CRAGException):
    """Raised when configuration is invalid."""
    pass
