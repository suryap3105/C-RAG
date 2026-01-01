import pytest
from unittest.mock import MagicMock
from crag.agent.cra import CognitiveRetrievalAgent
from crag.llm.interface import MockLLMClient

def test_agent_reasoning_loop():
    """
    Integration Test: Verifies the Agent -> HRM -> LLM loop.
    Uses a MockLLM to control the 'Think' step but runs the real Agent logic.
    """
    # Mock HRM to return specific candidates
    mock_hrm = MagicMock()
    mock_hrm.retrieve_initial_candidates.return_value = [{"name": "StartNode", "metadata": {"id": "A"}}]
    mock_hrm.expand_candidates.return_value = [{"name": "EndNode", "metadata": {"id": "B"}}]
    
    # Mock Reranker to just pass through
    mock_reranker = MagicMock()
    mock_reranker.score.return_value = [0.9]
    
    # Use deterministic MockLLM
    class ControlledLLM:
        def generate(self, prompt, system_prompt=""):
            if "StartNode" in prompt:
                return "EXPAND: B"
            if "EndNode" in prompt:
                 return "ANSWER_FOUND: The Answer"
            return "EXPAND"
            
    agent = CognitiveRetrievalAgent(mock_hrm, mock_reranker, llm_client=ControlledLLM())
    
    result = agent.solve("Query")
    
    assert result["answer"] == "The Answer"
    assert len(result["path"]) >= 2 # At least initial + 1 expansion
    # Verify interactions
    mock_hrm.retrieve_initial_candidates.assert_called()
    mock_hrm.expand_candidates.assert_called()
