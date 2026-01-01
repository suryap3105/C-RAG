import pytest
from unittest.mock import MagicMock, patch
from crag.retrieval.hybrid import HybridRetrievalModule

def test_hrm_deduplication():
    """
    Unit Test: Verifies that HRM correctly deduplicates results based on ID.
    Mocks VectorStore and KG.
    """
    mock_kg = MagicMock()
    mock_vs = MagicMock()
    
    # Setup mock returns
    mock_vs.search.return_value = [
        {"text": "Doc A", "metadata": {"id": "1"}},
        {"text": "Doc A Duplicate", "metadata": {"id": "1"}}, # Duplicate ID
        {"text": "Doc B", "metadata": {"id": "2"}}
    ]
    
    hrm = HybridRetrievalModule(mock_kg, mock_vs)
    results = hrm.retrieve_initial_candidates("query")
    
    assert len(results) == 2
    ids = [r['metadata']['id'] for r in results]
    assert "1" in ids
    assert "2" in ids
    # Verify mock interaction
    mock_vs.search.assert_called_once()
