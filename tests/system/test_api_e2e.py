from fastapi.testclient import TestClient
from crag.api.server import app
from unittest.mock import patch

client = TestClient(app)

def test_e2e_query_flow():
    """
    System Test: Hits the API endpoint.
    Mocks the heavy internal Agent to ensure we rely on API contract, not internal logic speed.
    """
    with patch("crag.api.server.agent") as mock_agent:
        # Setup mock return from the global 'agent' instance in server.py
        mock_agent.solve.return_value = {
            "answer": "System Test Answer",
            "path": [{"step": 1, "action": "test", "candidates": 5}]
        }
        
        payload = {"query": "sanity check"}
        response = client.post("/query", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "sanity check"
        assert data["answer"] == "System Test Answer"
        assert len(data["path"]) == 1
