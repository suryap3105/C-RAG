"""
pytest configuration and fixtures
"""
import pytest
import torch
import logging
from pathlib import Path

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create temporary directory for test data."""
    return tmp_path_factory.mktemp("test_data")


@pytest.fixture(scope="session")
def device():
    """Get compute device for tests."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    torch.manual_seed(42)
    import numpy as np
    np.random.seed(42)
    import random
    random.seed(42)


@pytest.fixture
def mock_graph_data():
    """Create mock graph data for testing."""
    from torch_geometric.data import Data
    
    x = torch.randn(10, 768)
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5, 0]
    ], dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index, num_nodes=10)


def pytest_addoption(parser):
    """Add custom pytest options."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests"
    )
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests"
    )


def pytest_configure(config):
    """Add custom markers."""
    config.addinivalue_line(
        "markers", "slow: mark test as slow to run"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on options."""
    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    skip_integration = pytest.mark.skip(reason="need --run-integration option to run")
    skip_gpu = pytest.mark.skip(reason="GPU not available")
    
    for item in items:
        if "slow" in item.keywords and not config.getoption("--run-slow"):
            item.add_marker(skip_slow)
            
        if "integration" in item.keywords and not config.getoption("--run-integration"):
            item.add_marker(skip_integration)
            
        if "gpu" in item.keywords and not torch.cuda.is_available():
            item.add_marker(skip_gpu)
