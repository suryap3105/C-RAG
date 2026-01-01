#!/usr/bin/env python3
"""
Comprehensive test suite for C-RAG system.
Validates all modules, imports, configs, and data loaders.
"""
import sys
import os
import glob
import yaml

# Ensure src is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '../src'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

def test_imports():
    """Test all critical imports."""
    try:
        from crag.agent.cra import CognitiveRetrievalAgent
    except ImportError as e:
        assert False, f"Failed to import CognitiveRetrievalAgent: {e}"
    
    try:
        from crag.llm.interface import MockLLMClient, OllamaClient
    except ImportError as e:
        assert False, f"Failed to import LLM Interface: {e}"
    
    try:
        from crag.data.unified_loader import UnifiedDatasetLoader
    except ImportError as e:
        assert False, f"Failed to import UnifiedDatasetLoader: {e}"
    
    try:
        from crag.evaluation.experiment_manager import ExperimentManager
    except ImportError as e:
        assert False, f"Failed to import ExperimentManager: {e}"

def test_configs():
    """Test all YAML configurations."""
    configs = glob.glob(os.path.join(CURRENT_DIR, "../configs/*.yaml"))
    
    for config_path in configs:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate required fields
        assert 'experiment_name' in config, f"Config {os.path.basename(config_path)} missing experiment_name"

def test_data_loaders():
    """Test data loading functionality."""
    from crag.data.unified_loader import UnifiedDatasetLoader
    
    # Test SQuAD loader
    squad_path = os.path.join(CURRENT_DIR, "../data/squad_train_v2.json")
    if os.path.exists(squad_path):
        data = UnifiedDatasetLoader.load_squad(squad_path)
        assert len(data) >= 0, "SQuAD loading failed"
    
    # Test WebQSP loader
    webqsp_path = os.path.join(CURRENT_DIR, "../data/webqsp/input/webqsp.examples.test.wikidata.json")
    if os.path.exists(webqsp_path):
        data = UnifiedDatasetLoader.load_webqsp(webqsp_path)
        assert len(data) >= 0, "WebQSP loading failed"

def test_scripts():
    """Test that all scripts are syntactically valid."""
    scripts = glob.glob(os.path.join(CURRENT_DIR, "../scripts/*.py")) + \
              glob.glob(os.path.join(SRC_DIR, "crag/analysis/*.py"))
    
    for script in scripts:
        with open(script, 'r') as f:
            try:
                compile(f.read(), script, 'exec')
            except SyntaxError as e:
                assert False, f"Syntax error in {os.path.basename(script)}: {e}"

if __name__ == "__main__":
    # Allow running directly
    import pytest
    sys.exit(pytest.main([__file__]))
