#!/usr/bin/env python3
"""
Comprehensive test suite for C-RAG system.
Validates all modules, imports, configs, and data loaders.
"""
import sys
import os
import traceback
sys.path.insert(0, 'src')

def test_imports():
    """Test all critical imports."""
    tests = []
    
    try:
        from crag.agent.cra import CognitiveRetrievalAgent
        tests.append(("CognitiveRetrievalAgent", "PASS"))
    except Exception as e:
        tests.append(("CognitiveRetrievalAgent", f"FAIL: {e}"))
    
    try:
        from crag.llm.interface import MockLLMClient, OllamaClient
        tests.append(("LLM Interface", "PASS"))
    except Exception as e:
        tests.append(("LLM Interface", f"FAIL: {e}"))
    
    try:
        from crag.data.unified_loader import UnifiedDatasetLoader
        tests.append(("UnifiedDatasetLoader", "PASS"))
    except Exception as e:
        tests.append(("UnifiedDatasetLoader", f"FAIL: {e}"))
    
    try:
        from crag.evaluation.experiment_manager import ExperimentManager
        tests.append(("ExperimentManager", "PASS"))
    except Exception as e:
        tests.append(("ExperimentManager", f"FAIL: {e}"))
    
    try:
        from crag.analysis.rerank_audit import main as rerank_main
        tests.append(("rerank_audit", "PASS"))
    except Exception as e:
        tests.append(("rerank_audit", f"FAIL: {e}"))
    
    return tests

def test_configs():
    """Test all YAML configurations."""
    import yaml
    import glob
    
    tests = []
    configs = glob.glob("configs/*.yaml")
    
    for config_path in configs:
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Validate required fields
            if 'experiment_name' in config:
                tests.append((os.path.basename(config_path), "PASS"))
            else:
                tests.append((os.path.basename(config_path), "WARN: No experiment_name"))
        except Exception as e:
            tests.append((os.path.basename(config_path), f"FAIL: {e}"))
    
    return tests

def test_data_loaders():
    """Test data loading functionality."""
    from crag.data.unified_loader import UnifiedDatasetLoader
    
    tests = []
    
    # Test SQuAD loader
    try:
        if os.path.exists("data/squad_train_v2.json"):
            data = UnifiedDatasetLoader.load_squad("data/squad_train_v2.json")
            if len(data) > 0:
                tests.append(("SQuAD Loader", f"PASS ({len(data)} questions)"))
            else:
                tests.append(("SQuAD Loader", "WARN: No data loaded"))
        else:
            tests.append(("SQuAD Loader", "SKIP: File not found"))
    except Exception as e:
        tests.append(("SQuAD Loader", f"FAIL: {e}"))
    
    # Test WebQSP loader
    try:
        if os.path.exists("data/webqsp/input/webqsp.examples.test.wikidata.json"):
            data = UnifiedDatasetLoader.load_webqsp("data/webqsp/input/webqsp.examples.test.wikidata.json")
            if len(data) > 0:
                tests.append(("WebQSP Loader", f"PASS ({len(data)} questions)"))
            else:
                tests.append(("WebQSP Loader", "WARN: No data loaded"))
        else:
            tests.append(("WebQSP Loader", "SKIP: File not found"))
    except Exception as e:
        tests.append(("WebQSP Loader", f"FAIL: {e}"))
    
    return tests

def test_scripts():
    """Test that all scripts are syntactically valid."""
    import glob
    
    tests = []
    scripts = glob.glob("scripts/*.py") + glob.glob("src/crag/analysis/*.py")
    
    for script in scripts:
        try:
            with open(script, 'r') as f:
                compile(f.read(), script, 'exec')
            tests.append((os.path.basename(script), "PASS"))
        except SyntaxError as e:
            tests.append((os.path.basename(script), f"FAIL: {e}"))
    
    return tests

def main():
    print("\n" + "="*70)
    print("C-RAG COMPREHENSIVE TEST SUITE")
    print("="*70 + "\n")
    
    # Test 1: Imports
    print("[1/4] Testing Module Imports...")
    import_tests = test_imports()
    for name, result in import_tests:
        status = "[OK]" if "PASS" in result else "[FAIL]"
        print(f"  {status} {name:30s} {result}")
    
    # Test 2: Configs
    print("\n[2/4] Testing Configuration Files...")
    config_tests = test_configs()
    for name, result in config_tests:
        status = "[OK]" if "PASS" in result else "[WARN]" if "WARN" in result else "[FAIL]"
        print(f"  {status} {name:30s} {result}")
    
    # Test 3: Data Loaders
    print("\n[3/4] Testing Data Loaders...")
    loader_tests = test_data_loaders()
    for name, result in loader_tests:
        status = "[OK]" if "PASS" in result else "[SKIP]" if "SKIP" in result else "[WARN]" if "WARN" in result else "[FAIL]"
        print(f"  {status} {name:30s} {result}")
    
    # Test 4: Scripts
    print("\n[4/4] Testing Script Syntax...")
    script_tests = test_scripts()
    for name, result in script_tests:
        status = "[OK]" if "PASS" in result else "[FAIL]"
        print(f"  {status} {name:30s} {result}")
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    all_tests = import_tests + config_tests + loader_tests + script_tests
    total = len(all_tests)
    passed = sum(1 for _, r in all_tests if "PASS" in r)
    failed = sum(1 for _, r in all_tests if "FAIL" in r)
    skipped = sum(1 for _, r in all_tests if "SKIP" in r)
    warnings = sum(1 for _, r in all_tests if "WARN" in r)
    
    print(f"Total Tests: {total}")
    print(f"  Passed:  {passed}")
    print(f"  Failed:  {failed}")
    print(f"  Skipped: {skipped}")
    print(f"  Warnings: {warnings}")
    
    if failed > 0:
        print("\n[FAIL] Some tests failed - review errors above")
        return 1
    elif warnings > 0:
        print("\n[WARN] All tests passed with warnings")
        return 0
    else:
        print("\n[OK] All tests passed!")
        return 0

if __name__ == "__main__":
    sys.exit(main())
