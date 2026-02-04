import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from crag.model.cross_encoder import ColBERTReranker
    from crag.retrieval.neural_hybrid import NeuroHybridRetrievalModule
    print("Imports successful.")
except ImportError as e:
    print(f"Import failed: {e}")
    exit(1)
except Exception as e:
    print(f"Error: {e}")
    exit(1)
