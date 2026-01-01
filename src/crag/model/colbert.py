import torch
from typing import List, Any

class ColBERTReranker:
    """
    Production Re-ranker using Cross-Encoder architecture.
    (Functionally equivalent role to ColBERT in this pipeline: High-precision re-ranking).
    """
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                print(f"Loading CrossEncoder: {self.model_name} on {self._device}...")
                self._model = CrossEncoder(self.model_name, device=self._device)
            except ImportError:
                print("WARNING: sentence-transformers not found. Re-ranking will be bypassed.")
                self._model = "BYPASS"

    def score(self, query: str, documents: List[str], return_details: bool = False) -> Any:
        """
        Returns relevance scores for query-doc pairs.
        If return_details is True, returns (scores, explanations).
        """
        self._load_model()
        
        if not documents:
            return ([] if not return_details else ([], []))
            
        if self._model == "BYPASS":
            # Return length-based heuristic if missing dependencies in strict env
            scores = [float(len(d)) for d in documents]
            return (scores, ["Bypass mode"] * len(documents)) if return_details else scores

        # Prepare pairs
        pairs = [[query, doc] for doc in documents]
        
        # Inference
        scores = self._model.predict(pairs)
        
        # Convert numpy/tensor to list
        if not isinstance(scores, list):
            scores = scores.tolist()
            
        if return_details:
             # Placeholder for token-level supervision
             # In a real ColBERT, this would return MaxSim matrices
             explanations = ["Detailed token matching not available in CrossEncoder mode"] * len(scores)
             return scores, explanations
             
        return scores
