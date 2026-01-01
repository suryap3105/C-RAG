from abc import ABC, abstractmethod
import requests
import json
from typing import List, Dict, Optional

class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generate text from the LLM."""
        pass

class OllamaClient(LLMClient):
    """
    Client for local Ollama instance (running Llama-3, Phi-3, etc.).
    Defaults to 'llama3' but configurable.
    """
    def __init__(self, model_name: str = "llama3", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url

    def generate(self, prompt: str, system_prompt: str = "You are a helpful reasoning agent.") -> str:
        url = f"{self.base_url}/api/generate"
       
        # Ensure model name includes tag if not present
        model = self.model_name if ":" in self.model_name else f"{self.model_name}:latest"
        
        payload = {
            "model": model,
            "prompt": f"{system_prompt}\n\n{prompt}",
            "stream": False
        }
        try:
            response = requests.post(url, json=payload, timeout=60)
            
            if response.status_code == 404:
                print(f"[ERROR] Ollama endpoint not found. Tried: {url}")
                print(f"[ERROR] Model: {model}")
                print(f"[HINT] Ensure Ollama is running: 'ollama serve'")
                print(f"[HINT] Check model exists: 'ollama list'")
                return "Error: Ollama endpoint not available."
            
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")
        except requests.RequestException as e:
            print(f"Ollama Request Failed: {e}")
            return "Error: Could not query LLM."

class MockLLMClient(LLMClient):
    """
    Fallback for testing without a running LLM.
    """
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        # print(f"[MockLLM] Receiving Prompt: {prompt[:50]}...")
        prompt_lower = prompt.lower()
        if "inception" in prompt_lower:
             if "christopher nolan" in prompt_lower: # Already found
                 return "ANSWER_FOUND: Christopher Nolan"
             return "HYPOTHESIS: The director is likely connected to the movie entity.\nMISSING: Director relation.\nACTION: EXPAND: director"
        
        if "tom hanks" in prompt_lower:
             if "forrest gump" in prompt_lower:
                 return "ANSWER_FOUND: Forrest Gump"
             return "HYPOTHESIS: Movies starred in are connected via 'starring' or 'cast' relations.\nMISSING: Movie list.\nACTION: EXPAND: cast/starring"

        if "spouse" in prompt_lower:
            return "HYPOTHESIS: Spouse info needed.\nACTION: EXPAND: spouse"
            
        return "HYPOTHESIS: Need more info.\nACTION: EXPAND: generic"
