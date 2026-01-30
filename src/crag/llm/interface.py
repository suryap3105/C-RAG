"""
C-RAG V3 LLM Interface
Production-Ready Multi-Provider LLM Client
"""
import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """Abstract LLM client interface."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        pass
        
    def generate_json(self, prompt: str, **kwargs) -> Dict:
        response = self.generate(prompt, **kwargs)
        try:
            return json.loads(response)
        except:
            return {}


class MockLLMClient(LLMClient):
    """
    Mock client for testing without LLM calls.
    """
    def __init__(self):
        self.responses = {
            'default': {
                "nodes": [
                    {"id": "Entity", "type": "Entity"},
                    {"id": "?", "type": "Target"}
                ],
                "edges": [
                    {"src": "Entity", "dst": "?", "relation": "RELATED_TO"}
                ]
            }
        }
        
    def _detect_query_type(self, prompt: str) -> Dict:
        prompt_lower = prompt.lower()
        
        if 'directed' in prompt_lower or 'director' in prompt_lower:
            return {
                "nodes": [
                    {"id": "Director", "type": "Person"},
                    {"id": "?", "type": "Movie"}
                ],
                "edges": [
                    {"src": "Director", "dst": "?", "relation": "DIRECTED"}
                ]
            }
        elif 'acted' in prompt_lower or 'actor' in prompt_lower:
            return {
                "nodes": [
                    {"id": "Actor", "type": "Person"},
                    {"id": "?", "type": "Movie"}
                ],
                "edges": [
                    {"src": "Actor", "dst": "?", "relation": "ACTED_IN"}
                ]
            }
        elif 'produced' in prompt_lower or 'production' in prompt_lower:
            return {
                "nodes": [
                    {"id": "Company", "type": "Company"},
                    {"id": "?", "type": "Movie"}
                ],
                "edges": [
                    {"src": "Company", "dst": "?", "relation": "PRODUCED"}
                ]
            }
            
        return self.responses['default']
        
    def generate(self, prompt: str, **kwargs) -> str:
        logger.debug("[MockLLM] Generating response")
        return json.dumps(self._detect_query_type(prompt))


class OllamaClient(LLMClient):
    """
    Ollama LLM client for local models.
    """
    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434",
                 temperature: float = 0.7, max_tokens: int = 1024):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        
    def generate(self, prompt: str, **kwargs) -> str:
        import requests
        
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get('temperature', self.temperature),
                "num_predict": kwargs.get('max_tokens', self.max_tokens)
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            return response.json().get('response', '')
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return '{}'


class OpenAIClient(LLMClient):
    """
    OpenAI API client.
    """
    def __init__(self, model: str = "gpt-4o-mini", api_key: str = None,
                 temperature: float = 0.7, max_tokens: int = 1024):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None
        
    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        return self._client
        
    def generate(self, prompt: str, **kwargs) -> str:
        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=kwargs.get('max_tokens', self.max_tokens)
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI error: {e}")
            return '{}'


class AnthropicClient(LLMClient):
    """
    Anthropic Claude API client.
    """
    def __init__(self, model: str = "claude-3-haiku-20240307", api_key: str = None,
                 temperature: float = 0.7, max_tokens: int = 1024):
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None
        
    def _get_client(self):
        if self._client is None:
            from anthropic import Anthropic
            self._client = Anthropic(api_key=self.api_key)
        return self._client
        
    def generate(self, prompt: str, **kwargs) -> str:
        try:
            client = self._get_client()
            response = client.messages.create(
                model=self.model,
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic error: {e}")
            return '{}'


def create_llm_client(provider: str = "mock", **kwargs) -> LLMClient:
    """Factory function to create LLM clients."""
    providers = {
        'mock': MockLLMClient,
        'ollama': OllamaClient,
        'openai': OpenAIClient,
        'anthropic': AnthropicClient
    }
    
    if provider not in providers:
        logger.warning(f"Unknown provider {provider}. Using mock.")
        return MockLLMClient()
        
    return providers[provider](**kwargs)
