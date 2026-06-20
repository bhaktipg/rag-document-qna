import os
import ollama
from loguru import logger

class OllamaLLM:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or os.getenv("OLLAMA_MODEL", "llama3.1:8b")

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """Generate response from local Ollama model."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            logger.info(f"Invoking Ollama with model={self.model_name}")
            response = ollama.chat(model=self.model_name, messages=messages)
            return response["message"]["content"]
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise RuntimeError(f"Ollama error: {e}")
            
    def generate_stream(self, prompt: str, system_prompt: str = None):
        """Yield response tokens progressively for streaming UI."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            logger.info(f"Invoking Ollama streaming with model={self.model_name}")
            stream = ollama.chat(model=self.model_name, messages=messages, stream=True)
            for chunk in stream:
                yield chunk["message"]["content"]
        except Exception as e:
            logger.error(f"Ollama streaming failed: {e}")
            raise RuntimeError(f"Ollama stream error: {e}")