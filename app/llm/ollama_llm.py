import os
import ollama
from loguru import logger


class OllamaLLM:
    """Wrapper for a locally-running Ollama instance.

    The Ollama host is resolved in this priority order:
      1. ``host`` constructor arg
      2. ``OLLAMA_HOST`` env var   (useful inside Docker / remote machines)
      3. Falls back to the Ollama SDK default (http://localhost:11434)
    """

    def __init__(self, model_name: str = None, host: str = None):
        self.model_name = model_name or os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        host = host or os.getenv("OLLAMA_HOST")  # e.g. "http://ollama:11434" in Docker
        self._client = ollama.Client(host=host) if host else ollama.Client()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_messages(self, prompt: str, system_prompt: str = None) -> list:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """Return a complete response string from Ollama."""
        messages = self._build_messages(prompt, system_prompt)
        try:
            logger.info(f"Ollama generate | model={self.model_name}")
            response = self._client.chat(model=self.model_name, messages=messages)
            return response["message"]["content"]
        except Exception as e:
            logger.error(f"Ollama generate failed: {e}")
            raise RuntimeError(f"Ollama error: {e}") from e

    def generate_stream(self, prompt: str, system_prompt: str = None):
        """Yield response tokens progressively for a streaming UI."""
        messages = self._build_messages(prompt, system_prompt)
        try:
            logger.info(f"Ollama stream | model={self.model_name}")
            stream = self._client.chat(
                model=self.model_name, messages=messages, stream=True
            )
            for chunk in stream:
                yield chunk["message"]["content"]
        except Exception as e:
            logger.error(f"Ollama stream failed: {e}")
            raise RuntimeError(f"Ollama stream error: {e}") from e
