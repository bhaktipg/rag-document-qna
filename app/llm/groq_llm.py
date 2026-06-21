import os
from groq import Groq
from loguru import logger


class GroqLLM:
    """Wrapper for the Groq Cloud API.

    The API key is resolved in this priority order:
      1. ``api_key`` constructor arg
      2. ``GROQ_API_KEY`` env var
    """

    def __init__(self, model_name: str = None, api_key: str = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model_name = model_name or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        if not self.api_key:
            logger.warning("GROQ_API_KEY is not set — Groq calls will fail.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_client(self) -> Groq:
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable is missing. "
                "Set it in your .env file or pass it as a constructor argument."
            )
        return Groq(api_key=self.api_key)

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
        """Return a complete response string from Groq."""
        client = self._get_client()
        messages = self._build_messages(prompt, system_prompt)
        try:
            logger.info(f"Groq generate | model={self.model_name}")
            completion = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
            )
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq generate failed: {e}")
            raise RuntimeError(f"Groq error: {e}") from e

    def generate_stream(self, prompt: str, system_prompt: str = None):
        """Yield response tokens from Groq for a streaming UI."""
        client = self._get_client()
        messages = self._build_messages(prompt, system_prompt)
        try:
            logger.info(f"Groq stream | model={self.model_name}")
            stream = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta is not None:
                    yield delta
        except Exception as e:
            logger.error(f"Groq stream failed: {e}")
            raise RuntimeError(f"Groq stream error: {e}") from e
