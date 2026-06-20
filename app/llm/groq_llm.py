import os
from groq import Groq
from loguru import logger

class GroqLLM:
    def __init__(self, model_name: str = None, api_key: str = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model_name = model_name or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        if not self.api_key:
            logger.warning("GROQ_API_KEY is not set. Groq calls will fail.")

    def _get_client(self) -> Groq:
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable is missing.")
        return Groq(api_key=self.api_key)

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """Generate response from Groq Cloud API."""
        client = self._get_client()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            logger.info(f"Invoking Groq with model={self.model_name}")
            completion = client.chat.completions.create(
                model=self.model_name,
                messages=messages
            )
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq API call failed: {e}")
            raise RuntimeError(f"Groq error: {e}")

    def generate_stream(self, prompt: str, system_prompt: str = None):
        """Yield response tokens from Groq API."""
        client = self._get_client()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            logger.info(f"Invoking Groq streaming with model={self.model_name}")
            stream = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=True
            )
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"Groq streaming failed: {e}")
            raise RuntimeError(f"Groq stream error: {e}")