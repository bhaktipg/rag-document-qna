import os
import ssl
import threading
from typing import List
import urllib3
import requests
from sentence_transformers import SentenceTransformer

# Disable SSL verification to prevent certificate errors during model download
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

# Monkeypatch requests.Session.send to force verify=False
_original_send = requests.Session.send
def _unverified_send(self, request, **kwargs):
    kwargs['verify'] = False
    filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'verify'}
    return _original_send(self, request, verify=False, **filtered_kwargs)
requests.Session.send = _unverified_send

class EmbeddingModel:
    """Wrapper class for the sentence-transformers model using the Singleton pattern."""
    
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(EmbeddingModel, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if getattr(self, "_initialized", False):
            return
        self.model = SentenceTransformer(model_name)
        self._initialized = True

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents/texts.
        
        Uses batching with size 32 for performance.
        Returns a list of 384-dimensional float32 vectors (converted to Python lists of floats).
        """
        if not texts:
            return []
        
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return [list(map(float, vec)) for vec in embeddings]

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string.
        
        Returns a single 384-dimensional float32 vector (converted to a Python list of floats).
        """
        if not text:
            return []
            
        embedding = self.model.encode(
            text,
            convert_to_numpy=True
        )
        return list(map(float, embedding))
