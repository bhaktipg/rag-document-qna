import os
from typing import List, Dict, Any, Generator, Tuple
from app.rag.retriever import RAGRetriever
from app.llm.ollama_llm import OllamaLLM
from app.llm.groq_llm import GroqLLM

# Default System Prompt for RAG
SYSTEM_PROMPT = """You are a helpful assistant. Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, say that you don't know. Keep the answer concise.

Retrieved Context:
{context}
"""

class RAGChain:
    def __init__(self, persist_dir: str = None, embedding_model_name: str = None):
        self.persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        self.embedding_model_name = embedding_model_name or os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.retriever = RAGRetriever()


    def _get_llm(self, provider: str = None, model_name: str = None):
        provider = provider or os.getenv("LLM_PROVIDER", "ollama")
        if provider == "groq":
            return GroqLLM(model_name=model_name)
        else:
            return OllamaLLM(model_name=model_name)

    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        formatted = []
        for i, chunk in enumerate(chunks):
            formatted.append(f"[Document: {chunk['source_file']}, Chunk ID: {chunk['chunk_id']}]\nContent: {chunk['text']}")
        return "\n\n".join(formatted)

    def ask(self, query: str, provider: str = None, model_name: str = None, top_k: int = 5, collections: List[str] = None) -> Tuple[str, List[Dict[str, Any]]]:
        """Runs the complete RAG pipeline synchronously."""
        llm = self._get_llm(provider, model_name)
        chunks = self.retriever.retrieve(query, top_k=top_k, collections=collections)
        
        if not chunks:
            # If no context is available, just answer directly or let the LLM know
            context = "No relevant context found in documents."
        else:
            context = self._format_context(chunks)
            
        system_prompt = SYSTEM_PROMPT.format(context=context)
        answer = llm.generate(prompt=query, system_prompt=system_prompt)
        return answer, chunks

    def ask_stream(self, query: str, provider: str = None, model_name: str = None, top_k: int = 5, collections: List[str] = None) -> Tuple[Generator[str, None, None], List[Dict[str, Any]]]:
        """Runs the RAG pipeline and returns a stream generator along with citations."""
        llm = self._get_llm(provider, model_name)
        chunks = self.retriever.retrieve(query, top_k=top_k, collections=collections)
        
        if not chunks:
            context = "No relevant context found in documents."
        else:
            context = self._format_context(chunks)
            
        system_prompt = SYSTEM_PROMPT.format(context=context)
        stream = llm.generate_stream(prompt=query, system_prompt=system_prompt)
        return stream, chunks