from typing import List, Dict, Any
from app.rag.vectorstore import similarity_search

class RAGRetriever:
    def __init__(self, vectorstore: Any = None):
        pass

    def retrieve(self, query: str, top_k: int = 5, collections: List[str] = None) -> List[Dict[str, Any]]:
        """Retrieve relevant document chunks with text and source metadata."""
        raw_results = similarity_search(
            query=query,
            k=top_k,
            collection_names=collections
        )
        
        formatted_results = []
        for doc in raw_results:
            formatted_results.append({
                "text": doc.page_content,
                "source_file": doc.metadata.get("source", "Unknown"),
                "chunk_id": doc.metadata.get("chunk_index", -1),
                "page": doc.metadata.get("page", 1)
            })
            
        return formatted_results