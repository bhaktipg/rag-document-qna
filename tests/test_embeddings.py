import pytest
from app.rag.embeddings import EmbeddingModel

def test_singleton_pattern():
    """Verify that multiple instantiations return the same object instance."""
    model1 = EmbeddingModel()
    model2 = EmbeddingModel()
    assert model1 is model2
    assert model1.model is model2.model

def test_embed_query():
    """Verify that a single query is embedded into a 384-dimensional float vector."""
    model = EmbeddingModel()
    query = "What is Retrieval-Augmented Generation?"
    embedding = model.embed_query(query)
    
    
    assert isinstance(embedding, list)
    assert len(embedding) == 384
    assert all(isinstance(x, float) for x in embedding)

def test_embed_documents():
    """Verify that a list of documents are embedded into 384-dimensional float vectors."""
    model = EmbeddingModel()
    docs = [
        "First document content snippet.",
        "Second document containing different terms.",
        "Third document to check batching behavior."
    ]
    embeddings = model.embed_documents(docs)
    
    assert isinstance(embeddings, list)
    assert len(embeddings) == 3
    for emb in embeddings:
        assert isinstance(emb, list)
        assert len(emb) == 384
        assert all(isinstance(x, float) for x in emb)

def test_empty_inputs():
    """Verify how the model behaves with empty input values."""
    model = EmbeddingModel()
    assert model.embed_query("") == []
    assert model.embed_documents([]) == []
