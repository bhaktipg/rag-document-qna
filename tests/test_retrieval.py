import pytest
import shutil
import os
from langchain_core.documents import Document
import app.rag.vectorstore as vs

@pytest.fixture(autouse=True)
def clean_db():
    # Before and after test, delete existing collections if any
    cols = vs.list_collections()
    for col in cols:
        vs.delete_collection(col)
    yield
    cols = vs.list_collections()
    for col in cols:
        vs.delete_collection(col)

def test_add_and_retrieve():
    chunks = [
        Document(page_content="Python is a programming language.", metadata={"source": "python.txt"}),
        Document(page_content="Apples are delicious fruits.", metadata={"source": "apples.txt"})
    ]
    
    vs.add_documents([chunks[0]])
    vs.add_documents([chunks[1]])
    
    # Search for programming language
    results = vs.similarity_search("What is Python?", k=1)
    assert len(results) >= 1
    assert "programming language" in results[0].page_content
    
    # Search for fruit
    results = vs.similarity_search("Tell me about apples.", k=1)
    assert len(results) >= 1
    assert "delicious fruits" in results[0].page_content
