# Integration test for ChromaDB vectorstore
import logging
import os

from app.rag.ingestion import load_document, chunk_document
from app.rag.vectorstore import add_documents, similarity_search, delete_collection, list_collections

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

def main():
    # Create a temporary text file
    tmp_path = os.path.join(os.getcwd(), "temp_test_file.txt")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write("""Artificial Intelligence (AI) is the simulation of human intelligence processes by machines, especially computer systems.
        Specific applications of AI include expert systems, natural language processing, speech recognition, and machine vision.
        """)

    # Load document and chunk it
    docs = load_document(tmp_path)
    if not docs:
        logging.error("Failed to load document %s", tmp_path)
        return
    chunks = chunk_document(docs)
    logging.info("Loaded %d chunks", len(chunks))

    # Add to vectorstore
    add_documents(chunks)
    logging.info("Current collections: %s", list_collections())

    # Search
    results = similarity_search("What is AI?", k=2)
    for i, doc in enumerate(results, 1):
        logging.info("Result %d (collection=%s): %s", i, doc.metadata.get("collection"), doc.page_content[:60])

    # Cleanup
    delete_collection(os.path.basename(tmp_path).replace(" ", "_").replace(".", "_").lower())
    os.remove(tmp_path)
    logging.info("Cleanup completed.")

if __name__ == "__main__":
    main()
