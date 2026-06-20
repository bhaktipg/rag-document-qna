# ChromaDB Vectorstore Operations
import logging

import os

logger = logging.getLogger(__name__)
from typing import List
import chromadb
from typing import Any

from langchain_core.documents import Document
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings

# Persistence directory for ChromaDB (default to ./chroma_db if not set)
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

# Initialise a persistent client (reuse for the lifetime of the process)
_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


def _sanitize_name(filename: str) -> str:
    """Sanitize a filename into a safe collection name.

    Replaces spaces and periods with underscores and lower‑cases the result.
    """
    return filename.replace(" ", "_").replace(".", "_").lower()


def get_or_create_collection(name: str) -> Collection:
    """Return an existing collection or create a new one.

    The ``name`` is sanitized to ensure it is a valid Chroma collection identifier.
    """
    collection_name = _sanitize_name(name)
    logger.debug("Getting or creating collection %s", collection_name)
    collection = _client.get_or_create_collection(name=collection_name)
    logger.info("Collection %s ready", collection_name)
    return collection


def add_documents(chunks: List[Document]) -> None:
    """Add a list of ``Document`` objects to the appropriate collection.

    All chunks are assumed to originate from the same source file. The source
    filename is taken from the ``source`` or ``filename`` metadata field and is
    used to locate/create the collection.
    """
    if not chunks:
        logger.debug("No chunks provided to add_documents; exiting early.")
        return

    # Determine the source file name – fall back to ``filename`` if ``source`` is missing.
    source = chunks[0].metadata.get("source") or chunks[0].metadata.get("filename")
    if not source:
        raise ValueError("Document metadata must contain a 'source' or 'filename' field.")

    logger.info("Adding %d documents to collection %s", len(chunks), source)
    collection = get_or_create_collection(source)

    ids = [f"{source}_{i}" for i in range(len(chunks))]
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]

    collection.add(
        documents=texts,
        ids=ids,
        metadatas=metadatas,
    )
    logger.debug("Added %d documents to collection %s", len(chunks), source)


def similarity_search(query: str, k: int = 4) -> List[Document]:
    """Search across all collections for the *k* most similar documents.

    The function iterates over every collection, runs a query, and aggregates the
    results into a flat list of ``Document`` objects. Metadata includes the
    originating collection name.
    """
    logger.info("Performing similarity search for query: %s (k=%d)", query, k)
    results: List[Document] = []
    for coll in _client.list_collections():
        logger.debug("Querying collection %s", coll.name)
        response = coll.query(
            query_texts=[query],
            n_results=k,
        )
        for text in response.get("documents", [[]])[0]:
            results.append(Document(page_content=text, metadata={"collection": coll.name}))
    logger.info("Similarity search returned %d documents", len(results))
    return results


def delete_collection(name: str) -> None:
    """Delete the collection identified by ``name`` (sanitized)."""
    collection_name = _sanitize_name(name)
    logger.info("Deleting collection %s", collection_name)
    _client.delete_collection(name=collection_name)
    logger.debug("Collection %s deleted", collection_name)


def list_collections() -> List[str]:
    """Return a list of all existing collection names."""
    collections = [c.name for c in _client.list_collections()]
    logger.debug("Current collections: %s", collections)
    return collections