import os
import re
import logging
from typing import List
import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

def validate_document(file_path: str) -> bool:
    """
    Validates if the file at file_path exists, is a file, is not empty,
    has a supported extension, and is not corrupt.
    """
    if not file_path or not isinstance(file_path, str):
        logger.error(f"Invalid file_path: {file_path}")
        return False

    if not os.path.exists(file_path):
        logger.error(f"File does not exist: {file_path}")
        return False

    if not os.path.isfile(file_path):
        logger.error(f"Path is not a file: {file_path}")
        return False

    if os.path.getsize(file_path) == 0:
        logger.error(f"File is empty: {file_path}")
        return False

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    if ext not in [".pdf", ".txt", ".md"]:
        logger.error(f"Unsupported file format: {ext}")
        return False

    # Check for corruption
    if ext == ".pdf":
        try:
            with fitz.open(file_path) as doc:
                # Attempt to read the page count to verify the PDF structure
                _ = len(doc)
        except Exception as e:
            logger.error(f"Corrupt PDF file: {file_path}. Error: {e}")
            return False
    elif ext in [".txt", ".md"]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                f.read(1024)
        except Exception as e:
            logger.error(f"Corrupt or unreadable text/markdown file: {file_path}. Error: {e}")
            return False

    return True

def extract_metadata(file_path: str, page_num: int, char_count: int = 0) -> dict:
    """
    Extracts metadata details from the file path and page number.
    """
    filename = os.path.basename(file_path)
    _, ext = os.path.splitext(file_path)
    file_type = ext.lstrip(".").lower()
    return {
        "filename": filename,
        "page_number": page_num,
        "file_type": file_type,
        "char_count": char_count
    }

def _strip_headers_footers(text: str) -> str:
    """
    Simple heuristic to strip headers and footers from page text.
    """
    if not text:
        return ""
    lines = text.split("\n")
    start_idx = 0
    end_idx = len(lines)

    # Match page numbers, headers, or footers
    page_pattern = re.compile(r"^(page\s+\d+|\d+|section\s+\d+.*)$", re.IGNORECASE)

    # Check the first line (header)
    if start_idx < end_idx:
        first_line = lines[start_idx].strip()
        if page_pattern.match(first_line) or (len(first_line) < 30 and any(first_line.lower().startswith(x) for x in ["chapter", "section", "page", "header"])):
            start_idx += 1

    # Check the last line (footer)
    if start_idx < end_idx:
        last_line = lines[end_idx - 1].strip()
        if page_pattern.match(last_line) or (len(last_line) < 20 and last_line.isdigit()):
            end_idx -= 1

    return "\n".join(lines[start_idx:end_idx]).strip()

def load_document(file_path: str) -> List[Document]:
    """
    Loads a document (PDF, TXT, MD), validates it, parses content,
    extracts metadata, and filters out pages with fewer than 50 characters.
    """
    if not validate_document(file_path):
        logger.error(f"Validation failed for: {file_path}")
        return []

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    documents = []

    try:
        if ext == ".pdf":
            with fitz.open(file_path) as doc:
                for page_idx, page in enumerate(doc):
                    page_num = page_idx + 1
                    raw_text = page.get_text()
                    clean_text = _strip_headers_footers(raw_text)
                    
                    if len(clean_text) < 50:
                        logger.info(f"Skipping page {page_num} in {file_path} due to character count ({len(clean_text)} < 50)")
                        continue
                    
                    metadata = extract_metadata(file_path, page_num, len(clean_text))
                    documents.append(Document(page_content=clean_text, metadata=metadata))

        elif ext == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
            loaded_docs = loader.load()
            for doc in loaded_docs:
                clean_text = doc.page_content.strip()
                if len(clean_text) < 50:
                    logger.info(f"Skipping TXT document {file_path} due to character count ({len(clean_text)} < 50)")
                    continue
                metadata = extract_metadata(file_path, 1, len(clean_text))
                documents.append(Document(page_content=clean_text, metadata=metadata))

        elif ext == ".md":
            loader = UnstructuredMarkdownLoader(file_path)
            loaded_docs = loader.load()
            # Combine all documents into one single text content representation
            combined_text = "\n\n".join([d.page_content for d in loaded_docs]).strip()
            if len(combined_text) < 50:
                logger.info(f"Skipping Markdown document {file_path} due to character count ({len(combined_text)} < 50)")
                return []
            
            metadata = extract_metadata(file_path, 1, len(combined_text))
            documents.append(Document(page_content=combined_text, metadata=metadata))

    except Exception as e:
        logger.error(f"Error loading document {file_path}: {e}")
        return []

    return documents

def chunk_document(documents: List[Document]) -> List[Document]:
    """
    Splits loaded Document objects into smaller chunks using RecursiveCharacterTextSplitter.
    Configured with a chunk size of 2048 characters, 200 character overlap,
    and a specific list of separators.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2048,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunked_docs = []
    
    for doc in documents:
        # Split this single document page
        chunks = splitter.split_documents([doc])
        total_chunks = len(chunks)
        
        for idx, chunk in enumerate(chunks):
            # Extract metadata as requested
            chunk_metadata = {
                "source": doc.metadata.get("filename", ""),
                "page": doc.metadata.get("page_number", 1),
                "chunk_index": idx,
                "total_chunks": total_chunks,
                "char_count": len(chunk.page_content)
            }
            chunked_docs.append(Document(page_content=chunk.page_content, metadata=chunk_metadata))
            
    return chunked_docs