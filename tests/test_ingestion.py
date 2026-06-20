import os
import pytest
import fitz  # PyMuPDF
from app.rag.ingestion import validate_document, extract_metadata, load_document, chunk_document
from langchain_core.documents import Document

def test_validate_document_nonexistent():
    assert validate_document("nonexistent_file.pdf") is False
    assert validate_document("") is False

def test_validate_document_unsupported(tmp_path):
    unsupported = tmp_path / "test.json"
    unsupported.write_text("{}")
    assert validate_document(str(unsupported)) is False

def test_validate_document_empty_file(tmp_path):
    empty_file = tmp_path / "test.txt"
    empty_file.write_text("")
    assert validate_document(str(empty_file)) is False

def test_extract_metadata():
    meta = extract_metadata("dir/sample.pdf", 3, 150)
    assert meta == {
        "filename": "sample.pdf",
        "page_number": 3,
        "file_type": "pdf",
        "char_count": 150
    }

def test_load_txt_document(tmp_path):
    txt_file = tmp_path / "test.txt"
    content = "This is a simple text file content that contains more than 50 characters to ensure it gets processed successfully."
    txt_file.write_text(content)
    
    docs = load_document(str(txt_file))
    assert len(docs) == 1
    assert docs[0].page_content == content
    assert docs[0].metadata["filename"] == "test.txt"
    assert docs[0].metadata["page_number"] == 1
    assert docs[0].metadata["file_type"] == "txt"
    assert docs[0].metadata["char_count"] == len(content)

def test_load_txt_document_too_short(tmp_path):
    txt_file = tmp_path / "test.txt"
    content = "Short text."
    txt_file.write_text(content)
    
    docs = load_document(str(txt_file))
    assert len(docs) == 0

def test_load_md_document(tmp_path):
    md_file = tmp_path / "test.md"
    content = "# Title\n\nThis is a markdown content file that is long enough to satisfy the page length validation parameter requirements."
    md_file.write_text(content)
    
    docs = load_document(str(md_file))
    assert len(docs) == 1
    # Depending on unstructured loader, leading hash or whitespace might be cleaned, but text should match overall
    assert len(docs[0].page_content) >= 50
    assert docs[0].metadata["filename"] == "test.md"
    assert docs[0].metadata["file_type"] == "md"

def test_load_pdf_document(tmp_path):
    pdf_path = tmp_path / "test.pdf"
    
    # Create a small valid PDF using fitz
    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text((50, 50), "Header Page 1\nThis is the body content of page 1 which is sufficiently long to pass the 50 char threshold.\nFooter Page 1")
    page2 = doc.new_page()
    page2.insert_text((50, 50), "Short page")
    doc.save(str(pdf_path))
    doc.close()
    
    docs = load_document(str(pdf_path))
    
    # page 2 should be skipped (< 50 chars)
    assert len(docs) == 1
    assert docs[0].metadata["page_number"] == 1
    assert docs[0].metadata["file_type"] == "pdf"
    assert "body content of page 1" in docs[0].page_content

def test_chunk_document():
    # Construct doc with long text
    long_text = ("Word " * 500) # 500 * 5 = 2500 chars (exceeds 2048)
    doc = Document(
        page_content=long_text,
        metadata={"filename": "test.txt", "page_number": 2}
    )
    
    chunks = chunk_document([doc])
    
    # Should split into at least 2 chunks
    assert len(chunks) > 1
    
    # Verify metadata fields
    first_chunk = chunks[0]
    assert first_chunk.metadata["source"] == "test.txt"
    assert first_chunk.metadata["page"] == 2
    assert first_chunk.metadata["chunk_index"] == 0
    assert first_chunk.metadata["total_chunks"] == len(chunks)
    assert first_chunk.metadata["char_count"] == len(first_chunk.page_content)
    
    # Check that overlap occurred or contents are correct
    assert chunks[1].metadata["chunk_index"] == 1
    assert chunks[1].metadata["total_chunks"] == len(chunks)
