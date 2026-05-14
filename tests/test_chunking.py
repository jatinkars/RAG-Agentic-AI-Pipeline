from langchain_core.documents import Document
from src.chunking import chunk_documents


def test_chunk_documents_returns_chunks():
    docs = [Document(page_content="This is a sample document. " * 100)]
    chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=10)
    assert len(chunks) > 1
