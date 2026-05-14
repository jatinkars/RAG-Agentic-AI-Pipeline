from pathlib import Path
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from src.chunking import chunk_documents
from src.config import settings
from src.vector_store import save_vector_store


def load_documents(data_dir: str):
    docs = []
    for path in Path(data_dir).glob("**/*"):
        if path.suffix.lower() == ".txt":
            docs.extend(TextLoader(str(path), encoding="utf-8").load())
        elif path.suffix.lower() == ".pdf":
            docs.extend(PyPDFLoader(str(path)).load())
    return docs


def main():
    documents = load_documents(settings.data_dir)
    if not documents:
        print("No documents found. Add .txt or .pdf files to data/sample_docs/")
        return

    chunks = chunk_documents(
        documents,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    save_vector_store(chunks)
    print(f"Indexed {len(chunks)} chunks.")


if __name__ == "__main__":
    main()
