from pathlib import Path
from langchain_community.vectorstores import FAISS, Chroma
from src.config import settings
from src.embeddings import get_embedding_model


def save_vector_store(documents):
    embeddings = get_embedding_model()
    Path(settings.index_dir).mkdir(exist_ok=True)

    if settings.vector_db.lower() == "chroma":
        store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=settings.index_dir,
        )
        store.persist()
        return store

    store = FAISS.from_documents(documents, embeddings)
    store.save_local(settings.index_dir)
    return store


def load_vector_store():
    embeddings = get_embedding_model()

    if settings.vector_db.lower() == "chroma":
        return Chroma(
            persist_directory=settings.index_dir,
            embedding_function=embeddings,
        )

    return FAISS.load_local(
        settings.index_dir,
        embeddings,
        allow_dangerous_deserialization=True,
    )
