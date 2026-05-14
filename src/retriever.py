from src.config import settings
from src.vector_store import load_vector_store


def retrieve_context(question: str, top_k: int | None = None):
    store = load_vector_store()
    results = store.similarity_search(question, k=top_k or settings.top_k)
    return results
