from langchain_community.embeddings import HuggingFaceEmbeddings
from src.config import settings


def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)
