from fastapi import FastAPI
from pydantic import BaseModel
from src.agent import generate_answer

app = FastAPI(title="RAG & Agentic AI Pipeline")


class QueryRequest(BaseModel):
    question: str


@app.get("/")
def health_check():
    return {"status": "ok", "message": "RAG pipeline API is running"}


@app.post("/query")
def query(request: QueryRequest):
    return generate_answer(request.question)
