# RAG & Agentic AI Pipeline
<img width="1535" height="1024" alt="image" src="https://github.com/user-attachments/assets/94cd66fb-0216-499b-873f-2f190f94324d" />

A retrieval-augmented generation system with vector search and an agentic orchestration layer. The project benchmarks retrieval accuracy and latency across multiple chunking strategies, making it useful for comparing RAG pipeline design choices before production deployment.

## Project Summary

This project builds a modular RAG pipeline that can ingest documents, split them into chunks, create embeddings, store them in a vector database, retrieve relevant context, and generate grounded responses using an LLM. It also includes an agentic orchestration layer using LangChain to coordinate retrieval, reasoning, and response generation.

## Key Features

- Document ingestion and preprocessing pipeline
- Configurable chunking strategies
- Embedding generation for semantic search
- Vector search using FAISS or ChromaDB
- Retrieval-augmented response generation
- Agentic orchestration using LangChain
- Benchmarking for retrieval accuracy and latency
- Modular code structure for experimentation and extension

## Tech Stack

- Python
- LangChain
- FAISS / ChromaDB
- OpenAI API / Hugging Face Transformers
- Sentence Transformers
- FastAPI
- pandas, NumPy
- Docker
- GitHub Actions

## Repository Structure

```text
rag-agentic-ai-pipeline/
├── src/
│   ├── config.py
│   ├── ingest.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── agent.py
│   ├── benchmark.py
│   └── app.py
├── data/
│   └── sample_docs/
├── notebooks/
├── tests/
├── .github/
│   └── workflows/
│       └── python-ci.yml
├── .env.example
├── .gitignore
├── Dockerfile
├── requirements.txt
├── LICENSE
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/rag-agentic-ai-pipeline.git
cd rag-agentic-ai-pipeline
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set environment variables

Copy the example environment file:

```bash
cp .env.example .env
```

Add your API key if using OpenAI:

```text
OPENAI_API_KEY=your_api_key_here
```

### 5. Add documents

Place text or PDF documents inside:

```text
data/sample_docs/
```

### 6. Run ingestion

```bash
python src/ingest.py
```

### 7. Run the API

```bash
uvicorn src.app:app --reload
```

Then visit:

```text
http://127.0.0.1:8000/docs
```

## Example Query

```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main themes in the documents?"}'
```

## Benchmarking

Run benchmark experiments across chunking strategies:

```bash
python src/benchmark.py
```

The benchmark tracks:

- Retrieval latency
- Top-k retrieval quality
- Context relevance
- Response grounding
- Chunking strategy impact

## Future Improvements

- Add hybrid search using BM25 and dense embeddings
- Add reranking with cross-encoders
- Add MLflow or Weights & Biases experiment tracking
- Add support for more document formats
- Add UI using Streamlit or React

## Author

Jatin Kumar Reddimalla

## License

This project is licensed under the MIT License.
