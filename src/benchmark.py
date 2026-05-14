import time
import pandas as pd
from src.chunking import chunk_documents
from src.ingest import load_documents
from src.vector_store import save_vector_store, load_vector_store
from src.config import settings


CHUNKING_EXPERIMENTS = [
    {"chunk_size": 300, "chunk_overlap": 30},
    {"chunk_size": 500, "chunk_overlap": 50},
    {"chunk_size": 800, "chunk_overlap": 80},
]

SAMPLE_QUERIES = [
    "What are the main concepts in the documents?",
    "Summarize the key findings.",
    "What evidence supports the conclusion?",
]


def run_benchmark():
    documents = load_documents(settings.data_dir)
    if not documents:
        print("No documents found for benchmarking.")
        return

    rows = []
    for experiment in CHUNKING_EXPERIMENTS:
        chunks = chunk_documents(documents, **experiment)
        save_vector_store(chunks)
        store = load_vector_store()

        for query in SAMPLE_QUERIES:
            start = time.perf_counter()
            results = store.similarity_search(query, k=settings.top_k)
            latency_ms = (time.perf_counter() - start) * 1000

            rows.append(
                {
                    "chunk_size": experiment["chunk_size"],
                    "chunk_overlap": experiment["chunk_overlap"],
                    "query": query,
                    "retrieved_docs": len(results),
                    "latency_ms": round(latency_ms, 2),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv("benchmark_results.csv", index=False)
    print(df)
    print("Saved results to benchmark_results.csv")


if __name__ == "__main__":
    run_benchmark()
