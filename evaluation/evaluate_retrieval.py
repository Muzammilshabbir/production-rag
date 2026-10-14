from app.rag.hybrid_search import hybrid_search
from app.rag.vector_store import VectorStore
from app.rag.bm25 import BM25Retriever

from evaluation.metrics import mean_reciprocal_rank
import json


def load_dataset(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate():
    dataset = load_dataset(
        "evaluation/dataset.json"
    )

    vector_store = VectorStore()

    bm25 = BM25Retriever(
        vector_store.get_all_documents()
    )

    all_retrieved_ids = []
    all_relevant_ids = []

    for item in dataset:

        query = item["query"]

        results = hybrid_search(
            query=query,
            vector_retriever=vector_store,
            bm25_retriever=bm25,
            top_k=10,
        )

        retrieved_ids = [
            result["document_id"]
            for result in results
        ]

        relevant_ids = set(
            item["relevant_ids"]
        )

        all_retrieved_ids.append(
            retrieved_ids
        )

        all_relevant_ids.append(
            relevant_ids
        )

    mrr = mean_reciprocal_rank(
        all_retrieved_ids,
        all_relevant_ids,
    )

    print(f"MRR: {mrr:.4f}")


if __name__ == "__main__":
    evaluate()