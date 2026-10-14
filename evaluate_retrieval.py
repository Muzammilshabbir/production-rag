import json

from app.rag.hybrid_search import hybrid_search
from evaluation.metrics import recall_at_k
from app.rag.hybrid_search import hybrid_search
from app.rag.vector_store import get_all_documents, VectorRetriever
from app.rag.bm25 import BM25Retriever

from evaluation.metrics import reciprocal_rank

with open(
    "evaluation/dataset.json",
    "r",
    encoding="utf-8",
) as file:

    dataset = json.load(file)


recall_1 = []
recall_3 = []
recall_5 = []


for item in dataset:

    if not item["answerable"]:
        continue

    question = item["question"]

    expected_text = item[
        "expected_text"
    ]

    results = hybrid_search(
        question,
        top_k=5,
    )

    recall_1.append(
        recall_at_k(
            results,
            expected_text,
            1,
        )
    )

    recall_3.append(
        recall_at_k(
            results,
            expected_text,
            3,
        )
    )

    recall_5.append(
        recall_at_k(
            results,
            expected_text,
            5,
        )
    )


def average(values):

    if not values:
        return 0

    return sum(values) / len(values)


print(
    f"Recall@1: "
    f"{average(recall_1):.2%}"
)

print(
    f"Recall@3: "
    f"{average(recall_3):.2%}"
)

print(
    f"Recall@5: "
    f"{average(recall_5):.2%}"
)


def load_dataset(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate():
    dataset = load_dataset(
        "evaluation/dataset.json"
    )

    vector_store = get_all_documents()

    bm25 = BM25Retriever(
        vector_store
    )

    all_retrieved_ids = []
    all_relevant_ids = []

    for item in dataset:

        query = item["query"]

        results = hybrid_search(
            query=query,
            top_k=10,
        )

        retrieved_ids = [
            result["id"]
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

    mrr = reciprocal_rank(
        all_retrieved_ids,
        all_relevant_ids,
    )

    print(f"MRR: {mrr:.4f}")


if __name__ == "__main__":
    evaluate()