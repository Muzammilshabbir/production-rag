from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


model = CrossEncoder(
    MODEL_NAME
)


def rerank(
    query: str,
    documents: list[str],
    top_k: int = 3,
) -> list[dict]:

    pairs = [
        [query, document]
        for document in documents
    ]

    scores = model.predict(
        pairs
    )

    print("Scores:\n", scores)

    results = [
        {
            "document": document,
            "score": float(score),
        }
        for document, score
        in zip(documents, scores)
    ]

    print("results\n",results)
    results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return results[:top_k]