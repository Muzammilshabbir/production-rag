from sentence_transformers import SentenceTransformer


model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


def calculate_similarity(
    query: str,
    document: str,
) -> float:

    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    )

    document_embedding = model.encode(
        document,
        normalize_embeddings=True,
    )

    return float(
        query_embedding @ document_embedding
    )


def is_relevant(
    query: str,
    document: str,
    threshold: float = 0.45,
) -> bool:

    score = calculate_similarity(
        query,
        document,
    )

    return score >= threshold