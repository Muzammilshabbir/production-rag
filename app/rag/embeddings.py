from sentence_transformers import (
    SentenceTransformer,
)

from app.config import EMBEDDING_MODEL


model = SentenceTransformer(
    EMBEDDING_MODEL
)


def embed_texts(
    texts: list[str],
) -> list[list[float]]:

    embeddings = model.encode(
        texts
    )

    return embeddings.tolist()


def embed_query(
    query: str,
) -> list[float]:

    embedding = model.encode(
        query
    )

    return embedding.tolist()