import chromadb

from app.config import CHROMA_PATH



client = chromadb.PersistentClient(
    path=CHROMA_PATH
)


collection = client.get_or_create_collection(
    name="documents"
)


def add_documents(
    documents: list[str],
    embeddings: list[list[float]],
    ids: list[str],
    document_id: str,
):

    metadatas = [
        {
            "document_id": document_id,
        }
        for _ in documents
    ]

    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas,
    )

def search(
    query_embedding: list[float],
    top_k: int = 5,
):

    return collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

def get_all_documents() -> list[dict]:
    result = collection.get(
        include=[
            "documents",
            "metadatas",
        ]
    )

    return [
        {
            "id": chunk_id,
            "document": document,
        }
        for chunk_id, document in zip(
            result["ids"],
            result["documents"],
        )
    ]

