from app.rag.embeddings import embed_query
from app.rag.vector_store import search
from langsmith import traceable

class VectorRetriever:

    @traceable(name="vector_search")
    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:

        query_embedding = embed_query(query)

        results = search(
            query_embedding,
            top_k=top_k,
        )

        documents = results["documents"][0]
        ids = results["ids"][0]
        distances = results["distances"][0]

        return [
            {
                "id": chunk_id,
                "document": document,
                "score": float(distance),
            }
            for chunk_id, document, distance in zip(
                ids,
                documents,
                distances,
            )
        ]