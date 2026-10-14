from rank_bm25 import BM25Okapi
from langsmith import traceable

class BM25Retriever:

    def __init__(self, documents: list[dict]):
        self.documents = documents

        tokenized_documents = [
            item["document"].lower().split()
            for item in documents
        ]

        self.bm25 = BM25Okapi(
            tokenized_documents
        )

    @traceable(name="bm25_search")
    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:

        query_tokens = query.lower().split()

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked_indexes = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        return [
            {
                "id": self.documents[index]["id"],
                "document": self.documents[index]["document"],
                "score": float(scores[index]),
            }
            for index in ranked_indexes
        ]