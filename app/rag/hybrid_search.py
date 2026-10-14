from collections import defaultdict

from langsmith import traceable

from app.rag.bm25 import BM25Retriever
from app.rag.vector_retriever import VectorRetriever
from app.rag.vector_store import get_all_documents


# def hybrid_search(
#     query: str,
#     top_k: int = 10,
#     rrf_k: int = 60,
# ):

#     query_embedding = embed_query(
#         query
#     )

#     vector_results = search(
#         query_embedding,
#         top_k=top_k,
#     )

#     vector_documents = (
#         vector_results["documents"][0]
#     )

#     all_documents = get_all_documents()

#     bm25 = BM25Retriever(
#         all_documents
#     )

#     bm25_results = bm25.search(
#         query,
#         top_k=top_k,
#     )

#     scores = defaultdict(float)

#     # Vector ranking
#     for rank, document in enumerate(
#         vector_documents,
#         start=1,
#     ):

#         scores[document] += (
#             1 / (rrf_k + rank)
#         )

#     # BM25 ranking
#     for rank, item in enumerate(
#         bm25_results,
#         start=1,
#     ):

#         scores[item["document"]] += (
#             1 / (rrf_k + rank)
#         )

#     ranked_documents = sorted(
#         scores.items(),
#         key=lambda item: item[1],
#         reverse=True,
#     )

#     return [
#         {
#             "document": document,
#             "score": score,
#         }
#         for document, score
#         in ranked_documents[:top_k]
#     ]

@traceable(name="hybrid_search")
def hybrid_search(
    query: str,
    top_k: int = 10,
) -> list[dict]:

    vector_results = VectorRetriever().search(
        query,
        top_k=top_k,
    )

    all_documents = get_all_documents()


    bm25 = BM25Retriever(
        all_documents
    )


    bm25_results = bm25.search(
        query,
        top_k=top_k,
    )

    fused_results = reciprocal_rank_fusion(
        [
            vector_results,
            bm25_results,
        ]
    )

    return fused_results[:top_k]

def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    k: int = 60,
) -> list[dict]:
    """
    Combine multiple ranked result lists using Reciprocal Rank Fusion.

    Each result must contain a unique 'id'.

    RRF score:
        score += 1 / (k + rank)
    """

    scores = defaultdict(float)
    documents = {}

    for result_list in result_lists:
        for rank, result in enumerate(result_list, start=1):
            doc_id = result["id"]

            scores[doc_id] += 1 / (k + rank)

            # Keep the actual document
            documents[doc_id] = result

    fused_results = []

    for doc_id, score in scores.items():
        result = documents[doc_id].copy()
        result["rrf_score"] = score
        fused_results.append(result)

    fused_results.sort(
        key=lambda x: x["rrf_score"],
        reverse=True,
    )

    return fused_results