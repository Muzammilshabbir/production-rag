from app.rag.chunker import chunk_text
from app.rag.embeddings import (
    embed_texts,
    embed_query,
)
from app.rag.vector_store import (
    add_documents,
    search,
)
from app.rag.generator import (
    generate_answer,
)
from app.rag.relevance import (
    calculate_similarity,
)
from app.rag.reranker import rerank

from app.rag.hybrid_search import hybrid_search
from app.rag.vector_store import get_all_documents

def ingest_document(
    document_id: str,
    text: str,
):

    chunks = chunk_text(text)

    embeddings = embed_texts(
        chunks
    )

    ids = [
        f"{document_id}-{i}"
        for i in range(len(chunks))
    ]
    print(ids)
    print("chunks\n", chunks)
    print("chunks\n", embeddings)

    add_documents(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        document_id=document_id,
    )


def answer_question(
    question: str,
):

    query_embedding = embed_query(
        question
    )

# simple vector search
    # results = search(
    #     query_embedding,
    #     top_k=10,
    # )

    # results = hybrid_search(
    #     question,
    #     top_k=10,
    # )

    results = hybrid_search(
        query=question,
        top_k=10,
    )
    # documents = results["documents"][0]

    documents = [
        item["document"]
        for item in results
    ]

    scored_documents = []

    for document in documents:

        score = calculate_similarity(
            question,
            document,
        )

        scored_documents.append(
            {
                "document": document,
                "score": score,
            }
    )

    scored_documents.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    print("Scored documents:\n", scored_documents)

    for item in scored_documents:

        print(
            f"Score: {item['score']:.3f}"
        )

        print(
            item["document"]
        )

        print()

    reranked = rerank(
        question,
        documents,
        top_k=3,
    )

    print("\n--- RERANKING ---")

    for item in reranked:

        print(
            f"Score: {item['score']:.4f}"
        )

        print(
            item["document"]
        )

        print()


    relevant_documents = [
        item["document"]
        for item in reranked
    ]

    print("Relevant documents:\n", relevant_documents)

    if not relevant_documents:
        return {
            "answer": (
                "I don't have enough information "
                "in the available documents to answer "
                "this question."
            ),
            "sources": [],
        }

    context = "\n\n".join(
        relevant_documents
    )
    
    print("Context:\n", context)

    answer = generate_answer(
        question,
        context,
    )

    return {
        "answer": answer,
        "sources": documents,
    }