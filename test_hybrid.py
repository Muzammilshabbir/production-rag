from app.rag.hybrid_search import hybrid_search


query = "How long do JWT access tokens last?"


results = hybrid_search(
    query,
    top_k=5,
)


print("\n========== VECTOR SEARCH ==========")

for document in results["vector"]:

    print(document)
    print()


print("\n========== BM25 SEARCH ==========")

for item in results["bm25"]:

    print(
        f"Score: {item['score']:.4f}"
    )

    print(
        item["document"]
    )

    print()