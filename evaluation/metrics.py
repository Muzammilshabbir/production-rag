def precision_at_k(
    results: list[dict],
    expected_text: str,
    k: int,
) -> float:

    top_results = results[:k]

    relevant = 0

    for result in top_results:

        if expected_text in result["document"]:
            relevant += 1

    if not top_results:
        return 0

    return relevant / len(top_results)

def recall_at_k(
    results,
    expected_text,
    k,
):
    top_results = results[:k]

    for result in top_results:
        if expected_text.lower() in result["document"].lower():
            return 1

    return 0


def reciprocal_rank(
    retrieved_ids,
    relevant_ids,
):
    """
    Reciprocal Rank for ONE query.
    """

    for rank, doc_id in enumerate(
        retrieved_ids,
        start=1,
    ):
        if doc_id in relevant_ids:
            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(
    all_retrieved_ids,
    all_relevant_ids,
):
    """
    MRR across multiple queries.
    """

    if not all_retrieved_ids:
        return 0.0

    reciprocal_ranks = []

    for retrieved_ids, relevant_ids in zip(
        all_retrieved_ids,
        all_relevant_ids,
    ):
        rr = reciprocal_rank(
            retrieved_ids,
            relevant_ids,
        )

        reciprocal_ranks.append(rr)

    return sum(reciprocal_ranks) / len(
        reciprocal_ranks
    )