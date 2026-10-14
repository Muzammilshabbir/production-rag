from langsmith import traceable


@traceable(name="test_search")
def search_documents(query: str):
    return [
        "Refunds are available within 30 days.",
        "Shipping takes 5 business days.",
    ]


@traceable(name="test_rag")
def answer_question(query: str):
    documents = search_documents(query)

    return {
        "query": query,
        "documents": documents,
    }


result = answer_question(
    "What is the refund policy?"
)

print(result)