from langsmith import traceable


@traceable
def search_documents(query: str):
    return [
        "Refunds are available within 30 days.",
        "Shipping usually takes 5 business days.",
    ]


@traceable
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