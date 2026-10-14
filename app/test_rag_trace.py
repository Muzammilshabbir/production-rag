from app.rag.hybrid_search import hybrid_search
from app.rag.bm25 import BM25Retriever
from app.rag.vector_retriever import VectorRetriever
from app.rag.vector_store import get_all_documents


documents = get_all_documents()

bm25 = BM25Retriever(documents)

vector = VectorRetriever()

results = hybrid_search(
    query="What is the refund policy?",
    vector_retriever=vector,
    bm25_retriever=bm25,
    top_k=5,
)

for result in results:
    print(result)