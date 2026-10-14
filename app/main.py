from fastapi import FastAPI
from pydantic import BaseModel

from app.services.rag_service import (
    answer_question,
)


app = FastAPI(
    title="Production RAG API"
)


class ChatRequest(BaseModel):

    question: str


@app.get("/health")
def health():

    return {
        "status": "ok"
    }


@app.post("/chat")
def chat(
    request: ChatRequest,
):

    return answer_question(
        request.question
    )