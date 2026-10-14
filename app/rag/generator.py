import ollama

from app.config import LLM_MODEL


def generate_answer(
    question: str,
    context: str,
) -> str:

    prompt = f"""
You are a question-answering assistant.

Answer the user's question using ONLY the
information contained in the provided context.

Rules:

1. Do not use outside knowledge.
2. Do not invent facts.
3. If the context does not contain enough
   information to answer the question, say so.
4. Keep the answer concise.
5. Every factual claim must be supported
   by the provided context.

Context:
{context}

Question:
{question}
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return response["message"]["content"]