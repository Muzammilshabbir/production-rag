# production-rag

Production-ready Retrieval-Augmented Generation (RAG) application.

for Bash: source .venv/Scripts/activate
uvicorn app.main:app --reload

## Structure

```
production-rag/
├── app/
│   ├── main.py
│   └── rag/
│       ├── chunker.py
│       ├── embeddings.py
│       ├── vector_store.py
│       └── generator.py
├── data/documents/
├── tests/
├── .env
├── requirements.txt
└── README.md
```

## Setup

1. Create and activate a virtual environment
2. Install dependencies: `pip install -r requirements.txt`
3. Copy `.env` and set your API keys / config
4. Place documents in `data/documents/`

