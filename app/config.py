import os

from dotenv import load_dotenv


load_dotenv()


LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "qwen3:8b"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "all-MiniLM-L6-v2"
)

CHROMA_PATH = os.getenv(
    "CHROMA_PATH",
    "./data/chroma"
)