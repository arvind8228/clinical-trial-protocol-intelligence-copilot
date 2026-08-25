from pathlib import Path


# Resolve the project root automatically
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Data paths
CHUNKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "protocol_chunks.jsonl"
)

CHROMA_PATH = (
    PROJECT_ROOT
    / "data"
    / "chroma"
)

DOCUMENT_SOURCES_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "document_sources.csv"
)


# Chroma collection
CHROMA_COLLECTION_NAME = "protocol_chunks_openai_v1"


# Models
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
GENERATION_MODEL_NAME = "gpt-5.6-terra"

RERANKER_MODEL_NAME = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# Retrieval configuration
SEMANTIC_CANDIDATE_K = 20
BM25_CANDIDATE_K = 20
FINAL_TOP_K = 5
RRF_K = 60


# Exact fallback for unsupported questions
ABSTENTION_MESSAGE = "Insufficient Evidence"