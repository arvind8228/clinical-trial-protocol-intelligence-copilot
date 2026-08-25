import pandas as pd
import chromadb

from protocol_copilot.config import (
    CHUNKS_PATH,
    CHROMA_PATH,
    CHROMA_COLLECTION_NAME,
)


def load_chunks():
    """
    Load the validated protocol chunks created during Day 2.
    """

    chunks_df = pd.read_json(
        CHUNKS_PATH,
        lines=True,
    )

    return chunks_df


def load_chroma_collection():
    """
    Connect to the persistent Chroma database and return
    the OpenAI embedding collection created during Day 3.
    """

    chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_PATH)
    )

    collection = chroma_client.get_collection(
        name=CHROMA_COLLECTION_NAME
    )

    return collection