import chromadb
import pandas as pd
from openai import OpenAI

from protocol_copilot.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PATH,
    CHUNKS_PATH,
    EMBEDDING_MODEL_NAME,
)


def load_chunks():
    return pd.read_json(
        CHUNKS_PATH,
        lines=True,
    )


def build_demo_collection(chunks_df):
    openai_client = OpenAI()

    texts = chunks_df[
        "text"
    ].astype(str).tolist()

    embedding_response = (
        openai_client.embeddings.create(
            model=EMBEDDING_MODEL_NAME,
            input=texts,
        )
    )

    embeddings = [
        item.embedding
        for item in embedding_response.data
    ]

    metadatas = []

    for _, row in chunks_df.iterrows():
        metadatas.append(
            {
                "document_name": str(
                    row["document_name"]
                ),
                "page_number": int(
                    row["page_number"]
                ),
            }
        )

    client = chromadb.EphemeralClient()

    collection = (
        client.create_collection(
            name=CHROMA_COLLECTION_NAME,
        )
    )

    collection.add(
        ids=chunks_df[
            "chunk_id"
        ].astype(str).tolist(),
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    return collection


def load_chroma_collection():
    if CHROMA_PATH.exists():
        persistent_client = (
            chromadb.PersistentClient(
                path=str(
                    CHROMA_PATH
                )
            )
        )

        collection_names = [
            collection.name
            for collection
            in persistent_client.list_collections()
        ]

        if (
            CHROMA_COLLECTION_NAME
            in collection_names
        ):
            return (
                persistent_client.get_collection(
                    name=CHROMA_COLLECTION_NAME
                )
            )

    chunks_df = load_chunks()

    return build_demo_collection(
        chunks_df
    )