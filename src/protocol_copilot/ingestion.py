import io
import re
import uuid

import chromadb
import pandas as pd
import tiktoken
from pypdf import PdfReader

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

from openai import OpenAI

from protocol_copilot.config import (
    EMBEDDING_MODEL_NAME,
)


CHUNK_SIZE = 500
CHUNK_OVERLAP = 75


def clean_page_text(text):
    """
    Apply conservative cleaning to extracted PDF text.

    The goal is to repair obvious extraction artefacts
    without changing the meaning of the protocol.
    """

    if not text:
        return ""

    # Normalize common whitespace characters.
    text = (
        text.replace("\u00a0", " ")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )

    # Repair words split across line breaks.
    # Example:
    # "profes-\nsionals" -> "professionals"
    text = re.sub(
        r"(?<=[A-Za-z])-\s*\n\s*(?=[a-z])",
        "",
        text,
    )

    # Replace remaining line breaks with spaces.
    text = re.sub(
        r"\s*\n\s*",
        " ",
        text,
    )

    # Collapse repeated whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def extract_pdf_pages(pdf_bytes):
    """
    Extract text from an uploaded PDF page by page.

    Page-level extraction lets us preserve page citations.
    """

    reader = PdfReader(
        io.BytesIO(pdf_bytes)
    )

    page_rows = []

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):

        raw_text = (
            page.extract_text()
            or ""
        )

        clean_text = clean_page_text(
            raw_text
        )

        page_rows.append(
            {
                "page_number": page_number,
                "text": clean_text,
            }
        )

    return pd.DataFrame(
        page_rows
    )


def chunk_uploaded_protocol(
    pdf_bytes,
    document_name="UPLOAD_PROTOCOL",
):
    """
    Convert an uploaded protocol into page-contained chunks.

    The runtime ingestion path uses the same 500-token
    chunk size and 75-token overlap selected during
    project development.
    """

    pages_df = extract_pdf_pages(
        pdf_bytes
    )

    splitter = (
        RecursiveCharacterTextSplitter
        .from_tiktoken_encoder(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
    )

    # Use the same tokenizer family used for
    # chunk-size diagnostics during development.
    encoding = tiktoken.get_encoding(
        "cl100k_base"
    )

    chunk_rows = []

    for _, page_row in pages_df.iterrows():

        page_number = int(
            page_row["page_number"]
        )

        page_text = page_row["text"]

        if not page_text.strip():
            continue

        page_chunks = splitter.split_text(
            page_text
        )

        for chunk_number, chunk_text in enumerate(
            page_chunks,
            start=1,
        ):

            chunk_id = (
                f"{document_name}"
                f"_P{page_number:03d}"
                f"_C{chunk_number:03d}"
            )

            token_count = len(
                encoding.encode(
                    chunk_text
                )
            )

            chunk_rows.append(
                {
                    "chunk_id": chunk_id,
                    "document_name": document_name,
                    "page_number": page_number,
                    "text": chunk_text,
                    "token_count": token_count,
                }
            )

    chunks_df = pd.DataFrame(
        chunk_rows
    )

    if chunks_df.empty:
        raise ValueError(
            "No usable text could be extracted from the uploaded PDF."
        )

    return chunks_df


def build_uploaded_collection(
    chunks_df,
    openai_client=None,
):
    """
    Embed uploaded chunks and store them in an isolated,
    temporary in-memory Chroma collection.

    The uploaded document is never added to the evaluated
    persistent protocol collection.
    """

    if chunks_df.empty:
        raise ValueError(
            "Cannot build an index from an empty chunk table."
        )

    if openai_client is None:
        openai_client = OpenAI()

    # Generate embeddings for all uploaded chunks.
    embedding_response = (
        openai_client.embeddings.create(
            model=EMBEDDING_MODEL_NAME,
            input=chunks_df["text"].tolist(),
        )
    )

    embeddings = [
        item.embedding
        for item in embedding_response.data
    ]

    # Create an isolated in-memory Chroma database.
    chroma_client = (
        chromadb.EphemeralClient()
    )

    collection_name = (
        "uploaded_protocol_"
        f"{uuid.uuid4().hex[:12]}"
    )

    collection = (
        chroma_client.create_collection(
            name=collection_name
        )
    )

    # Store chunks with the metadata required by
    # our existing retrieval pipeline.
    collection.add(
        ids=chunks_df[
            "chunk_id"
        ].tolist(),
        embeddings=embeddings,
        documents=chunks_df[
            "text"
        ].tolist(),
        metadatas=[
            {
                "document_name": row[
                    "document_name"
                ],
                "page_number": int(
                    row["page_number"]
                ),
            }
            for _, row in chunks_df.iterrows()
        ],
    )

    return collection