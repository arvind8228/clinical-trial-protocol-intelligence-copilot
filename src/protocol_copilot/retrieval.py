import re
from functools import lru_cache

import pandas as pd

from openai import OpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from protocol_copilot.config import (
    EMBEDDING_MODEL_NAME,
    RERANKER_MODEL_NAME,
    SEMANTIC_CANDIDATE_K,
    BM25_CANDIDATE_K,
    FINAL_TOP_K,
    RRF_K,
)

from protocol_copilot.data import (
    load_chunks,
    load_chroma_collection,
)


def tokenize(text):
    """
    Tokenize text for BM25 keyword retrieval.
    """

    return re.findall(
        r"[A-Za-z0-9]+",
        text.lower(),
    )


@lru_cache(maxsize=1)
def load_reranker():
    """
    Load the cross-encoder reranker once and reuse it.
    """

    return CrossEncoder(
        RERANKER_MODEL_NAME
    )


def retrieve_candidates(
    question,
    document_name,
    chunks_df,
    collection,
    openai_client,
    semantic_k=SEMANTIC_CANDIDATE_K,
    bm25_k=BM25_CANDIDATE_K,
):
    """
    Retrieve candidate chunks using semantic search and BM25,
    then combine their rankings with Reciprocal Rank Fusion.
    """

    # Keep retrieval inside the selected protocol
    document_chunks = (
        chunks_df[
            chunks_df["document_name"] == document_name
        ]
        .copy()
        .reset_index(drop=True)
    )

    if document_chunks.empty:
        raise ValueError(
            f"No chunks found for protocol: {document_name}"
        )


    # Create an embedding for the user's question
    embedding_response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL_NAME,
        input=question,
    )

    query_embedding = (
        embedding_response
        .data[0]
        .embedding
    )


    # Semantic retrieval from Chroma
    semantic_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(
            semantic_k,
            len(document_chunks),
        ),
        where={
            "document_name": document_name
        },
        include=[
            "metadatas",
            "distances",
        ],
    )


    semantic_ranks = {
        chunk_id: rank
        for rank, chunk_id in enumerate(
            semantic_results["ids"][0],
            start=1,
        )
    }


    # Build BM25 only over the selected protocol
    tokenized_chunks = [
        tokenize(text)
        for text in document_chunks["text"]
    ]

    bm25 = BM25Okapi(
        tokenized_chunks
    )

    query_tokens = tokenize(
        question
    )

    bm25_scores = bm25.get_scores(
        query_tokens
    )


    # Keep the strongest BM25 candidates
    bm25_order = (
        bm25_scores
        .argsort()[::-1]
        [:min(bm25_k, len(document_chunks))]
    )


    bm25_ranks = {}

    for rank, row_index in enumerate(
        bm25_order,
        start=1,
    ):

        chunk_id = document_chunks.loc[
            row_index,
            "chunk_id",
        ]

        bm25_ranks[chunk_id] = rank


    # Candidate pool is the union of both retrievers
    candidate_ids = (
        set(semantic_ranks)
        | set(bm25_ranks)
    )


    candidate_rows = []


    for chunk_id in candidate_ids:

        semantic_rank = semantic_ranks.get(
            chunk_id
        )

        bm25_rank = bm25_ranks.get(
            chunk_id
        )


        # Reciprocal Rank Fusion
        rrf_score = 0.0

        if semantic_rank is not None:
            rrf_score += (
                1 / (RRF_K + semantic_rank)
            )

        if bm25_rank is not None:
            rrf_score += (
                1 / (RRF_K + bm25_rank)
            )


        chunk_row = document_chunks[
            document_chunks["chunk_id"] == chunk_id
        ].iloc[0]


        candidate_rows.append(
            {
                "chunk_id": chunk_id,
                "document_name": document_name,
                "page_number": chunk_row["page_number"],
                "text": chunk_row["text"],
                "token_count": chunk_row["token_count"],
                "semantic_rank": semantic_rank,
                "bm25_rank": bm25_rank,
                "rrf_score": rrf_score,
            }
        )


    candidates_df = (
        pd.DataFrame(candidate_rows)
        .sort_values(
            "rrf_score",
            ascending=False,
        )
        .reset_index(drop=True)
    )


    candidates_df["hybrid_rank"] = (
        candidates_df.index + 1
    )


    return candidates_df


def rerank_candidates(
    question,
    candidates_df,
    top_k=FINAL_TOP_K,
):
    """
    Rerank hybrid candidates by directly scoring each
    question-chunk pair with a cross-encoder.
    """

    reranker = load_reranker()


    question_chunk_pairs = [
        [
            question,
            chunk_text,
        ]
        for chunk_text in candidates_df["text"]
    ]


    reranker_scores = reranker.predict(
        question_chunk_pairs
    )


    reranked_df = (
        candidates_df
        .copy()
    )

    reranked_df["reranker_score"] = (
        reranker_scores
    )


    reranked_df = (
        reranked_df
        .sort_values(
            "reranker_score",
            ascending=False,
        )
        .reset_index(drop=True)
    )


    reranked_df["reranker_rank"] = (
        reranked_df.index + 1
    )


    return reranked_df.head(
        top_k
    )


def retrieve_evidence(
    question,
    document_name,
    chunks_df=None,
    collection=None,
    openai_client=None,
    top_k=FINAL_TOP_K,
):
    """
    Run the complete retrieval pipeline:

    semantic search + BM25
    → RRF candidate fusion
    → cross-encoder reranking
    → final evidence
    """

    # Load reusable resources when they are not supplied
    if chunks_df is None:
        chunks_df = load_chunks()

    if collection is None:
        collection = load_chroma_collection()

    if openai_client is None:
        openai_client = OpenAI()


    candidates_df = retrieve_candidates(
        question=question,
        document_name=document_name,
        chunks_df=chunks_df,
        collection=collection,
        openai_client=openai_client,
    )


    evidence_df = rerank_candidates(
        question=question,
        candidates_df=candidates_df,
        top_k=top_k,
    )


    return evidence_df