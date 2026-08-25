from time import perf_counter

from openai import OpenAI

from protocol_copilot.config import (
    ABSTENTION_MESSAGE,
)

from protocol_copilot.data import (
    load_chunks,
    load_chroma_collection,
)

from protocol_copilot.generation import (
    generate_grounded_answer,
)

from protocol_copilot.retrieval import (
    retrieve_evidence,
)

from protocol_copilot.validation import (
    validate_citations,
    validate_critical_fact_grounding,
)


def ask_protocol_question(
    question,
    document_name,
    chunks_df=None,
    collection=None,
    openai_client=None,
):
    """
    Run the complete protocol question-answering pipeline.

    The pipeline:
    1. Retrieves evidence.
    2. Generates a grounded answer.
    3. Validates citations.
    4. Checks important quantitative claims.
    5. Applies a conservative reliability gate.
    6. Returns the answer, evidence, validation results,
       and latency measurements.
    """

    # Load processed chunks if they were not supplied.
    if chunks_df is None:
        chunks_df = load_chunks()

    # Connect to Chroma if it was not supplied.
    if collection is None:
        collection = load_chroma_collection()

    # Create the OpenAI client if needed.
    if openai_client is None:
        openai_client = OpenAI()

    # Measure retrieval time.
    retrieval_start = perf_counter()

    evidence_df = retrieve_evidence(
        question=question,
        document_name=document_name,
        chunks_df=chunks_df,
        collection=collection,
        openai_client=openai_client,
    )

    retrieval_seconds = (
        perf_counter()
        - retrieval_start
    )

    # Measure generation time.
    generation_start = perf_counter()

    raw_answer = generate_grounded_answer(
        question=question,
        evidence_df=evidence_df,
        openai_client=openai_client,
    )

    generation_seconds = (
        perf_counter()
        - generation_start
    )

    # Start validation timing.
    validation_start = perf_counter()

    # Collect the chunk IDs actually retrieved.
    retrieved_chunk_ids = (
        evidence_df["chunk_id"]
        .tolist()
    )

    # Validate that all citations refer to retrieved evidence.
    citation_validation = validate_citations(
        answer=raw_answer,
        retrieved_chunk_ids=retrieved_chunk_ids,
    )

    # Check that important numerical details in the answer
    # are supported by the cited evidence.
    critical_fact_validation = (
        validate_critical_fact_grounding(
            answer=raw_answer,
            evidence_df=evidence_df,
        )
    )

    validation_seconds = (
        perf_counter()
        - validation_start
    )

    # Both checks must pass.
    reliability_gate_passed = (
        citation_validation[
            "citation_check_passed"
        ]
        and critical_fact_validation[
            "critical_fact_check_passed"
        ]
    )

    # Fail closed if the generated answer does not pass validation.
    if reliability_gate_passed:
        final_answer = raw_answer
    else:
        final_answer = ABSTENTION_MESSAGE

    total_seconds = (
        retrieval_seconds
        + generation_seconds
        + validation_seconds
    )

    return {
        "question": question,
        "document_name": document_name,
        "answer": final_answer,
        "raw_answer": raw_answer,
        "reliability_gate_passed": reliability_gate_passed,
        "citation_validation": citation_validation,
        "critical_fact_validation": critical_fact_validation,
        "evidence": evidence_df,
        "latency": {
            "retrieval_seconds": retrieval_seconds,
            "generation_seconds": generation_seconds,
            "validation_seconds": validation_seconds,
            "total_seconds": total_seconds,
        },
    }