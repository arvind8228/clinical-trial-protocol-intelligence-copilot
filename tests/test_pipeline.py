import pandas as pd

import protocol_copilot.pipeline as pipeline


def make_evidence_df(text):
    """
    Create a small evidence table for pipeline tests.
    """

    return pd.DataFrame(
        [
            {
                "chunk_id": "PROTO_003_P005_C001",
                "document_name": "INTEGRA",
                "page_number": 5,
                "text": text,
            }
        ]
    )


def test_pipeline_returns_supported_answer(monkeypatch):
    evidence_df = make_evidence_df(
        "Professionals participated in a 7-h training programme."
    )

    monkeypatch.setattr(
        pipeline,
        "retrieve_evidence",
        lambda **kwargs: evidence_df,
    )

    monkeypatch.setattr(
        pipeline,
        "generate_grounded_answer",
        lambda **kwargs: (
            "The training programme lasted 7 hours. "
            "[PROTO_003_P005_C001]"
        ),
    )

    result = pipeline.ask_protocol_question(
        question="How long was the training programme?",
        document_name="INTEGRA",
        chunks_df=object(),
        collection=object(),
        openai_client=object(),
    )

    assert result["reliability_gate_passed"] is True

    assert (
        result["answer"]
        == result["raw_answer"]
    )

    assert (
        result["citation_validation"][
            "citation_check_passed"
        ]
        is True
    )

    assert (
        result["critical_fact_validation"][
            "critical_fact_check_passed"
        ]
        is True
    )


def test_pipeline_fails_closed_on_unsupported_detail(
    monkeypatch,
):
    evidence_df = make_evidence_df(
        "Participants received six sessions "
        "over a 10-week period."
    )

    monkeypatch.setattr(
        pipeline,
        "retrieve_evidence",
        lambda **kwargs: evidence_df,
    )

    monkeypatch.setattr(
        pipeline,
        "generate_grounded_answer",
        lambda **kwargs: (
            "Participants received six sessions, "
            "each lasting 1 hour. "
            "[PROTO_003_P005_C001]"
        ),
    )

    result = pipeline.ask_protocol_question(
        question="Describe the intervention.",
        document_name="INTEGRA",
        chunks_df=object(),
        collection=object(),
        openai_client=object(),
    )

    assert result["reliability_gate_passed"] is False

    assert result["answer"] == (
        "Insufficient Evidence"
    )

    assert result["raw_answer"] != (
        "Insufficient Evidence"
    )

    assert "1 hour" in (
        result["critical_fact_validation"][
            "unsupported_critical_facts"
        ]
    )


def test_pipeline_preserves_valid_abstention(
    monkeypatch,
):
    evidence_df = make_evidence_df(
        "The protocol describes study procedures."
    )

    monkeypatch.setattr(
        pipeline,
        "retrieve_evidence",
        lambda **kwargs: evidence_df,
    )

    monkeypatch.setattr(
        pipeline,
        "generate_grounded_answer",
        lambda **kwargs: (
            "Insufficient Evidence"
        ),
    )

    result = pipeline.ask_protocol_question(
        question="What were the final trial results?",
        document_name="INTEGRA",
        chunks_df=object(),
        collection=object(),
        openai_client=object(),
    )

    assert result["answer"] == (
        "Insufficient Evidence"
    )

    assert result["reliability_gate_passed"] is True

    assert (
        result["citation_validation"][
            "is_abstention"
        ]
        is True
    )