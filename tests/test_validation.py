import pandas as pd

from protocol_copilot.validation import (
    extract_citations,
    extract_critical_facts,
    validate_citations,
    validate_critical_fact_grounding,
)


def test_extract_citations():
    answer = (
        "Follow-up was 3 months "
        "[PROTO_001_P001_C002]."
    )

    citations = extract_citations(
        answer
    )

    assert citations == [
        "PROTO_001_P001_C002"
    ]


def test_valid_retrieved_citation_passes():
    result = validate_citations(
        answer=(
            "Follow-up was 3 months. "
            "[PROTO_001_P001_C002]"
        ),
        retrieved_chunk_ids=[
            "PROTO_001_P001_C002"
        ],
    )

    assert result["has_citations"] is True
    assert result["all_citations_valid"] is True
    assert result["citation_check_passed"] is True


def test_unretrieved_citation_fails():
    result = validate_citations(
        answer=(
            "Unsupported claim. "
            "[PROTO_999_P001_C001]"
        ),
        retrieved_chunk_ids=[
            "PROTO_001_P001_C002"
        ],
    )

    assert result["all_citations_valid"] is False
    assert result["citation_check_passed"] is False

    assert result["invalid_citations"] == [
        "PROTO_999_P001_C001"
    ]


def test_supported_answer_without_citation_fails():
    result = validate_citations(
        answer="Follow-up was 3 months.",
        retrieved_chunk_ids=[
            "PROTO_001_P001_C002"
        ],
    )

    assert result["has_citations"] is False
    assert result["citation_check_passed"] is False


def test_abstention_passes_without_citation():
    result = validate_citations(
        answer="Insufficient Evidence",
        retrieved_chunk_ids=[
            "PROTO_001_P001_C002"
        ],
    )

    assert result["is_abstention"] is True
    assert result["has_citations"] is False
    assert result["citation_check_passed"] is True


def test_extract_critical_facts_normalizes_numbers():
    facts = extract_critical_facts(
        "Training included a 7-h programme, "
        "a 2-h update, and a 3-month follow-up."
    )

    assert "7 hour" in facts
    assert "2 hour" in facts
    assert "3 month" in facts


def test_supported_critical_fact_passes():
    evidence_df = pd.DataFrame(
        [
            {
                "chunk_id": "PROTO_003_P005_C001",
                "text": (
                    "Primary-care professionals participated "
                    "in a 7-h training programme for coaching."
                ),
            }
        ]
    )

    result = validate_critical_fact_grounding(
        answer=(
            "The coaching training lasted 7 hours. "
            "[PROTO_003_P005_C001]"
        ),
        evidence_df=evidence_df,
    )

    assert result[
        "critical_fact_check_passed"
    ] is True

    assert result[
        "unsupported_critical_facts"
    ] == []


def test_unsupported_extra_detail_fails():
    evidence_df = pd.DataFrame(
        [
            {
                "chunk_id": "PROTO_004_P006_C001",
                "text": (
                    "Participants received six one-to-one "
                    "sessions over a 10-week period."
                ),
            }
        ]
    )

    result = validate_critical_fact_grounding(
        answer=(
            "Participants received six sessions, "
            "each lasting up to 1 hour. "
            "[PROTO_004_P006_C001]"
        ),
        evidence_df=evidence_df,
    )

    assert result[
        "critical_fact_check_passed"
    ] is False

    assert "1 hour" in result[
        "unsupported_critical_facts"
    ]
def test_uploaded_protocol_citation_is_recognized():
    citations = extract_citations(
        "Supported result [UPLOAD_TEST_P005_C001]"
    )

    assert citations == [
        "UPLOAD_TEST_P005_C001"
    ]