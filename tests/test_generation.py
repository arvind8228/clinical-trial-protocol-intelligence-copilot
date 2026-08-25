import pandas as pd

from protocol_copilot.generation import (
    build_grounded_prompt,
    format_evidence_context,
)


def test_format_evidence_context():
    evidence_df = pd.DataFrame(
        [
            {
                "chunk_id": "PROTO_001_P001_C002",
                "document_name": "CERTAIN",
                "page_number": 1,
                "text": "Participants were followed for 3 months.",
            }
        ]
    )

    context = format_evidence_context(
        evidence_df
    )

    assert "Source 1" in context
    assert "PROTO_001_P001_C002" in context
    assert "CERTAIN" in context
    assert "Page: 1" in context
    assert "Participants were followed for 3 months." in context


def test_grounded_prompt_contains_question_and_evidence():
    prompt = build_grounded_prompt(
        question="What was the follow-up period?",
        context=(
            "Chunk ID: PROTO_001_P001_C002\n"
            "Evidence: Follow-up was 3 months."
        ),
    )

    assert "What was the follow-up period?" in prompt
    assert "Follow-up was 3 months." in prompt


def test_grounded_prompt_requires_abstention():
    prompt = build_grounded_prompt(
        question="What were the final trial results?",
        context="No final trial results are reported.",
    )

    assert "Insufficient Evidence" in prompt
    assert "Do not invent" in prompt


def test_grounded_prompt_requires_exact_citations():
    prompt = build_grounded_prompt(
        question="What was measured?",
        context="Evidence text",
    )

    assert "[CHUNK_ID]" in prompt
    assert "Every factual claim" in prompt