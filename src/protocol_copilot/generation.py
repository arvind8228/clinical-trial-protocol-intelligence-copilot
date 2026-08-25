from openai import OpenAI

from protocol_copilot.config import (
    ABSTENTION_MESSAGE,
    GENERATION_MODEL_NAME,
)


def format_evidence_context(evidence_df):
    """
    Convert retrieved chunks into a structured evidence block
    for the language model.
    """

    evidence_sections = []

    for source_number, (_, row) in enumerate(
        evidence_df.iterrows(),
        start=1,
    ):

        evidence_sections.append(
            f"""Source {source_number}
Chunk ID: {row['chunk_id']}
Protocol: {row['document_name']}
Page: {row['page_number']}
Evidence:
{row['text']}"""
        )

    return "\n\n".join(
        evidence_sections
    )


def build_grounded_prompt(question, context):
    """
    Build the evidence-only prompt used by the RAG system.
    """

    return f"""
You are answering a question about a clinical trial protocol.

Use only the evidence provided below.

Rules:
1. Do not use outside knowledge.
2. Do not invent, assume, or infer missing information.
3. If the evidence does not support the answer, respond exactly:
   {ABSTENTION_MESSAGE}
4. Every factual claim in a supported answer must have a citation.
5. Place each citation directly after the claim it supports.
6. Citations must use the exact format [CHUNK_ID].
7. Preserve differences between groups, time points, outcomes, and study procedures.
8. Do not present ambiguous table or flowchart extraction as certain.
9. Prefer clearly stated evidence over uncertain layout-derived interpretation.
10. Keep the answer concise.

Question:
{question}

Evidence:
{context}
""".strip()


def generate_grounded_answer(
    question,
    evidence_df,
    openai_client=None,
):
    """
    Generate an answer using only the retrieved protocol evidence.
    """

    if openai_client is None:
        openai_client = OpenAI()


    # Convert retrieved chunks into model context
    context = format_evidence_context(
        evidence_df
    )


    # Build the strict evidence-only prompt
    prompt = build_grounded_prompt(
        question,
        context,
    )


    # Generate the grounded answer
    response = openai_client.responses.create(
        model=GENERATION_MODEL_NAME,
        input=prompt,
    )


    return response.output_text.strip()