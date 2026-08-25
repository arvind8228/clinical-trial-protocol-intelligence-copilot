import re

from protocol_copilot.config import (
    ABSTENTION_MESSAGE,
)


# Match citation IDs from both:
# - Pre-indexed protocols: PROTO_003_P005_C001
# - Uploaded protocols: UPLOAD_TEST_P005_C001
CITATION_PATTERN = re.compile(
    r"\[([A-Z][A-Z0-9_]*_P\d+_C\d+)\]"
)


# Convert common written numbers into digits so that
# "seven hours" and "7 hours" can be compared.
NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
}


# Normalize different ways of writing the same unit.
UNIT_ALIASES = {
    "%": "percent",
    "percent": "percent",
    "percentage": "percent",

    "minute": "minute",
    "minutes": "minute",
    "min": "minute",
    "mins": "minute",

    "hour": "hour",
    "hours": "hour",
    "h": "hour",
    "hr": "hour",
    "hrs": "hour",

    "day": "day",
    "days": "day",

    "week": "week",
    "weeks": "week",

    "month": "month",
    "months": "month",

    "year": "year",
    "years": "year",

    "participant": "participant",
    "participants": "participant",

    "patient": "patient",
    "patients": "patient",

    "subject": "subject",
    "subjects": "subject",

    "session": "session",
    "sessions": "session",

    "visit": "visit",
    "visits": "visit",

    "arm": "arm",
    "arms": "arm",

    "group": "group",
    "groups": "group",
}


# Find important quantitative facts such as:
# 7 hours
# 7-h
# 3 months
# 20 participants
# 6 sessions
CRITICAL_FACT_PATTERN = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*"
    r"(%|percentage|percent|"
    r"minutes?|mins?|"
    r"hours?|hrs?|h|"
    r"days?|weeks?|months?|years?|"
    r"participants?|patients?|subjects?|"
    r"sessions?|visits?|arms?|groups?)"
    r"(?=\b|[\s.,;:)\]]|$)",
    re.IGNORECASE,
)


def extract_citations(answer):
    """
    Extract chunk IDs cited in a generated answer.
    """

    return CITATION_PATTERN.findall(
        answer
    )


def validate_citations(
    answer,
    retrieved_chunk_ids,
):
    """
    Check that citations in the generated answer refer only
    to chunks that were actually retrieved for the model.
    """

    # Exact abstention is allowed without citations.
    if answer.strip() == ABSTENTION_MESSAGE:
        return {
            "is_abstention": True,
            "cited_chunk_ids": [],
            "citation_count": 0,
            "has_citations": False,
            "invalid_citations": [],
            "all_citations_valid": True,
            "citation_check_passed": True,
        }


    # Extract all citations from the answer.
    cited_chunk_ids = extract_citations(
        answer
    )


    # Find citations that were not part of retrieved evidence.
    invalid_citations = [
        chunk_id
        for chunk_id in cited_chunk_ids
        if chunk_id not in retrieved_chunk_ids
    ]


    has_citations = (
        len(cited_chunk_ids) > 0
    )


    all_citations_valid = (
        len(invalid_citations) == 0
    )


    # A supported answer must contain at least one citation
    # and every citation must come from retrieved evidence.
    citation_check_passed = (
        has_citations
        and all_citations_valid
    )


    return {
        "is_abstention": False,
        "cited_chunk_ids": cited_chunk_ids,
        "citation_count": len(cited_chunk_ids),
        "has_citations": has_citations,
        "invalid_citations": invalid_citations,
        "all_citations_valid": all_citations_valid,
        "citation_check_passed": citation_check_passed,
    }


def normalize_number_words(text):
    """
    Normalize written numbers and hyphenated expressions.

    Examples:
    "seven hours" -> "7 hours"
    "7-h" -> "7 h"
    "3-month" -> "3 month"
    """

    normalized_text = (
        text.lower()
        .replace("–", "-")
        .replace("—", "-")
        .replace("-", " ")
    )


    # Replace written numbers with digits.
    for word, digit in NUMBER_WORDS.items():
        normalized_text = re.sub(
            rf"\b{word}\b",
            digit,
            normalized_text,
        )


    return normalized_text


def extract_critical_facts(text):
    """
    Extract important quantitative facts from text.

    Examples:
    "7 hours" -> "7 hour"
    "7-h" -> "7 hour"
    "3 months" -> "3 month"
    "20 participants" -> "20 participant"
    """

    normalized_text = normalize_number_words(
        text
    )


    critical_facts = []


    for number, unit in CRITICAL_FACT_PATTERN.findall(
        normalized_text
    ):

        canonical_unit = UNIT_ALIASES[
            unit.lower()
        ]


        critical_facts.append(
            f"{number} {canonical_unit}"
        )


    # Remove duplicates while preserving order.
    return list(
        dict.fromkeys(
            critical_facts
        )
    )


def validate_critical_fact_grounding(
    answer,
    evidence_df,
):
    """
    Check that important quantitative facts in the answer
    also appear in the chunks cited by that answer.

    This is an additional deterministic grounding safeguard.

    It does not prove complete semantic correctness, but it
    helps detect unsupported numerical details such as an
    invented duration, participant count, follow-up period,
    session count, or visit count.
    """

    # An abstention contains no factual answer to validate.
    if answer.strip() == ABSTENTION_MESSAGE:
        return {
            "critical_facts": [],
            "evidence_critical_facts": [],
            "unsupported_critical_facts": [],
            "critical_fact_check_passed": True,
        }


    # Find the chunks cited by the generated answer.
    cited_chunk_ids = extract_citations(
        answer
    )


    # Create a lookup from chunk ID to evidence text.
    evidence_lookup = {
        row["chunk_id"]: row["text"]
        for _, row in evidence_df.iterrows()
    }


    # Use only evidence that the answer actually cited.
    cited_evidence_text = "\n".join(
        evidence_lookup[chunk_id]
        for chunk_id in cited_chunk_ids
        if chunk_id in evidence_lookup
    )


    # Extract important quantitative facts from the answer.
    answer_facts = extract_critical_facts(
        answer
    )


    # Extract the same kind of facts from cited evidence.
    evidence_facts = extract_critical_facts(
        cited_evidence_text
    )


    evidence_fact_set = set(
        evidence_facts
    )


    # Any quantitative fact present in the answer but absent
    # from cited evidence is treated as unsupported.
    unsupported_facts = [
        fact
        for fact in answer_facts
        if fact not in evidence_fact_set
    ]


    return {
        "critical_facts": answer_facts,
        "evidence_critical_facts": evidence_facts,
        "unsupported_critical_facts": unsupported_facts,
        "critical_fact_check_passed": (
            len(unsupported_facts) == 0
        ),
    }