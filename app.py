import hashlib
import re

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from protocol_copilot.config import DOCUMENT_SOURCES_PATH
from protocol_copilot.data import (
    load_chunks,
    load_chroma_collection,
)
from protocol_copilot.ingestion import (
    chunk_uploaded_protocol,
    build_uploaded_collection,
)
from protocol_copilot.pipeline import ask_protocol_question


load_dotenv()


st.set_page_config(
    page_title="Clinical Trial Protocol Intelligence Copilot",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
<style>

:root {
    color-scheme: light;
}

html,
body,
.stApp,
[data-testid="stAppViewContainer"] {
    background: #F7FBFF !important;
    color: #102A43 !important;
}

.block-container {
    max-width: 1120px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

header[data-testid="stHeader"],
[data-testid="stDecoration"],
#MainMenu,
footer {
    display: none !important;
}

.hero-shell {
    position: relative;
    overflow: hidden;
    background: #102A43;
    border-radius: 24px;
    padding: 42px 44px;
    margin-bottom: 30px;
    box-shadow: 0 16px 42px rgba(16, 42, 67, 0.14);
}

.hero-shell::after {
    content: "";
    position: absolute;
    width: 390px;
    height: 390px;
    right: -145px;
    top: -190px;
    border-radius: 50%;
    border: 1px solid rgba(223, 243, 255, 0.10);
    box-shadow:
        0 0 0 50px rgba(223, 243, 255, 0.035),
        0 0 0 100px rgba(223, 243, 255, 0.025);
}

.hero-eyebrow,
.hero-title,
.hero-copy,
.hero-features {
    position: relative;
    z-index: 1;
}

.hero-eyebrow {
    color: #A9D8FF;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    margin-bottom: 12px;
}

.hero-title {
    color: #FFFFFF;
    font-size: 2.3rem;
    font-weight: 750;
    letter-spacing: -0.035em;
    line-height: 1.12;
    max-width: 780px;
    margin-bottom: 14px;
}

.hero-copy {
    color: #D8E8F5;
    font-size: 1rem;
    line-height: 1.65;
    max-width: 760px;
}

.hero-features {
    margin-top: 24px;
}

.hero-feature {
    display: inline-block;
    background: rgba(223, 243, 255, 0.08);
    color: #DFF3FF;
    border: 1px solid rgba(223, 243, 255, 0.13);
    border-radius: 999px;
    padding: 7px 11px;
    margin-right: 7px;
    margin-bottom: 5px;
    font-size: 0.74rem;
    font-weight: 650;
}

.section-kicker {
    color: #2F6FED;
    font-size: 0.73rem;
    font-weight: 750;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    margin-bottom: 5px;
}

.section-title {
    color: #102A43;
    font-size: 1.3rem;
    font-weight: 730;
    margin-bottom: 5px;
}

.section-copy {
    color: #64748B;
    font-size: 0.91rem;
    line-height: 1.55;
    margin-bottom: 16px;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF !important;
    border: 1px solid #D9E6F2 !important;
    border-radius: 18px !important;
    box-shadow: 0 5px 18px rgba(16, 42, 67, 0.035);
}

[data-testid="stWidgetLabel"] p {
    color: #334E68 !important;
    font-size: 0.85rem !important;
    font-weight: 650 !important;
}

[data-testid="stButton"] button[kind="primary"] {
    background: #102A43 !important;
    color: #FFFFFF !important;
    border: 1px solid #102A43 !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    box-shadow: none !important;
}

[data-testid="stButton"] button[kind="primary"]:hover {
    background: #173B5C !important;
    color: #FFFFFF !important;
    border-color: #173B5C !important;
}

[data-testid="stButton"] button[kind="secondary"] {
    background: #FFFFFF !important;
    color: #102A43 !important;
    border: 1px solid #D9E6F2 !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    box-shadow: none !important;
}

[data-testid="stButton"] button[kind="secondary"]:hover {
    background: #EAF4FF !important;
    color: #2457C5 !important;
    border-color: #9CC7F5 !important;
}

[data-testid="stTextInput"] div[data-baseweb="base-input"],
[data-testid="stTextInput"] div[data-baseweb="input"] {
    background: #FFFFFF !important;
    border-color: #D9E6F2 !important;
    border-radius: 11px !important;
}

[data-testid="stTextInput"] input {
    background: #FFFFFF !important;
    color: #102A43 !important;
    -webkit-text-fill-color: #102A43 !important;
}

[data-testid="stTextInput"] input::placeholder {
    color: #94A3B8 !important;
    -webkit-text-fill-color: #94A3B8 !important;
}

[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
[data-testid="stSelectbox"] div[role="button"] {
    background: #FFFFFF !important;
    color: #102A43 !important;
    border-color: #D9E6F2 !important;
    border-radius: 11px !important;
}

[data-testid="stSelectbox"] div[data-baseweb="select"] span {
    color: #102A43 !important;
}

[data-testid="stSelectbox"] svg {
    fill: #64748B !important;
}

div[data-baseweb="popover"],
ul[role="listbox"] {
    background: #FFFFFF !important;
}

li[role="option"] {
    background: #FFFFFF !important;
    color: #102A43 !important;
}

li[role="option"]:hover {
    background: #EAF4FF !important;
    color: #2457C5 !important;
}

[data-testid="stFormSubmitButton"] button {
    background: #2F6FED !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 11px !important;
    font-weight: 700 !important;
    min-height: 2.8rem;
    padding-left: 1.3rem;
    padding-right: 1.3rem;
    box-shadow: 0 5px 14px rgba(47, 111, 237, 0.17);
}

[data-testid="stFormSubmitButton"] button:hover {
    background: #2457C5 !important;
    color: #FFFFFF !important;
    border: none !important;
}

[data-testid="stFileUploaderDropzone"] {
    background: #F2F8FF !important;
    border: 1.5px dashed #9CC7F5 !important;
    border-radius: 16px !important;
    padding: 24px !important;
}

[data-testid="stFileUploaderDropzone"] div,
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] p {
    color: #334E68 !important;
}

[data-testid="stFileUploaderDropzone"] small {
    color: #64748B !important;
}

[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploaderDropzone"] button[kind="secondary"] {
    background: #FFFFFF !important;
    color: #2F6FED !important;
    border: 1px solid #BDD7FF !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    box-shadow: 0 3px 10px rgba(47, 111, 237, 0.08) !important;
}

[data-testid="stFileUploaderDropzone"] button:hover,
[data-testid="stFileUploaderDropzone"] button[kind="secondary"]:hover {
    background: #EAF4FF !important;
    color: #2457C5 !important;
    border-color: #8ABCF2 !important;
}

[data-testid="stFileUploaderFile"] {
    background: #FFFFFF !important;
    border: 1px solid #D9E6F2 !important;
    border-radius: 10px !important;
}

[data-testid="stFileUploaderFile"] * {
    color: #102A43 !important;
}

.document-status,
.status-supported,
.status-insufficient {
    display: inline-block;
    border-radius: 999px;
    padding: 5px 10px;
    font-size: 0.72rem;
    font-weight: 750;
    letter-spacing: 0.04em;
}

.document-status,
.status-supported {
    background: #E7F1FF;
    color: #2457C5;
    border: 1px solid #BDD7FF;
}

.status-insufficient {
    background: #FFF7E6;
    color: #8A5B13;
    border: 1px solid #F2D7A0;
}

.document-status {
    margin-bottom: 7px;
}

.document-subtitle {
    color: #64748B;
    font-size: 0.9rem;
    margin-top: -5px;
    margin-bottom: 10px;
}

.document-meta {
    color: #64748B;
    font-size: 0.84rem;
    margin-top: 5px;
}

.reliability-heading {
    color: #102A43;
    font-size: 1.1rem;
    font-weight: 730;
    margin-top: 20px;
    margin-bottom: 10px;
}

.reliability-card {
    background: #FFFFFF;
    border: 1px solid #D9E6F2;
    border-radius: 15px;
    padding: 16px 17px;
    min-height: 88px;
}

.reliability-label {
    color: #64748B;
    font-size: 0.76rem;
    margin-bottom: 8px;
}

.reliability-value {
    color: #102A43;
    font-size: 0.98rem;
    font-weight: 700;
}

.reliability-check {
    color: #2F6FED;
    margin-right: 5px;
    font-weight: 800;
}

.evidence-page {
    color: #2F6FED;
    font-size: 0.75rem;
    font-weight: 750;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
}

[data-testid="stExpander"] {
    background: #FFFFFF !important;
    border: 1px solid #D9E6F2 !important;
    border-radius: 14px !important;
    overflow: hidden;
}

[data-testid="stExpander"] details {
    background: #FFFFFF !important;
}

[data-testid="stExpander"] summary {
    background: #FFFFFF !important;
    color: #102A43 !important;
}

[data-testid="stExpander"] summary:hover {
    background: #F2F8FF !important;
}

[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span,
[data-testid="stExpander"] summary * {
    color: #102A43 !important;
}

[data-testid="stExpander"] summary svg {
    fill: #64748B !important;
    color: #64748B !important;
}

[data-testid="stMetric"] {
    background: #F2F8FF !important;
    border: 1px solid #D9E6F2 !important;
    border-radius: 12px !important;
    padding: 14px !important;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] * {
    color: #64748B !important;
}

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] div,
[data-testid="stMetricValue"] * {
    color: #102A43 !important;
}

</style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_demo_chunks():
    return load_chunks()


@st.cache_resource
def get_demo_collection():
    return load_chroma_collection()


@st.cache_data
def get_document_sources():
    try:
        return pd.read_csv(
            DOCUMENT_SOURCES_PATH
        )
    except Exception:
        return pd.DataFrame()


def format_protocol_name(
    protocol_name,
):
    return protocol_name.replace(
        "_",
        "-",
    )


def format_uploaded_display_name(
    filename,
):
    stem = filename.rsplit(
        ".",
        1,
    )[0]

    stem = re.sub(
        r"[_-]+",
        " ",
        stem,
    ).strip()

    stem = re.sub(
        r"\bprotocol\b",
        "",
        stem,
        flags=re.IGNORECASE,
    ).strip()

    if not stem:
        return "Uploaded Protocol"

    if len(stem.split()) == 1:
        clean_name = stem.upper()
    else:
        clean_name = stem.title()

    return f"{clean_name} Protocol"


def get_protocol_metadata(
    sources_df,
    protocol_name,
):
    if sources_df.empty:
        return None

    if "short_name" not in sources_df.columns:
        return None

    matches = sources_df[
        sources_df["short_name"]
        == protocol_name
    ]

    if matches.empty:
        return None

    row = matches.iloc[0]

    journal = row.get(
        "journal"
    )

    year = row.get(
        "publication_year"
    )

    metadata_parts = []

    if pd.notna(journal):
        metadata_parts.append(
            str(journal)
        )

    if pd.notna(year):
        try:
            year = int(year)
        except Exception:
            pass

        metadata_parts.append(
            str(year)
        )

    if not metadata_parts:
        return None

    return " · ".join(
        metadata_parts
    )


def create_upload_document_name(
    pdf_bytes,
):
    pdf_hash = hashlib.sha256(
        pdf_bytes
    ).hexdigest()[:10].upper()

    return f"UPLOAD_{pdf_hash}"


def make_citation_display(
    answer,
):
    return re.sub(
        r"\[([A-Z][A-Z0-9_]*_P\d+_C\d+)\]",
        r"`[\1]`",
        answer,
    )


def make_evidence_preview(
    text,
    max_characters=700,
):
    if len(text) <= max_characters:
        return text

    preview = text[
        :max_characters
    ].rsplit(
        " ",
        1,
    )[0]

    return preview + "..."


def prepare_uploaded_protocol(
    pdf_bytes,
):
    file_hash = hashlib.sha256(
        pdf_bytes
    ).hexdigest()

    previous_hash = (
        st.session_state.get(
            "uploaded_file_hash"
        )
    )

    if previous_hash != file_hash:
        document_name = (
            create_upload_document_name(
                pdf_bytes
            )
        )

        chunks_df = (
            chunk_uploaded_protocol(
                pdf_bytes=pdf_bytes,
                document_name=document_name,
            )
        )

        collection = (
            build_uploaded_collection(
                chunks_df=chunks_df,
            )
        )

        st.session_state[
            "uploaded_file_hash"
        ] = file_hash

        st.session_state[
            "uploaded_document_name"
        ] = document_name

        st.session_state[
            "uploaded_chunks_df"
        ] = chunks_df

        st.session_state[
            "uploaded_collection"
        ] = collection

    return (
        st.session_state[
            "uploaded_document_name"
        ],
        st.session_state[
            "uploaded_chunks_df"
        ],
        st.session_state[
            "uploaded_collection"
        ],
    )


if "source_mode" not in st.session_state:
    st.session_state[
        "source_mode"
    ] = "Demo library"


demo_chunks_df = (
    get_demo_chunks()
)

demo_collection = (
    get_demo_collection()
)

sources_df = (
    get_document_sources()
)


demo_protocol_names = sorted(
    demo_chunks_df[
        "document_name"
    ]
    .dropna()
    .unique()
    .tolist()
)


protocol_display_to_internal = {
    format_protocol_name(
        protocol_name
    ): protocol_name
    for protocol_name in demo_protocol_names
}


demo_protocol_display_names = list(
    protocol_display_to_internal.keys()
)


hero_html = (
    '<div class="hero-shell">'
    '<div class="hero-eyebrow">'
    'Clinical research · Evidence intelligence'
    '</div>'
    '<div class="hero-title">'
    'Clinical Trial Protocol Intelligence Copilot'
    '</div>'
    '<div class="hero-copy">'
    'Ask questions about clinical-trial protocols and receive '
    'evidence-grounded answers with direct source citations '
    'and reliability checks.'
    '</div>'
    '<div class="hero-features">'
    '<span class="hero-feature">'
    'Hybrid retrieval'
    '</span>'
    '<span class="hero-feature">'
    'Evidence citations'
    '</span>'
    '<span class="hero-feature">'
    'Reliability checks'
    '</span>'
    '</div>'
    '</div>'
)


st.markdown(
    hero_html,
    unsafe_allow_html=True,
)


active_document_name = None
active_display_name = None
active_document_subtitle = None
active_chunks_df = None
active_collection = None
active_page_count = None
active_chunk_count = None


with st.container(
    border=True
):
    st.markdown(
        '<div class="section-kicker">'
        'Step 01'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">'
        'Choose a protocol source'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-copy">'
        'Use an evaluated demo protocol or upload a new '
        'clinical-trial protocol PDF.'
        '</div>',
        unsafe_allow_html=True,
    )

    source_col1, source_col2, source_spacer = st.columns(
        [1.15, 1.15, 4.7],
        gap="small",
    )

    with source_col1:
        demo_selected = (
            st.session_state[
                "source_mode"
            ]
            == "Demo library"
        )

        if st.button(
            "Demo library",
            key="source_demo",
            type=(
                "primary"
                if demo_selected
                else "secondary"
            ),
            use_container_width=True,
        ):
            if not demo_selected:
                st.session_state[
                    "source_mode"
                ] = "Demo library"

                st.rerun()

    with source_col2:
        upload_selected = (
            st.session_state[
                "source_mode"
            ]
            == "Upload PDF"
        )

        if st.button(
            "Upload PDF",
            key="source_upload",
            type=(
                "primary"
                if upload_selected
                else "secondary"
            ),
            use_container_width=True,
        ):
            if not upload_selected:
                st.session_state[
                    "source_mode"
                ] = "Upload PDF"

                st.rerun()

    source_mode = st.session_state[
        "source_mode"
    ]

    if source_mode == "Demo library":
        selected_display_name = (
            st.selectbox(
                "Protocol",
                demo_protocol_display_names,
            )
        )

        selected_protocol = (
            protocol_display_to_internal[
                selected_display_name
            ]
        )

        active_document_name = (
            selected_protocol
        )

        active_display_name = (
            selected_display_name
        )

        active_document_subtitle = (
            get_protocol_metadata(
                sources_df,
                selected_protocol,
            )
        )

        active_chunks_df = (
            demo_chunks_df
        )

        active_collection = (
            demo_collection
        )

        selected_chunks = (
            demo_chunks_df[
                demo_chunks_df[
                    "document_name"
                ]
                == selected_protocol
            ]
        )

        active_page_count = (
            selected_chunks[
                "page_number"
            ]
            .nunique()
        )

        active_chunk_count = len(
            selected_chunks
        )

    else:
        st.markdown(
            "#### Upload a protocol PDF"
        )

        st.caption(
            "Drop a PDF below or browse from your computer. "
            "The document is indexed only for this application session."
        )

        uploaded_file = (
            st.file_uploader(
                "Protocol PDF",
                type=["pdf"],
                label_visibility="collapsed",
            )
        )

        if uploaded_file is not None:
            pdf_bytes = (
                uploaded_file.getvalue()
            )

            with st.spinner(
                "Extracting, chunking and indexing the protocol..."
            ):
                try:
                    (
                        active_document_name,
                        active_chunks_df,
                        active_collection,
                    ) = prepare_uploaded_protocol(
                        pdf_bytes
                    )

                except Exception as exc:
                    st.error(
                        "The uploaded PDF could not be processed: "
                        f"{exc}"
                    )

                    st.stop()

            active_display_name = (
                format_uploaded_display_name(
                    uploaded_file.name
                )
            )

            active_document_subtitle = (
                "Uploaded PDF · Temporary session index"
            )

            active_page_count = (
                active_chunks_df[
                    "page_number"
                ]
                .nunique()
            )

            active_chunk_count = len(
                active_chunks_df
            )

            st.success(
                "Protocol ready for questions."
            )


if active_document_name is not None:
    with st.container(
        border=True
    ):
        st.markdown(
            '<span class="document-status">'
            'ACTIVE PROTOCOL'
            '</span>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"## {active_display_name}"
        )

        if active_document_subtitle:
            st.markdown(
                '<div class="document-subtitle">'
                f'{active_document_subtitle}'
                '</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            '<div class="document-meta">'
            f'{active_page_count} pages'
            ' &nbsp;&nbsp;·&nbsp;&nbsp; '
            f'{active_chunk_count} chunks'
            ' &nbsp;&nbsp;·&nbsp;&nbsp; '
            'Ready'
            '</div>',
            unsafe_allow_html=True,
        )

    with st.container(
        border=True
    ):
        st.markdown(
            '<div class="section-kicker">'
            'Step 02'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-title">'
            'Ask the protocol'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-copy">'
            'Ask about eligibility criteria, interventions, '
            'outcomes, follow-up schedules or study procedures.'
            '</div>',
            unsafe_allow_html=True,
        )

        with st.form(
            "protocol_question_form",
            clear_on_submit=False,
        ):
            question = (
                st.text_input(
                    "Question",
                    placeholder=(
                        "What would you like to know "
                        "about this protocol?"
                    ),
                    label_visibility="collapsed",
                )
            )

            ask_clicked = (
                st.form_submit_button(
                    "Ask Copilot",
                    type="primary",
                )
            )

    if ask_clicked:
        if not question.strip():
            st.warning(
                "Enter a question before asking the copilot."
            )

        else:
            with st.spinner(
                "Retrieving evidence and running reliability checks..."
            ):
                try:
                    result = (
                        ask_protocol_question(
                            question=question.strip(),
                            document_name=active_document_name,
                            chunks_df=active_chunks_df,
                            collection=active_collection,
                        )
                    )

                except Exception as exc:
                    st.error(
                        "The question could not be processed: "
                        f"{exc}"
                    )

                    st.stop()

            answer = result[
                "answer"
            ]

            is_abstention = (
                answer.strip()
                == "Insufficient Evidence"
            )

            with st.container(
                border=True
            ):
                if is_abstention:
                    st.markdown(
                        '<span class="status-insufficient">'
                        'INSUFFICIENT EVIDENCE'
                        '</span>',
                        unsafe_allow_html=True,
                    )

                else:
                    st.markdown(
                        '<span class="status-supported">'
                        'SUPPORTED'
                        '</span>',
                        unsafe_allow_html=True,
                    )

                st.markdown(
                    "## Answer"
                )

                st.markdown(
                    make_citation_display(
                        answer
                    )
                )

                if is_abstention:
                    st.caption(
                        "The retrieved protocol evidence was not "
                        "sufficient to support a reliable answer."
                    )

            st.markdown(
                '<div class="reliability-heading">'
                'Reliability checks'
                '</div>',
                unsafe_allow_html=True,
            )

            citation_passed = (
                result[
                    "citation_validation"
                ][
                    "citation_check_passed"
                ]
            )

            critical_fact_passed = (
                result[
                    "critical_fact_validation"
                ][
                    "critical_fact_check_passed"
                ]
            )

            gate_passed = (
                result[
                    "reliability_gate_passed"
                ]
            )

            (
                reliability_col1,
                reliability_col2,
                reliability_col3,
            ) = st.columns(
                3,
                gap="medium",
            )

            with reliability_col1:
                st.markdown(
                    '<div class="reliability-card">'
                    '<div class="reliability-label">'
                    'Citation validation'
                    '</div>'
                    '<div class="reliability-value">'
                    '<span class="reliability-check">'
                    '✓'
                    '</span>'
                    f'{"Passed" if citation_passed else "Failed"}'
                    '</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )

            with reliability_col2:
                st.markdown(
                    '<div class="reliability-card">'
                    '<div class="reliability-label">'
                    'Critical quantitative facts'
                    '</div>'
                    '<div class="reliability-value">'
                    '<span class="reliability-check">'
                    f'{"✓" if critical_fact_passed else "!"}'
                    '</span>'
                    f'{"Grounded" if critical_fact_passed else "Flagged"}'
                    '</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )

            with reliability_col3:
                st.markdown(
                    '<div class="reliability-card">'
                    '<div class="reliability-label">'
                    'Reliability gate'
                    '</div>'
                    '<div class="reliability-value">'
                    '<span class="reliability-check">'
                    f'{"✓" if gate_passed else "!"}'
                    '</span>'
                    f'{"Passed" if gate_passed else "Blocked"}'
                    '</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )

            cited_chunk_ids = (
                result[
                    "citation_validation"
                ][
                    "cited_chunk_ids"
                ]
            )

            cited_evidence_df = (
                result[
                    "evidence"
                ][
                    result[
                        "evidence"
                    ][
                        "chunk_id"
                    ].isin(
                        cited_chunk_ids
                    )
                ]
            )

            st.markdown(
                "### Evidence used"
            )

            st.caption(
                "Protocol passages directly cited "
                "by the generated answer."
            )

            if cited_evidence_df.empty:
                if is_abstention:
                    st.info(
                        "No evidence passage was cited "
                        "because the system abstained."
                    )

                else:
                    st.warning(
                        "No cited evidence was available."
                    )

            else:
                for _, row in (
                    cited_evidence_df.iterrows()
                ):
                    with st.container(
                        border=True
                    ):
                        st.markdown(
                            '<div class="evidence-page">'
                            f'PAGE {row["page_number"]}'
                            '</div>',
                            unsafe_allow_html=True,
                        )

                        st.markdown(
                            f"`{row['chunk_id']}`"
                        )

                        st.write(
                            row["text"]
                        )

            other_evidence_df = (
                result[
                    "evidence"
                ][
                    ~result[
                        "evidence"
                    ][
                        "chunk_id"
                    ].isin(
                        cited_chunk_ids
                    )
                ]
            )

            if not other_evidence_df.empty:
                with st.expander(
                    "Additional retrieved context"
                ):
                    st.caption(
                        "These passages were retrieved into the "
                        "final evidence set but were not cited "
                        "in the answer."
                    )

                    for index, (_, row) in enumerate(
                        other_evidence_df.iterrows()
                    ):
                        st.markdown(
                            f"**Page {row['page_number']}** · "
                            f"`{row['chunk_id']}`"
                        )

                        st.write(
                            make_evidence_preview(
                                row["text"]
                            )
                        )

                        if index < (
                            len(other_evidence_df)
                            - 1
                        ):
                            st.divider()

            with st.expander(
                "How was this answer produced?"
            ):
                st.markdown(
                    """
**Retrieval**

Semantic search + BM25  
→ Reciprocal Rank Fusion  
→ Cross-encoder reranking  
→ Top-5 evidence

**Generation and validation**

Evidence-grounded generation  
→ Citation validation  
→ Critical-fact grounding check
                    """
                )

                latency = result[
                    "latency"
                ]

                (
                    latency_col1,
                    latency_col2,
                    latency_col3,
                ) = st.columns(
                    3
                )

                latency_col1.metric(
                    "Evidence retrieval",
                    f"{latency['retrieval_seconds']:.2f}s",
                )

                latency_col2.metric(
                    "Answer generation",
                    f"{latency['generation_seconds']:.2f}s",
                )

                latency_col3.metric(
                    "End-to-end",
                    f"{latency['total_seconds']:.2f}s",
                )

                if source_mode == "Upload PDF":
                    st.caption(
                        "Temporary document ID: "
                        f"{active_document_name}"
                    )

            st.caption(
                "Research-document assistant only. "
                "Not intended for diagnosis, treatment decisions "
                "or clinical decision-making."
            )
