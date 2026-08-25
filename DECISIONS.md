# Engineering Decisions

This document records the main design decisions behind the Clinical Trial Protocol Intelligence Copilot.

The project prioritises retrieval quality, evidence traceability and fail-closed behaviour over adding unnecessary architectural complexity.

## 1. Keep chunks inside individual PDF pages

### Decision

Chunks are page-contained and never cross PDF page boundaries.

### Why

Clinical-trial protocols are frequently reviewed by page number. Allowing a chunk to span two pages would make evidence citations harder to interpret and verify.

Page-contained chunks provide:

- direct page traceability
- simpler evidence inspection
- stable chunk identifiers
- clearer debugging of retrieval failures

### Trade-off

A sentence or concept split across two pages may be divided between chunks.

This is accepted in exchange for stronger citation traceability.

## 2. Use approximately 500-token chunks with 75-token overlap

### Decision

The evaluated corpus uses:

```text
Target chunk size: 500 tokens
Overlap: 75 tokens
```

### Why

Two chunking configurations were compared during development.

The smaller configuration provided more focused retrieval while still preserving enough local context for protocol questions.

The overlap helps preserve information near chunk boundaries.

### Trade-off

Smaller chunks improve retrieval precision but may separate information that is distributed across a long section.

## 3. Reject the original MiniLM embedding baseline

### Decision

`all-MiniLM-L6-v2` was tested but not selected for the final retrieval system.

### Why

Its effective sequence-length limit was too small for much of the selected chunking configuration.

A large proportion of the corpus chunks exceeded the model's supported context length.

Using an embedding model that silently truncates substantial parts of the evidence would weaken retrieval reliability.

### Final choice

The production semantic index uses:

```text
OpenAI text-embedding-3-small
1536 dimensions
```

## 4. Combine semantic retrieval with BM25

### Decision

The production retriever uses both:

```text
Semantic retrieval
+
BM25 lexical retrieval
```

### Why

Semantic search works well when the wording of the question differs from the wording in the protocol.

BM25 is strong when exact terminology, abbreviations, numbers or unusual phrases matter.

During development, some questions were retrieved well by one method and poorly by the other.

Using both improves robustness.

### Example

A retrieval failure analysis showed cases where BM25 ranked the correct evidence very highly while semantic search ranked it much lower.

This demonstrated that semantic retrieval alone was not sufficient.

## 5. Use Reciprocal Rank Fusion instead of combining raw scores

### Decision

Semantic and BM25 results are combined using Reciprocal Rank Fusion (RRF).

### Why

Semantic similarity scores and BM25 scores are produced by different scoring systems and are not directly comparable.

Adding or averaging those raw values would introduce an arbitrary weighting problem.

RRF combines the ranked positions instead.

This allows both retrieval methods to contribute without pretending their raw scores share the same scale.

## 6. Retrieve a wider candidate pool before reranking

### Decision

The production retrieval configuration uses:

```text
20 semantic candidates
20 BM25 candidates
```

before fusion and reranking.

### Why

Initial evaluation showed that the correct evidence was often present within the wider candidate set even when it did not appear in the final Top-5.

This indicated that many failures were ranking problems rather than complete retrieval failures.

Increasing candidate recall gave the reranker more opportunities to promote the correct evidence.

## 7. Add a cross-encoder reranker

### Decision

The fused candidate set is reranked using:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

### Why

The original hybrid system often retrieved the correct evidence somewhere in the candidate set but did not rank it highly enough.

The cross-encoder evaluates the question and candidate passage together, allowing more precise relevance scoring.

### Measured effect

On the controlled 40-question benchmark:

| Metric | Baseline Hybrid | With Reranker |
|---|---:|---:|
| Hit@1 | 60.0% | 70.0% |
| Hit@3 | 73.3% | 86.7% |
| Hit@5 | 80.0% | 93.3% |
| Hit@10 | 93.3% | 96.7% |

The largest practical improvement occurred at Top-3 and Top-5, which matters because only a small evidence set is eventually provided to the LLM.

## 8. Send only the Top-5 evidence passages to generation

### Decision

The final generation context contains five evidence passages.

### Why

Providing too much retrieved text can introduce:

- irrelevant context
- conflicting passages
- unnecessary token usage
- greater opportunity for unsupported synthesis

The retrieval and reranking stages are therefore responsible for reducing the candidate set before generation.

### Principle

The LLM should reason over a small, high-quality evidence set rather than a large unfiltered context.

## 9. Constrain generation to retrieved evidence

### Decision

The generation prompt instructs the model to answer only from the supplied evidence.

If the evidence is insufficient, the expected response is:

```text
Insufficient Evidence
```

### Why

The purpose of the system is document intelligence, not general medical question answering.

Information that may be true in general is still unsupported if it is not present in the selected protocol evidence.

## 10. Require inline chunk-ID citations

### Decision

Generated factual answers use citations such as:

```text
[PROTO_003_P005_C001]
```

### Why

The identifier provides traceability to:

```text
document
page
chunk
```

This makes evidence inspectable and allows deterministic validation after generation.

## 11. Validate citations deterministically

### Decision

Citation checking is performed using Python rather than asking another LLM whether the citations look valid.

### Rule

Every cited chunk ID must exist in the final retrieved evidence supplied to the model.

### Why

This check is objective and deterministic.

An LLM-based validator would add unnecessary cost and could itself hallucinate.

### Important limitation

A valid citation proves that the cited passage was retrieved.

It does **not** prove that every claim attached to that citation is semantically supported.

## 12. Add a critical quantitative-fact safeguard

### Decision

The production pipeline checks selected quantitative facts in generated answers against cited evidence.

Examples include:

```text
percentages
durations
participant counts
patient counts
sessions
visits
study groups
```

### Why

Manual evaluation identified a case where the model used a valid citation but added an unsupported quantitative detail.

Citation validation alone therefore did not catch the problem.

The additional check is designed to catch this type of failure.

### Scope

This is a targeted deterministic safeguard.

It is not a complete semantic entailment system and should not be described as one.

## 13. Fail closed when validation fails

### Decision

If the reliability gate fails, the generated answer is not returned to the user.

Instead the application returns:

```text
Insufficient Evidence
```

### Why

For evidence-grounded document intelligence, returning no answer is preferable to returning unsupported information.

The raw generated response may still be retained internally for debugging.

## 14. Keep uploaded protocols separate from the demo index

### Decision

Runtime uploads are processed into a separate temporary session-specific Chroma index.

They are not added to the persistent demo collection.

### Why

This prevents uploaded documents from contaminating the evaluated demo corpus and allows each uploaded protocol to be queried independently.

It also keeps runtime ingestion logically separate from the pre-built benchmark collection.

## 15. Use the same query pipeline for demo and uploaded documents

### Decision

Once an uploaded document has been processed, both document modes use the same:

```text
semantic retrieval
BM25 retrieval
RRF
cross-encoder reranking
Top-5 evidence selection
grounded generation
citation validation
quantitative safeguard
reliability gate
```

### Why

A separate retrieval pipeline for uploads would make generalisation claims weaker and increase maintenance complexity.

The upload feature therefore tests the same core RAG system on documents that were not part of the original corpus.

## 16. Evaluate on an untouched sixth protocol

### Decision

After the main system was completed, a new clinical-trial protocol was uploaded and evaluated without changing retrieval or generation settings.

### Result

```text
9 / 10 questions correct
2 / 2 unsupported questions correctly abstained
```

### Why

The original 40-question benchmark influenced retrieval development.

A separate unseen protocol therefore provides a more realistic test of runtime generalisation.

### Observed failure

One multi-fact question asked about both IVR and group-call frequency.

The system correctly retrieved and answered the IVR frequency but incorrectly stated that the group-call frequency was not specified.

The protocol contained the weekly group-call frequency in a separate passage.

This failure is retained as a documented limitation.

## 17. Do not add agents or graph orchestration without a clear need

### Decision

The current system does not use:

```text
LangGraph
multi-agent orchestration
GraphRAG
fine-tuning
```

### Why

The task is fundamentally:

```text
retrieve
rerank
generate
validate
```

Adding agent orchestration would increase complexity without addressing the main observed failure modes.

The project prioritises measurable retrieval and grounding improvements over architectural novelty.

## 18. Keep the application focused on research-document intelligence

### Decision

The application is intentionally scoped to protocol exploration.

It is not positioned as:

- a diagnostic system
- a treatment recommendation system
- a clinical decision-support tool
- a regulatory decision system

### Why

The engineering goal is reliable retrieval and evidence-grounded question answering over research documents.

Maintaining a narrow scope makes the system's behaviour and evaluation easier to reason about.

## Summary

The central engineering principle of this project is:

> **Retrieve broadly, rank carefully, generate narrowly, validate deterministically, and abstain when support is insufficient.**

The architecture is intentionally designed around reliability and evidence traceability rather than maximum answer coverage.