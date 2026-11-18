# V13 Validation Proof

## Core hybrid benchmark

| Retriever | Hit@1 | Hit@3 | Hit@5 | MRR |
| --- | ---: | ---: | ---: | ---: |
| Dense | 0.90 | 1.00 | 1.00 | 0.9333 |
| BM25 | 0.80 | 1.00 | 1.00 | 0.8833 |
| Hybrid | 0.90 | 1.00 | 1.00 | **0.9500** |

## Fresh-process backend

```text
Backend cache independent: PASS
BM25 independent: PASS
Pinecone dense search independent: PASS
Hybrid retrieval independent: PASS
Cross-provider retrieval independent: PASS

Tests passed: 5/5
V13 FRESH PROCESS BACKEND: PASS
V13 PRODUCTION BACKEND INDEPENDENCE: PASS
```

## Fresh-process streaming

```text
Callback received tokens: PASS
Raw answer created: PASS
Citation validation: PASS
Trusted final citations: PASS
SQLite persisted 2 messages: PASS

Tests passed: 5/5
V13 FRESH PROCESS STREAMING: PASS
V13 PRODUCTION LIVE STREAMING: PASS
```

## Streamlit production validation

Validated in the browser:

- global retrieval;
- single-provider retrieval;
- cross-provider retrieval;
- live token streaming;
- trusted deterministic citations;
- retrieved evidence inspection;
- SQLite persistent memory;
- conversational follow-up rewriting;
- comparison-aware global routing;
- citation hardening.

## Corpus

27 PDFs, 3,451 pages, 11,502 chunks across AWS, Azure, GCP, Oracle, and IBM.
