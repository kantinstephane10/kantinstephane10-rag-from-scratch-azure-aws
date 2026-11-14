# RAG V8 Validation — Query-Aware Retrieval + Deterministic Citations

## Validation environment

V8 was tested locally in VS Code / Jupyter against the 143-page MBA portfolio PDF using:

- `sentence-transformers/all-MiniLM-L6-v2`
- `cross-encoder/ms-marco-MiniLM-L6-v2`
- ChromaDB
- Ollama `qwen3:4b-instruct`
- 800-character chunks
- 120-character overlap
- same-course neighbor expansion ±1

Observed preprocessing results:

```text
Pages loaded: 143
All chunks: 366
Course chunks: 354
Final indexed chunks: 351
Embedding shape: (351, 384)
Vectors stored: 351
```

## Why V8 was needed

V7 worked well for focused questions, but testing exposed two weaknesses:

1. The LLM could generate the right answer with the wrong course label, for example `[C200, PDF page 115]` for Agentic RAG content that belongs to `MMT2-BEN1`.
2. A Top-K-only retrieval strategy is not sufficient for questions such as “identify all courses...” because a few global chunks cannot prove portfolio-wide coverage.

## V8 decisions

### Query-aware retrieval

Focused questions keep the proven V5.2 path:

```text
Top 8 semantic
→ same-course neighbor expansion
→ CrossEncoder reranking
→ best 3
```

Broad questions use course-aware retrieval:

```text
search every course
→ neighbor expansion
→ rerank inside each course
→ best evidence from every course
```

### Deterministic citations

Ollama no longer decides the final course/page label.

The LLM emits:

```text
[SOURCE 1]
```

Python replaces it using trusted retrieval metadata:

```text
[MMT2-BEN1, PDF page 115]
```

## Observed local result

```text
Focused router: PASS
Broad router: PASS
Five CliftonStrengths: PASS
Agentic RAG technologies: PASS
Wrong C200 citation removed: PASS
Correct MMT2-BEN1 citation: PASS
All six courses searched: PASS
Project Management course found: PASS
Grounded refusal: PASS

Tests passed: 9/9
Accuracy: 100.0%
V8 LOCAL TEST: PASS
```

## Key regression proof

Question:

```text
What technologies were proposed for the Agentic RAG architecture?
```

Observed answer:

```text
Microsoft Foundry Agent Service and Azure AI Search.
[MMT2-BEN1, PDF page 115]
```

The previous incorrect citation `[C200, PDF page 115]` was no longer present.

Question:

```text
Identify all courses in my MBA portfolio that discuss project management.
```

The broad retrieval path represented all six courses before generation and correctly identified `MGT2-KKM1` as the Project Management course.

Question:

```text
What is Microsoft's current stock price?
```

Observed grounded refusal:

```text
I don't have enough information in the provided document to answer that.
```
