# Validation Proof — RAG V6 End-to-End Evaluation

## Branch

`feature/mba-rag-v6-end-to-end-evaluation`

## Purpose

This file records the observed V6 end-to-end validation of retrieval, generation, citations, and grounded refusal behavior.

## Pipeline

```text
PDF
→ Recursive Chunking
→ MiniLM
→ ChromaDB
→ Top 8
→ Neighbor Expansion
→ CrossEncoder
→ Top 3
→ Ollama qwen3:4b-instruct
→ Answer
→ Citation
→ Evaluation
```

## Final Results

```text
Total tests: 9
Total passed: 9

Answer correctness:
8/8 (100.0%)

Correct source retrieval:
8/8 (100.0%)

Valid citation accuracy:
8/8 (100.0%)

Grounded refusal accuracy:
1/1 (100.0%)

Overall end-to-end accuracy:
9/9 (100.0%)

Failed tests:
None
```

## Representative Passing Results

### C200

Question:

```text
What are my five CliftonStrengths?
```

Observed answer:

```text
My five CliftonStrengths are Learner, Strategic, Belief, Context, and Deliberative. [C200, PDF page 10]
```

### EHM2

Question:

```text
What two ethical traits did I identify in Satya Nadella?
```

Observed answer:

```text
The two ethical traits I identified in Satya Nadella are empathy and accountability. [EHM2, PDF page 31]
```

### C212-KLM2

Question:

```text
Which country did I recommend as the first international market for Renew Vision Signature?
```

Observed answer:

```text
I recommended Canada as the first international market for the Renew Vision Signature product line. [C212-KLM2, PDF page 62]
```

### MGT2-KKM1

Question:

```text
What was the main schedule concern identified by the earned value analysis?
```

Observed answer:

```text
The main schedule concern identified by the earned value analysis was that the project is behind schedule. This is indicated by the schedule performance index of approximately 0.71. [MGT2-KKM1, PDF page 82]
```

### MMT2-BEN1 — Neighbor Expansion Regression Test

Question:

```text
What technologies were proposed for the Agentic RAG architecture?
```

Observed answer:

```text
The proposed technologies for the Agentic RAG architecture are the Microsoft Foundry Agent Service and Azure AI Search. [MMT2-BEN1, PDF page 115]
```

### C207-QUM4

Question:

```text
What relationship was examined in the linear regression analysis?
```

Observed answer:

```text
The linear regression analysis examined the relationship between monthly nurse well-being program participation rate and monthly nurse attrition rate. [C207-QUM4, PDF page 135]
```

### Grounding / Refusal Test

Question:

```text
What is Microsoft's current stock price?
```

Observed answer:

```text
I don't have enough information in the provided document to answer that.
```

This demonstrates that the model followed the RAG grounding rule instead of answering from outside model knowledge.

## Validation Status

```text
Answer correctness: PASS
Source retrieval: PASS
Citation validation: PASS
Grounded refusal: PASS
Overall: PASS — 100%
```

## Privacy Proof

The private PDF and generated ChromaDB files remain excluded from Git.
