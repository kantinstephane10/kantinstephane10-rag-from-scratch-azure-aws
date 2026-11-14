# Validation Proof — RAG V4 Ollama

## Branch

`feature/mba-rag-v4-ollama`

## Purpose

This file records the observed successful local run of the V4 MBA RAG pipeline.

## Observed Indexing Result

```text
PDF pages: 143
Original chunks: 366
Course chunks: 354
Indexed chunks: 351
Chunk size: 800
Chunk overlap: 120
Embedding model: sentence-transformers/all-MiniLM-L6-v2
Embedding dimensions: 384
Reranker: cross-encoder/ms-marco-MiniLM-L6-v2
Generation model: qwen3:4b-instruct
Vectors stored: 351
```

## Validation Question

```text
What are my five CliftonStrengths?
```

## Observed Ollama Answer

```text
My five CliftonStrengths are Learner, Strategic, Belief, Context, and Deliberative. [C200, PDF page 10]
```

## Top Retrieved Evidence

```text
Source 1
Course: C200
PDF page: 10
Reranker score: 6.1063
```

The source text contained all five required themes:

```text
Learner
Strategic
Belief
Context
Deliberative
```

## Validation Status

```text
PDF loading: PASS
Chunking: PASS
Embedding: PASS
ChromaDB indexing: PASS
Semantic retrieval: PASS
CrossEncoder reranking: PASS
Ollama generation: PASS
Citation: PASS
Overall: PASS
```

## Privacy Proof

The source PDF is not committed. This branch ignores:

```text
*.pdf
data/text_files/pdf/
chroma_db/
chroma_db_*/
```
