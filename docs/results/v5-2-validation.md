# Validation Proof — RAG V5.2 Neighbor Expansion

## Branch

`feature/mba-rag-v5-2-neighbor-expansion`

## Purpose

This file records the strict retrieval evaluation that validated neighbor expansion.

## Configuration

```text
PDF pages: 143
Indexed chunks: 351
Chunk size: 800
Chunk overlap: 120
Embedding model: sentence-transformers/all-MiniLM-L6-v2
Reranker: cross-encoder/ms-marco-MiniLM-L6-v2
Initial semantic candidates: 8
Neighbor range: ±1 chunk
Final context chunks: 3
```

## Strict Retrieval Evaluation

Twelve questions spanning all six MBA courses were evaluated.

```text
Questions: 12

Hit@1:  9/12  (75.0%)
Hit@3: 12/12 (100.0%)
Hit@5: 12/12 (100.0%)
Hit@8: 12/12 (100.0%)

MRR: 0.875
```

## Proof of the Neighbor-Expansion Fix

The earlier version failed this question:

```text
What technologies were proposed for the Agentic RAG architecture?
```

Without neighbor expansion, the answer-bearing chunk did not appear in the Top 8.

With V5.2:

```text
Rank: 1
PDF page: 115
RAG chunk: 272
Source type: neighbor
Reranker score: 5.0393

Expected evidence found:
- Microsoft Foundry
- Azure AI Search
```

The semantic search found the adjacent introductory chunk. Neighbor expansion added the next answer-bearing chunk, and the CrossEncoder promoted it to Rank 1.

## Validation Status

```text
All 12 strict retrieval tests: PASS
Hit@3: 100.0%
Hit@8: 100.0%
MRR: 0.875
Neighbor expansion regression check: PASS
```

## Privacy Proof

No PDF or local vector database is committed.
