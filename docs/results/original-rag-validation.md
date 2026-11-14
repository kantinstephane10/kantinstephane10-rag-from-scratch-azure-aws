# Validation Proof — Original Azure/AWS RAG

## Branch

`main`

## Purpose

This file records the observed working result for the original Azure/AWS RAG pipeline.

## Pipeline Under Test

```text
Azure/AWS training text
→ text loading and cleaning
→ recursive chunking
→ all-MiniLM-L6-v2 embeddings
→ ChromaDB
→ semantic retrieval
→ local Hugging Face generation
→ grounded answer
```

## Validation Question

```text
How does Azure provide private connectivity to storage?
```

## Observed Result

The retriever returned the relevant Azure networking/storage material and the local generation model produced the correct grounded answer identifying **Azure Private Endpoint** as the direct private-connectivity method described by the training source.

## Validation Status

```text
Retrieval: PASS
Generation: PASS
Grounding: PASS
Overall: PASS
```

## Reproduce

```powershell
uv sync
uv run python src/rag_pipeline.py
```

This proof records a local development validation. The generated ChromaDB data is not committed.
