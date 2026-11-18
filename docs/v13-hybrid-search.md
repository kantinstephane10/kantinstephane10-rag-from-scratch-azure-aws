# RAG V13 — Hybrid Search (Dense + Sparse)

V13 upgrades the V12 multi-document, multi-cloud RAG system from dense-only retrieval to a validated hybrid retrieval pipeline.

## Milestone status

**RAG V13 — HYBRID SEARCH: COMPLETE**

Validated stack:

- Dense semantic retrieval with Pinecone
- Sparse lexical retrieval with BM25
- Reciprocal Rank Fusion (RRF)
- Same-document neighbor expansion
- Low-information filtering and deduplication
- CrossEncoder reranking
- Grounded Ollama generation
- Deterministic provider/document/PDF-page citations
- SQLite persistent conversational memory
- Live Streamlit token streaming
- Global, single-provider, and cross-provider retrieval
- Comparison-aware global routing
- Hardened citation validation

## Corpus

| Provider | PDFs | Pages | Chunks |
| --- | ---: | ---: | ---: |
| AWS | 8 | 2,324 | 7,437 |
| Azure | 5 | 174 | 790 |
| GCP | 5 | 71 | 232 |
| Oracle | 4 | 116 | 492 |
| IBM | 5 | 766 | 2,551 |
| **Total** | **27** | **3,451** | **11,502** |

The source PDFs remain local and are intentionally excluded from Git.

## V13 hybrid retrieval

The retrieval path is:

```text
User query
   ↓
Conversation-aware rewrite
   ↓
┌───────────────────────────────┐
│ Dense retrieval              │
│ MiniLM → Pinecone → Top 20   │
└───────────────────────────────┘
             +
┌───────────────────────────────┐
│ Sparse retrieval             │
│ BM25 token index → Top 20    │
└───────────────────────────────┘
             ↓
Reciprocal Rank Fusion (RRF)
             ↓
Same-document neighbors ±1
             ↓
Deduplicate / filter weak evidence
             ↓
CrossEncoder reranking
             ↓
Best 5 evidence chunks
             ↓
Ollama qwen3:4b-instruct
             ↓
Strict [SOURCE N] validation
             ↓
Trusted provider/document/PDF-page citations
             ↓
SQLite memory + Streamlit UI
```

## Retrieval configuration

```text
DENSE_TOP_K = 20
BM25_TOP_K = 20
RRF_K = 60
FUSION_TOP_K = 25
NEIGHBOR_RANGE = 1
FINAL_TOP_K = 5
MIN_EVIDENCE_CHARS = 120
```

Comparison retrieval uses a larger provider-specific candidate pool:

```text
COMPARISON_DENSE_TOP_K = 30
COMPARISON_BM25_TOP_K = 30
COMPARISON_FUSION_TOP_K = 40
COMPARISON_FINAL_TOP_K = 5
```

## Models and infrastructure

- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Embedding dimension: 384
- Sparse retriever: `rank_bm25.BM25Okapi`
- Reranker: `cross-encoder/ms-marco-MiniLM-L6-v2`
- Vector database: Pinecone
- Pinecone index: `multicloud-rag-v12`
- Namespace: `cloud-documents`
- Vector field: `embedding`
- Generator: Ollama `qwen3:4b-instruct`
- Persistent memory: SQLite
- UI: Streamlit

The existing V12 Pinecone index is intentionally reused because V13 changes retrieval orchestration rather than the persisted dense-vector representation.

## Why hybrid search

Dense semantic retrieval is strong when the user's wording differs from the source document.

BM25 is strong when exact technical identifiers matter, for example:

- `SEC11-BP06`
- `HTTP-502`
- service names
- configuration identifiers
- product terminology

RRF combines the two ranked lists by rank position instead of directly mixing incompatible cosine and BM25 score scales.

## Starter benchmark

A 10-query development benchmark produced:

| Retriever | Hit@1 | Hit@3 | Hit@5 | MRR |
| --- | ---: | ---: | ---: | ---: |
| Dense | 0.90 | 1.00 | 1.00 | 0.9333 |
| BM25 | 0.80 | 1.00 | 1.00 | 0.8833 |
| Hybrid | 0.90 | 1.00 | 1.00 | **0.9500** |

Interpretation: on this starter benchmark, hybrid retrieval preserved dense Hit@1 while improving MRR. This is a development result, not a claim of universal superiority.

## Production validation

V13 was intentionally tested outside the notebook to catch hidden Jupyter dependencies.

Fresh-process backend validation:

```text
Backend cache independent:             PASS
BM25 independent:                      PASS
Pinecone dense search independent:     PASS
Hybrid retrieval independent:          PASS
Cross-provider retrieval independent:  PASS

Tests passed: 5/5
V13 FRESH PROCESS BACKEND: PASS
```

Fresh-process live streaming validation:

```text
Callback received tokens:       PASS
Raw answer created:             PASS
Citation validation:            PASS
Trusted final citations:        PASS
SQLite persisted 2 messages:    PASS

Tests passed: 5/5
V13 PRODUCTION LIVE STREAMING: PASS
```

The Streamlit browser was then validated with the complete production path:

```text
Streamlit
  → rag_v13 backend
  → SQLite memory
  → Dense Pinecone + BM25
  → RRF
  → neighbor expansion
  → CrossEncoder
  → Ollama live token streaming
  → citation validation
  → trusted final answer
```

## Cross-provider retrieval

Cross-provider questions retrieve evidence separately for each requested provider before generation.

This prevents a provider with a larger corpus from dominating the comparison simply because it has more chunks.

Global mode also includes deterministic comparison-aware routing. Examples:

```text
Compare AWS and Azure
    → cross-provider: AWS + Azure

What is AWS?
Compare it to Azure
    → conversation context resolves AWS + Azure

Compare different cloud services side by side
    → cross-provider: AWS + Azure + GCP + Oracle + IBM
```

## Grounding and citation hardening

The model emits neutral evidence references such as:

```text
[SOURCE 1]
```

Python validates the source number and converts it to trusted metadata:

```text
[AWS | AWS Reliability Pillar | PDF page 9]
```

V13 also hardens malformed model output:

- combined markers are normalized;
- negative source numbers are rejected;
- out-of-range source numbers are rejected;
- malformed markers cannot be displayed as trusted citations;
- adjacent duplicate final citations are collapsed.

Missing evidence is not treated as proof that a provider lacks a capability.

## Persistent memory

SQLite stores conversational turns and evidence. A recent-memory window is used to rewrite follow-up questions into standalone retrieval questions.

Example:

```text
Turn 1:
What does AWS recommend for reliable workloads?

Turn 2:
How does Azure compare with that?

Standalone retrieval question:
How does Azure compare with AWS's recommendations for reliable workloads?
```

## Corpus coverage finding

The V13 coverage audit showed that retrieval works across all five providers, but the current 27-PDF corpus is intentionally asymmetric.

Strong areas include AWS security/reliability, Azure reliability, Oracle IAM/networking, and IBM object storage. GCP and several Oracle/IBM domains need broader service-specific documentation before broad five-provider service comparisons can be expected to be symmetric.

This is a corpus-quality backlog item and does not block the V13 retrieval milestone.

## Architecture diagram

See:

- [V13 hybrid-search architecture with cloud/service icons](rag-v13-hybrid-architecture-official-icons.png)

## Next milestone

**V14 — Metadata-Aware Retrieval**

V14 will add structured topic/domain metadata and use it as part of retrieval strategy rather than only as an optional provider/document filter.
