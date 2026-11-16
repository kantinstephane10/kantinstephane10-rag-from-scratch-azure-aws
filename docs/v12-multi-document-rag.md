# RAG V12 — Multi-Document RAG

## Goal

Extend the validated V11 Pinecone RAG system from one source PDF to a large multi-document cloud corpus.

Local V12 corpus:

- AWS: 8 PDFs
- Azure: 5 PDFs
- GCP: 5 PDFs
- Oracle: 4 PDFs
- IBM: 5 PDFs
- Total: 27 PDFs
- Downloaded corpus size: 107.22 MB

The PDFs remain local and are protected by the repository PDF ignore rule.

---

## Why V12 matters

A single-document RAG system can assume that neighboring chunks belong to the same source.

A multi-document RAG system cannot make that assumption.

V12 therefore introduces explicit document identity:

- provider
- document_id
- document_title
- source
- pdf_page
- document_chunk_number
- chunk_id

This lets the system distinguish documents that discuss similar cloud concepts and prevents retrieval logic from crossing document boundaries accidentally.

---

## V12 build sequence

### Stage 1 — Multi-document ingestion

Status: COMPLETE.

Observed Jupyter validation:

~~~text
Documents: 27
Pages: 3451
Chunks: 11502
Tests passed: 17/17
Accuracy: 100.0%
V12 STAGE 1 MULTI-DOCUMENT INGESTION: PASS
~~~

Observed provider distribution:

~~~text
AWS      | PDFs: 8 | Pages: 2324 | Chunks: 7437
Azure    | PDFs: 5 | Pages: 174  | Chunks: 790
GCP      | PDFs: 5 | Pages: 71   | Chunks: 232
Oracle   | PDFs: 4 | Pages: 116  | Chunks: 492
IBM      | PDFs: 5 | Pages: 766  | Chunks: 2551
~~~

One corpus-quality anomaly was observed: GCP_Enterprise_Foundations_Blueprint.pdf produced only 1 chunk. The file is still represented correctly in the corpus, but its content quality should be reviewed during later retrieval validation.

~~~text
data/cloud_documents/
├── AWS/
├── Azure/
├── GCP/
├── Oracle/
└── IBM/
        ↓
recursive PDF discovery
        ↓
PyMuPDF
        ↓
generic text cleaning
        ↓
provider/document/page metadata
        ↓
RecursiveCharacterTextSplitter
        ↓
document-safe chunk IDs
        ↓
document-safe neighbor lookup
~~~

Notebook-first validation:

- notebooks/v12_multi_document_rag_stage1.ipynb

The notebook contains the complete Stage 1 implementation in one code cell:
PDF discovery, loading, cleaning, metadata, chunking, document-safe neighbors, and all validation tests.

Run the notebook first and confirm all 17/17 tests pass.

Only after the notebook passes do we move the validated logic into the Python files:

- src/rag_v12_multidoc.py
- src/rag_v12_multidoc_test.py

CLI validation after notebook validation:

~~~powershell
uv run python src/rag_v12_multidoc_test.py
~~~

Stage 1 must pass in Jupyter before Pinecone indexing is added.

---

### Stage 2 — Pinecone multi-document index

Status: COMPLETE.

Observed Jupyter validation:

~~~text
Index: multicloud-rag-v12
Namespace: cloud-documents
Documents: 27
Pages: 3451
Chunks: 11502
Embedding shape: (11502, 384)
Tests passed: 22/22
Accuracy: 100.0%
V12 STAGE 2 PINECONE MULTI-DOCUMENT INDEX: PASS
~~~

Validated behaviors:

- Pinecone API connectivity
- 384-dimensional MiniLM embeddings
- cosine vector metric
- 11,502 persisted Pinecone records
- global multi-document semantic retrieval
- provider filters for AWS, Azure, GCP, Oracle, and IBM
- document-specific filtering
- restored provider/document/page metadata
- document-safe neighbor expansion

### Stage 3 — Multi-document retrieval

Planned:

- focused retrieval across the corpus;
- provider-specific questions;
- one-document questions;
- cross-document questions;
- cross-provider questions;
- CrossEncoder reranking across documents.

---

### Stage 4 — Document-aware citations

Planned citation shape:

~~~text
[AWS | AWS Security Reference Architecture | PDF page 42]
~~~

The model will still emit neutral [SOURCE N] markers and Python will construct citations from trusted metadata.

---

### Stage 5 — Streamlit

Planned:

- V12 labeling;
- show provider;
- show source document;
- show PDF page;
- show document-aware retrieved evidence;
- preserve V10 SQLite conversation memory.

---

## Important design decision

Neighbor expansion is keyed by:

~~~text
(document_id, document_chunk_number)
~~~

not by one global chunk number.

This prevents an AWS document's final chunk from pulling the first chunk of a different AWS, Azure, GCP, Oracle, or IBM document.

That is a core correctness requirement for multi-document RAG.

---

## V12 status

V12 is in progress.

Do not mark it complete until:

1. multi-document ingestion passes — COMPLETE (17/17);
2. Pinecone multi-document indexing passes;
3. provider/document filtering passes;
4. retrieval and reranking pass;
5. document-aware citations pass;
6. V9/V10 regressions remain healthy where applicable;
7. Streamlit browser validation passes.


### Stage 3 — Multi-document retrieval + reranking

Status: COMPLETE.

Observed Jupyter validation:

~~~text
Pinecone index: multicloud-rag-v12
Documents: 27
Pages: 3451
Chunks: 11502
Embedding model: sentence-transformers/all-MiniLM-L6-v2
Reranker: cross-encoder/ms-marco-MiniLM-L6-v2
Semantic Top K: 8
Neighbor range: 1
Final Top K: 5
Tests passed: 27/27
Accuracy: 100.0%
V12 STAGE 3 MULTI-DOCUMENT RETRIEVAL + RERANKING: PASS
~~~

Validated behaviors:

- global multi-cloud retrieval
- AWS provider retrieval
- Azure provider retrieval
- GCP provider retrieval
- Oracle provider retrieval
- IBM provider retrieval
- specific-document retrieval
- cross-provider retrieval
- cross-document retrieval
- document-safe neighbor expansion
- duplicate removal
- CrossEncoder reranking
- citation-ready provider/document/page metadata


### Stage 4 — Grounded answers + deterministic citations

Status: COMPLETE.

Observed Jupyter validation:

~~~text
Pinecone index: multicloud-rag-v12
Documents: 27
Pages: 3451
Chunks: 11502
Embedding model: sentence-transformers/all-MiniLM-L6-v2
Reranker: cross-encoder/ms-marco-MiniLM-L6-v2
LLM: qwen3:4b-instruct
Global attempts: 1
AWS attempts: 1
Document attempts: 1
Comparison attempts: 1
Comparison cited providers: ['AWS', 'Azure']
Tests passed: 52/52
Accuracy: 100.0%
V12 STAGE 4 GROUNDED ANSWERS + DETERMINISTIC CITATIONS: PASS
~~~

Validated behaviors:

- grounded Ollama answers
- deterministic provider/document/PDF-page citations
- strict SOURCE-number validation
- invalid SOURCE detection
- same-document citation integrity
- cross-provider evidence balancing
- AWS + Azure comparison answering
- comparison citations include both requested providers
- low-information comparison evidence filtered out
- no raw SOURCE markers remain in final answers
- final citations map only to retrieved evidence
