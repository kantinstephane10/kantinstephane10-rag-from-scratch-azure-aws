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

Planned:

- new V12 Pinecone index;
- new V12 namespace;
- index all 27 documents;
- store provider/document metadata in Pinecone;
- search across all documents;
- filter by provider;
- filter by document;
- preserve document-safe neighbor expansion.

---

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
