# RAG V11 Design Plan — Multi-Document RAG

## Goal

V11 evolves the system from one source PDF to multiple source documents while preserving all validated V10 behavior.

V10 already provides:

- persistent SQLite conversation memory;
- multiple chat conversations;
- focused/broad query routing;
- neighbor expansion;
- CrossEncoder reranking;
- broad-query relevance filtering;
- deterministic citations;
- grounded refusal;
- Streamlit conversation management.

V11 adds **document awareness**.

---

## Target architecture

```text
Document 1 ─┐
Document 2 ─┤
Document 3 ─┤
Document N ─┘
     ↓
Document Registry
     ↓
Load + Clean
     ↓
Document Metadata
     ↓
Recursive Chunking
     ↓
Noise Filtering
     ↓
MiniLM Embeddings
     ↓
ChromaDB
     ↓
User Question
     ↓
Conversation Memory
     ↓
Follow-Up Rewriter
     ↓
Document Scope
    /        \
Selected     All
Document     Documents
    \        /
     ↓
Query Router
     ↓
Focused / Broad Retrieval
     ↓
CrossEncoder
     ↓
Grounded Answer
     ↓
Document-Aware Citations
     ↓
Save Turn to SQLite
```

---

## New document metadata

Every chunk should carry document-level metadata in addition to the existing course/page metadata.

Planned metadata:

```python
{
    "document_id": "...",
    "document_name": "...",
    "source": "...",
    "pdf_page": ...,
    "course": "...",
    "chapter": "...",
    "rag_chunk_number": ...
}
```

The exact metadata fields can vary by file type later, but V11 should establish a consistent document identity first.

---

## Document registry

V11 should use a simple registry rather than hard-coding one PDF variable.

Target shape:

```python
DOCUMENTS = [
    {
        "document_id": "mba_portfolio",
        "document_name": "MBA Portfolio",
        "path": Path("data/text_files/pdf/...pdf"),
    },

    {
        "document_id": "document_2",
        "document_name": "Document 2",
        "path": Path("data/text_files/pdf/...pdf"),
    },
]
```

This remains simple Python and avoids unnecessary classes or framework abstractions.

---

## Retrieval modes

V11 should support two document scopes.

### 1. Selected-document search

```text
User selects one document
        ↓
ChromaDB metadata filter
        ↓
Search only that document
```

Example:

```python
where={
    "document_id": "mba_portfolio"
}
```

### 2. All-document search

```text
All documents
      ↓
Search candidates from each document
      ↓
Rerank evidence
      ↓
Keep strongest supported evidence
      ↓
Generate cross-document answer
```

The all-document path must not allow one large document to dominate retrieval simply because it has more chunks.

---

## Document-aware citations

V10 citations currently use course and PDF page metadata.

V11 should extend citations so the source document is always identifiable.

Target example:

```text
[MBA Portfolio | MMT2-BEN1 | PDF page 115]
```

For a document without course metadata:

```text
[Cloud Runbook | PDF page 12]
```

Citation construction must remain deterministic in Python.

The LLM should continue to emit neutral markers such as:

```text
[SOURCE 1]
```

Python should map those markers to trusted metadata.

---

## Streamlit changes

The V11 sidebar should add document scope without breaking V10 conversation controls.

Target UI:

```text
Conversations
────────────────────────
+ New Conversation
Previous Conversations
Rename
Clear
Delete

Document Scope
────────────────────────
○ All Documents
○ MBA Portfolio
○ Document 2
○ Document 3

RAG Configuration
────────────────────────
...
```

The selected document scope should affect retrieval, not persistent chat identity.

---

## V10 behavior that must remain unchanged

V11 must preserve:

- SQLite persistence;
- restart persistence;
- multiple conversations;
- auto-title;
- rename;
- clear;
- delete;
- focused retrieval;
- broad retrieval;
- course relevance gate;
- deterministic citations;
- grounded refusal;
- existing V9 regression behavior.

---

## Initial V11 test plan

### Test group A — document loading

Validate:

- all configured documents exist;
- every document receives a unique `document_id`;
- every chunk receives `document_id`;
- every chunk receives `document_name`;
- chunks remain associated with the correct source document.

### Test group B — selected-document retrieval

Validate:

- selecting Document A returns only Document A;
- selecting Document B returns only Document B;
- citations identify the selected document correctly.

### Test group C — all-document retrieval

Validate:

- all configured documents are searched;
- relevant evidence can be returned from more than one document;
- unrelated documents do not appear in the answer;
- source-document citations remain correct.

### Test group D — V10 regression

Re-run:

```powershell
uv run python src/rag_v10_memory_test.py
uv run python src/rag_v9_evaluation.py
```

Both existing suites must continue to pass.

### Test group E — browser validation

Validate:

- document selector works;
- document scope can change between questions;
- conversation persistence remains intact;
- switching conversations preserves their messages;
- document-aware citations render correctly;
- restart persistence still works.

---

## Implementation order

```text
1. Add document registry
2. Add document_id/document_name metadata
3. Load multiple documents
4. Index all documents
5. Add selected-document retrieval filter
6. Add all-document retrieval
7. Add document-aware deterministic citations
8. Add Streamlit document selector
9. Add V11 regression tests
10. Run V10 regression tests
11. Browser validation
12. Update README and architecture
```

---

## Guardrail

Do not call V11 complete until actual multi-document tests are observed.

No V11 accuracy or pass rate should be documented before those tests run successfully.
