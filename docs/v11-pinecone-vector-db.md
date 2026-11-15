# RAG V11 — Pinecone Vector Database

## Status

Implementation is ready for local validation.

Do not mark V11 complete until the Pinecone-specific regression and the existing RAG regression both pass.

---

## Goal

Replace the local ChromaDB vector store with Pinecone while preserving the validated RAG behavior from V10.

V11 changes only the vector database layer.

The following remain unchanged:

- PyMuPDF loading;
- text cleaning;
- course/page metadata;
- RecursiveCharacterTextSplitter;
- MiniLM embeddings;
- neighbor expansion;
- CrossEncoder reranking;
- focused vs. broad query routing;
- broad-query course relevance gate;
- Ollama generation;
- deterministic citations;
- grounded refusal;
- SQLite persistent conversation memory;
- Streamlit multi-conversation management.

---

## V10

```text
MiniLM
  ↓
ChromaDB
  ↓
Neighbor Expansion
  ↓
CrossEncoder
```

## V11

```text
MiniLM
  ↓
Pinecone
  ↓
Neighbor Expansion
  ↓
CrossEncoder
```

---

## Pinecone design

V11 keeps embeddings local.

```text
Document Chunk
      ↓
all-MiniLM-L6-v2
      ↓
384-dimensional normalized embedding
      ↓
Pinecone dense_vector field
      ↓
Cosine semantic search
```

Pinecone does not generate the embeddings in this milestone.

The existing Hugging Face MiniLM model remains the embedding model.

---

## Pinecone index

Default index:

```text
mba-rag-v11
```

Default namespace:

```text
mba-portfolio
```

Dense vector field:

```text
embedding
```

Configuration:

```text
dimension = 384
metric = cosine
```

---

## Stored Pinecone document

Each RAG chunk is stored as a Pinecone document containing:

```python
{
    "_id": "mba_chunk_0000",
    "embedding": [...384 floats...],
    "text": "...",
    "source": "...",
    "pdf_page": 10,
    "course": "C200",
    "chapter": "...",
    "rag_chunk_number": 0,
}
```

Only the `embedding` field is declared as the searchable dense-vector schema field.

The remaining values are stored as metadata fields and are returned during retrieval.

---

## Focused retrieval

```text
Question
  ↓
MiniLM query embedding
  ↓
Pinecone Top 8
  ↓
Neighbor Expansion ±1
  ↓
CrossEncoder
  ↓
Best 3
```

---

## Broad retrieval

```text
Question
  ↓
MiniLM query embedding
  ↓
For each MBA course:
    Pinecone metadata filter
    ↓
    Top 4
    ↓
    Neighbor Expansion ±1
    ↓
    CrossEncoder
    ↓
    Best 2
  ↓
Course Relevance Gate
  ↓
Grounded Answer
```

Pinecone metadata filtering replaces the previous ChromaDB `where={"course": ...}` filter.

---

## Environment variables

Required:

```text
PINECONE_API_KEY
```

Optional:

```text
PINECONE_INDEX_NAME
PINECONE_NAMESPACE
```

Defaults:

```text
PINECONE_INDEX_NAME=mba-rag-v11
PINECONE_NAMESPACE=mba-portfolio
```

Never commit the real Pinecone API key.

---

## Windows PowerShell setup

Example:

```powershell
$env:PINECONE_API_KEY="YOUR_REAL_PINECONE_API_KEY"
$env:PINECONE_INDEX_NAME="mba-rag-v11"
$env:PINECONE_NAMESPACE="mba-portfolio"
```

Verify:

```powershell
$env:PINECONE_API_KEY
```

Do not paste the real key into GitHub, README files, screenshots, or chat logs.

---

## Dependency

V11 uses:

```text
pinecone>=10.3,<11
```

The repository no longer requires ChromaDB on this branch.

---

## Validation order

### 1. Pinecone vector database test

```powershell
uv run python src/rag_v11_pinecone_test.py
```

This validates:

- API key availability;
- index creation/reuse;
- dense vector field;
- 384 dimensions;
- cosine metric;
- document persistence;
- semantic search;
- metadata restoration;
- similarity scores;
- metadata filtering;
- focused retrieval;
- C200 page 10 retrieval;
- CrossEncoder compatibility.

No pass count is documented until this test is actually run.

### 2. Existing RAG regression

```powershell
uv run python src/rag_v9_evaluation.py
```

The existing 12/12 regression must continue to pass with Pinecone.

### 3. V10 memory regression

```powershell
uv run python src/rag_v10_memory_test.py
```

SQLite persistence must remain unaffected.

### 4. Streamlit

```powershell
uv run streamlit run src/streamlit_app.py
```

Browser validation should confirm:

- Pinecone-backed focused retrieval;
- Pinecone-backed broad retrieval;
- persistent SQLite memory;
- new/switch/rename/clear/delete conversations;
- restart persistence;
- deterministic citations;
- grounded refusal.

---

## Source document

The source PDF remains private and local.

Expected local path:

```text
data/text_files/pdf/WGU_MBA_IT_Management_Bound_Edition_Biography_At_Beginning.pdf
```

The PDF is intentionally excluded from GitHub.
