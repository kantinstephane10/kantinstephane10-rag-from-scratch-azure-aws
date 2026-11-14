# RAG From Scratch — Working MBA RAG Branch

This branch contains the current working end-to-end RAG pipeline built during the learning project.

It uses a private local MBA portfolio PDF as the knowledge source, but **the PDF is intentionally not stored in this repository**.

## Branch

```text
feature/mba-rag-v7-streamlit-ui
```

## Architecture Diagram

![RAG V7 Streamlit Architecture](docs/rag-v7-streamlit-architecture.svg)

## Validation Proof

The observed V6 result is saved in [`docs/results/v6-validation.md`](docs/results/v6-validation.md).

```text
Total tests: 9
Total passed: 9
Answer correctness: 8/8 (100.0%)
Correct source retrieval: 8/8 (100.0%)
Valid citation accuracy: 8/8 (100.0%)
Grounded refusal accuracy: 1/1 (100.0%)
Overall end-to-end accuracy: 9/9 (100.0%)
```

Run the evaluation locally with:

```powershell
uv run python src/rag_v6_evaluation.py
```

## Engineering Challenges, Decisions, and Why This Diagram

### Challenge — perfect retrieval does not prove the RAG system is correct

V5.2 reached **100% Hit@3** on the 12-question retrieval test. That proved the right evidence was available to the generation step, but it did **not** prove that Ollama would:

- use the evidence correctly;
- include the required facts;
- cite a source it actually received;
- refuse a question that the document cannot answer.

A RAG system can have excellent retrieval and still hallucinate during generation.

### Decision — evaluate the complete pipeline separately

Instead of changing the production RAG code again, V6 adds a separate evaluation harness:

```text
src/rag_v6_evaluation.py
```

**Why:** production code and test code have different responsibilities. Keeping evaluation separate makes the working RAG pipeline easier to understand and allows repeatable regression testing.

### What V6 validates

For answerable questions, the evaluation checks:

1. **Answer evidence** — required facts must appear in the generated answer.
2. **Correct source retrieval** — the expected course must be in the retrieved Top 3.
3. **Citation validity** — a cited course/page must exactly match a source that Ollama actually received.

For an unanswerable question, the evaluation checks:

4. **Grounded refusal** — Ollama must refuse instead of answering from outside model knowledge.

### Why we included an out-of-scope question

The test:

```text
What is Microsoft's current stock price?
```

is intentionally unrelated to information available in the MBA portfolio.

Qwen may have outside knowledge, but a grounded RAG assistant should answer only from retrieved document evidence.

The required behavior is:

```text
I don't have enough information in the provided document to answer that.
```

### Result of the decision

The observed V6 evaluation produced:

```text
Total tests: 9
Total passed: 9

Answer correctness:        8/8 (100%)
Correct source retrieval:  8/8 (100%)
Valid citation accuracy:   8/8 (100%)
Grounded refusal accuracy: 1/1 (100%)

Overall: 9/9 (100%)
```

### Why the V6 diagram is designed this way

The top part of the V6 diagram preserves the validated V5.2 production pipeline because V6 did not replace the retriever.

The new lower section represents the **evaluation harness**.

This visual separation communicates an important engineering principle:

```text
Production RAG
      ↓
observable answer + sources
      ↓
independent evaluation
```

The diagram shows four checks after generation—answer correctness, source correctness, citation validation, and refusal behavior—because V6's purpose is to prove that the whole system behaves correctly, not merely that retrieval works.

The 100% result appears at the bottom as measured evidence, not as an assumed claim.

## V7 — Streamlit Interface

V7 adds a browser-based interface without changing the validated RAG backend.

### New file

```text
src/streamlit_app.py
```

### Run the UI

```powershell
git switch feature/mba-rag-v7-streamlit-ui
uv sync
uv run streamlit run src/streamlit_app.py
```

Streamlit should open the application in your browser.

### UI Features

- chat-style question input;
- example questions;
- persistent chat history during the session;
- clear-chat button;
- visible RAG configuration;
- grounded Ollama answers;
- expandable retrieved-source evidence;
- course, chapter, PDF page, retrieval type, and reranker score;
- private PDF remains local.

## Engineering Challenges, Decisions, and Why the V7 Diagram

### Challenge 1 — Streamlit reruns Python after every interaction

A normal Streamlit application reruns the script when the user interacts with a widget. If we initialized the embedding model, reranker, Ollama connection, PDF chunks, and ChromaDB on every rerun, the UI would repeatedly perform expensive startup work.

**Decision:** wrap RAG initialization in:

```python
@st.cache_resource
```

**Why:** the heavy RAG resources are initialized once and reused during normal UI reruns.

### Challenge 2 — Avoid mixing UI code with the validated RAG logic

V6 already proved that retrieval, answer generation, citations, and grounded refusal work correctly.

**Decision:** keep `src/rag_pipeline.py` as the RAG backend and create a separate:

```text
src/streamlit_app.py
```

The UI imports and calls the existing `ask_rag()` function.

**Why:** this preserves separation of concerns:

```text
RAG backend = retrieval + generation

Streamlit UI = user interaction + display
```

A UI bug should not require rewriting the retrieval pipeline.

### Challenge 3 — A friendly UI must not hide the evidence

A simple chatbot can display only the generated answer, but that would hide the retrieval behavior we spent several versions validating.

**Decision:** every assistant response includes an expandable:

```text
View retrieved sources
```

panel.

The panel displays:

- course;
- chapter;
- PDF page;
- semantic or neighbor retrieval source;
- CrossEncoder score;
- retrieved source text.

**Why:** the interface remains easy to use while preserving explainability and making retrieval failures visible.

### Challenge 4 — Preserve local privacy

The source MBA portfolio is private and should not be uploaded to GitHub or a hosted service just to build the interface.

**Decision:** Streamlit runs locally and continues using the local PDF, local ChromaDB, local embedding/reranking models, and local Ollama model.

### Why the V7 diagram is designed this way

The diagram adds **Streamlit as a layer above the validated RAG pipeline**, rather than inserting Streamlit inside retrieval.

That distinction is intentional:

```text
User
 ↓
Streamlit UI
 ↓
Validated RAG backend
 ↓
Answer + evidence
 ↓
Streamlit display
```

The `st.cache_resource` box appears between the UI and backend because resource caching is the main new runtime concern introduced by Streamlit.

The lower part of the diagram still shows the complete V5.2/V6 retrieval pipeline. This makes it clear that V7 changes **how the user interacts with RAG**, not how the evidence is retrieved or generated.

## V7 Validation Status

The backend is already validated by V6:

```text
9/9 end-to-end tests passed
Overall accuracy: 100%
```

The Streamlit UI itself still requires a local runtime validation because it depends on your private PDF and local Ollama installation.

The exact test plan is saved here:

[`docs/results/v7-validation-plan.md`](docs/results/v7-validation-plan.md)

After the local UI run succeeds, this branch will receive a final V7 validation proof with the observed results.

## Current Architecture

```text
Private Local PDF
       ↓
PyMuPDFLoader
       ↓
Text Cleaning
       ↓
Course / Chapter / Page Metadata
       ↓
RecursiveCharacterTextSplitter
800 characters / 120 overlap
       ↓
Front-Matter + Boilerplate Filtering
       ↓
Hugging Face MiniLM Embeddings
all-MiniLM-L6-v2
       ↓
ChromaDB
       ↓
User Question
       ↓
Question Embedding
       ↓
Top-8 Semantic Retrieval
       ↓
Neighbor Expansion (±1 chunk)
       ↓
CrossEncoder Reranking
ms-marco-MiniLM-L6-v2
       ↓
Best 3 Chunks
       ↓
Context Construction
       ↓
Ollama
qwen3:4b-instruct
       ↓
Grounded Answer
       ↓
Course + PDF Page Citation
```

## Privacy

The source PDF is private and must remain local.

The repository ignores:

```text
*.pdf
data/text_files/pdf/
chroma_db_*/
```

Do not commit personal PDFs or local ChromaDB files.

## Local PDF Location

Create this folder locally:

```text
data/
└── text_files/
    └── pdf/
```

Place the private PDF there using this filename:

```text
WGU_MBA_IT_Management_Bound_Edition_Biography_At_Beginning.pdf
```

The file will remain ignored by Git.

## Models

### Embeddings

```text
sentence-transformers/all-MiniLM-L6-v2
```

384-dimensional local embeddings.

### Reranker

```text
cross-encoder/ms-marco-MiniLM-L6-v2
```

The system retrieves eight candidates from ChromaDB and reranks them to the best three.

### Answer Generation

```text
qwen3:4b-instruct
```

The generation model runs locally through Ollama.

## Setup

Install Python 3.12+ and `uv`.

```powershell
git clone https://github.com/kantinstephane10/kantinstephane10-rag-from-scratch-azure-aws.git

cd kantinstephane10-rag-from-scratch-azure-aws

git switch feature/mba-rag-v7-streamlit-ui

uv sync
```

Install Ollama separately, then pull the model:

```powershell
ollama pull qwen3:4b-instruct
```

Confirm it exists:

```powershell
ollama list
```

## Run

After placing the private PDF in the local ignored folder:

```powershell
uv run python src/rag_pipeline.py
```

The CLI will ask:

```text
Ask a question about the MBA document:
```

Example:

```text
What are my five CliftonStrengths?
```

A working answer should be grounded in the retrieved document and include a source citation such as:

```text
My five CliftonStrengths are Learner, Strategic, Belief,
Context, and Deliberative. [C200, PDF page 10]
```

## Current RAG Features

- PDF loading with PyMuPDF
- text cleaning
- recursive character chunking
- 800-character chunks
- 120-character overlap
- course/chapter/page metadata
- front-matter exclusion
- low-value boilerplate filtering
- local Hugging Face embeddings
- persistent ChromaDB vector storage
- semantic Top-K retrieval
- neighboring-chunk expansion
- CrossEncoder reranking
- local Ollama generation
- grounded prompting
- course and PDF-page citations

## Retrieval Evaluation

The V5.2 retriever was evaluated with 12 questions spanning all six MBA courses.

Results:

```text
Hit@1:  9/12  (75.0%)
Hit@3: 12/12 (100.0%)
Hit@5: 12/12 (100.0%)
Hit@8: 12/12 (100.0%)
MRR:   0.875
```

A retrieval failure involving the Agentic RAG architecture was diagnosed and fixed with neighboring-chunk expansion. The semantic search found an adjacent introductory chunk, and the answer-bearing next chunk was added before reranking.

## Next Step

Run and validate the V7 Streamlit interface locally. After the UI passes the documented tests, save the observed result as the V7 proof before moving to conversational memory or agent capabilities.
