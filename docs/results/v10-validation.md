# RAG V10 Validation — Persistent Conversational Memory

## Scope

V10 extends the validated V9 conversational RAG system with persistent SQLite-backed conversation storage and multi-conversation management.

V10 adds:

- SQLite persistent memory;
- conversation persistence across application and Jupyter restarts;
- multiple independent conversations;
- automatic conversation titles;
- conversation switching;
- rename;
- clear;
- delete;
- local persistence with SQLite as the source of truth.

The validated V9 RAG backend remains unchanged for retrieval and generation.

---

## Core RAG initialization

Observed locally:

```text
Pages loaded: 143
All chunks: 366
Course chunks: 354
Final indexed chunks: 351
Embedding shape: (351, 384)
Vectors stored: 351
```

Models:

- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Reranker: `cross-encoder/ms-marco-MiniLM-L6-v2`
- Generation: Ollama `qwen3:4b-instruct`

---

## V10 notebook validation

The Stage 2 V10 notebook tested SQLite persistence, multi-conversation isolation, conversation management, retrieval behavior, citation correctness, and grounded refusal.

Observed result:

```text
RAG V10 STAGE 2 FINAL RESULT
======================================================================

Tests passed: 31/31
Accuracy: 100.0%
V10 CONVERSATION MANAGEMENT: PASS
```

Validated behaviors included:

- Conversation A creation;
- empty conversation initialization;
- five CliftonStrengths retrieval;
- two-message persistence after the first turn;
- automatic title creation;
- SQLite reload;
- follow-up rewriting;
- Learner as the highest-ranked CliftonStrength;
- four-message persistence after the follow-up;
- Conversation B creation;
- isolation between conversations;
- Agentic RAG technology retrieval;
- grounded MMT2-BEN1 citation;
- second follow-up rewrite;
- Azure AI Search retrieval answer;
- conversation listing;
- rename;
- broad routing;
- project-management course detection;
- C200 false-positive removal;
- course relevance gate;
- grounded refusal;
- full-history persistence;
- recent-memory window;
- clear;
- delete;
- isolation after deletion.

---

## True restart persistence proof

After restarting the Jupyter kernel, the application recovered the previously stored conversation directly from SQLite.

Observed result:

```text
Saved conversations found: 3

PERSISTED CONVERSATION FOUND

Title:
V10-STAGE2-What are my five CliftonStrengths?

Messages recovered: 4

V10 TRUE RESTART PERSISTENCE: PASS
```

Recovered conversation:

```text
USER
What are my five CliftonStrengths?

ASSISTANT
My five CliftonStrengths are Learner, Strategic, Belief, Context,
and Deliberative [C200, PDF page 10].

USER
Which one was ranked highest?

ASSISTANT
Learner was ranked highest. [C200, PDF page 10]
```

This proves the conversation was restored from disk rather than from Python process memory.

---

## Repo-native SQLite memory regression

The V10 branch includes:

```text
src/rag_v10_memory_test.py
```

Run:

```powershell
uv run python src/rag_v10_memory_test.py
```

Observed result:

```text
RAG V10 REPO MEMORY TEST
======================================================================

Tests passed: 18/18
Accuracy: 100.0%
V10 REPO MEMORY TEST: PASS
```

The repo-native test validates:

- database creation;
- conversation creation;
- empty initial conversation;
- message persistence;
- user/assistant restoration;
- query metadata restoration;
- standalone-question restoration;
- retrieved-source restoration;
- automatic title creation;
- recent-memory loading;
- second conversation creation;
- conversation isolation;
- conversation listing;
- rename;
- clear;
- delete.

---

## V9 regression validation on V10 branch

V10 was also tested against the existing V9 end-to-end regression suite to verify that persistent memory did not break the RAG pipeline.

Run:

```powershell
uv run python src/rag_v9_evaluation.py
```

Observed result:

```text
V9 REGRESSION EVALUATION

Five CliftonStrengths: PASS
Follow-up 1 rewritten with memory: PASS
Follow-up 1 answer = Learner: PASS
Agentic RAG technologies: PASS
Agentic RAG citation grounded: PASS
Follow-up 2 rewritten with memory: PASS
Follow-up 2 answer = Azure AI Search: PASS
Broad router: PASS
Project Management course found: PASS
C200 false positive removed: PASS
Course relevance gate: PASS
Grounded refusal: PASS

Tests passed: 12/12
Accuracy: 100.0%
V9 REGRESSION TEST: PASS
```

---

## Streamlit browser validation

The V10 Streamlit UI was tested directly from the repository.

Validated browser behaviors:

- create a new conversation;
- ask a focused question;
- ask a memory-dependent follow-up;
- create a second conversation;
- switch between conversations;
- recover previous messages;
- automatic conversation title;
- rename conversation;
- clear conversation;
- delete conversation;
- broad-query retrieval;
- grounded refusal;
- stop Streamlit;
- restart Streamlit;
- recover saved conversations from SQLite.

### Rename issue and fix

During browser testing, SQLite rename itself was already passing the repo memory test, but the Streamlit rename widget did not update the displayed title.

The Streamlit rename flow was changed to use a form so that the text value and rename action are submitted together.

After the fix, rename was successfully validated in the browser.

Fix commit:

```text
75e86f837c5a263c8c51b19517edf7c15d4d7238
```

---

## Persistence architecture

V9:

```text
Streamlit session_state
        ↓
Application stops
        ↓
Conversation disappears
```

V10:

```text
Streamlit
    ↓
SQLite conversation database
    ↓
Application stops
    ↓
Database remains on disk
    ↓
Application restarts
    ↓
Previous conversations are loaded
```

SQLite stores the complete conversation history.

The LLM receives only the most recent:

```text
12 messages
```

This keeps persistent storage and active LLM context separate.

---

## Local SQLite database

Default path:

```text
data/memory/rag_v10_memory.db
```

The database is ignored by Git so local conversation history is not committed.

---

## Source document

V10 uses:

```text
data/text_files/pdf/WGU_MBA_IT_Management_Bound_Edition_Biography_At_Beginning.pdf
```

The source PDF is private, kept local, and intentionally excluded from GitHub by `.gitignore`.

---

## Final V10 status

```text
Notebook conversation-management validation: 31/31 PASS
True restart persistence: PASS
Repo SQLite regression: 18/18 PASS
V9 RAG regression: 12/12 PASS
Streamlit browser validation: PASS
Rename browser validation after fix: PASS

RAG V10 PERSISTENT MEMORY MILESTONE: COMPLETE
```
