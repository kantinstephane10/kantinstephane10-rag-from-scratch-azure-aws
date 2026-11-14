# V7 Streamlit Validation Plan

## Branch

`feature/mba-rag-v7-streamlit-ui`

## Status

**Pending local UI execution.**

The V7 code has been added, but a runtime proof should only be recorded after the Streamlit application is executed on the local machine with:

- the private MBA PDF;
- local Ollama;
- `qwen3:4b-instruct`.

The backend inherited from V6 has already passed 9/9 end-to-end tests.

## Run

```powershell
git switch feature/mba-rag-v7-streamlit-ui
uv sync
uv run streamlit run src/streamlit_app.py
```

## UI Validation Checklist

### Test 1 — Application startup

Expected:

```text
RAG ready — 351 indexed chunks.
```

### Test 2 — Known C200 question

Question:

```text
What are my five CliftonStrengths?
```

Expected facts:

```text
Learner
Strategic
Belief
Context
Deliberative
```

Expected citation:

```text
[C200, PDF page 10]
```

### Test 3 — Neighbor-expansion regression question

Question:

```text
What technologies were proposed for the Agentic RAG architecture?
```

Expected facts:

```text
Microsoft Foundry Agent Service
Azure AI Search
```

The source panel should show the answer-bearing neighbor chunk.

### Test 4 — Grounded refusal

Question:

```text
What is Microsoft's current stock price?
```

Expected:

```text
I don't have enough information in the provided document to answer that.
```

### Test 5 — Source panel

For a normal answer, open:

```text
View retrieved sources
```

Confirm the UI displays:

- course;
- chapter;
- PDF page;
- retrieval source (`semantic` or `neighbor`);
- reranker score;
- retrieved source text.

### Test 6 — Chat history

Ask at least two questions and confirm previous user/assistant messages remain visible.

### Test 7 — Clear chat

Click:

```text
Clear chat
```

Expected: visible chat history is cleared without rebuilding the RAG backend.

## Proof to Record After Local Run

After the UI passes these checks, replace this validation plan with a V7 validation proof containing:

- observed startup result;
- representative questions and answers;
- source-panel evidence;
- grounded-refusal result;
- UI screenshot if desired;
- final PASS/FAIL status.
