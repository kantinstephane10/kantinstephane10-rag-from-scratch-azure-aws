# RAG V9 Validation — Conversational Memory

## Scope

V9 extends the validated V8 retrieval architecture with:

- short-term conversation memory;
- follow-up question rewriting;
- a strict broad-query course relevance gate;
- citation deduplication;
- cleanup of contradictory “supported answer + refusal” responses.

## Local notebook validation

The conversational backend was executed locally in VS Code / Jupyter against the 143-page MBA portfolio PDF.

Observed initialization:

```text
Pages loaded: 143
All chunks: 366
Course chunks: 354
Final indexed chunks: 351
Embedding shape: (351, 384)
Vectors stored: 351
```

Observed automated result:

```text
Five CliftonStrengths : PASS
Follow-up 1 rewritten with memory : PASS
Follow-up 1 answer = Learner : PASS
Agentic RAG technologies : PASS
Follow-up 2 rewritten with memory : PASS
Follow-up 2 answer = Azure AI Search : PASS
Broad retrieval still works : PASS
Project Management course found : PASS
Grounded refusal : PASS

Tests passed: 9/9
Accuracy: 100.0%
V9 CONVERSATIONAL MEMORY TEST: PASS
```

## Memory proof

Initial question:

```text
What are my five CliftonStrengths?
```

Follow-up:

```text
Which one was ranked highest?
```

Observed internal rewrite:

```text
Which of my five CliftonStrengths was ranked highest?
```

Observed answer:

```text
Learner was ranked highest. [C200, PDF page 10]
```

Second conversation:

```text
What technologies were proposed for the Agentic RAG architecture?
```

Follow-up:

```text
Which one provides the search and retrieval layer?
```

Observed internal rewrite:

```text
Which of the technologies—Microsoft Foundry Agent Service or Azure AI Search—
provides the search and retrieval layer in the Agentic RAG architecture?
```

Observed answer identified Azure AI Search and cited MMT2-BEN1 page 115.

## Streamlit validation

The browser UI was then tested locally.

### Conversation memory

The Streamlit memory panel displayed both the original follow-up and the standalone rewritten retrieval question. The follow-up answer correctly returned Learner.

### Broad-query false-positive fix

A broad question initially allowed C200 as an indirect project-management match.

A strict course relevance gate was added.

Final observed browser result:

```text
Identify all courses in my MBA portfolio that discuss project management.

The course in your MBA portfolio that discusses project management is
MGT2-KKM1: Project Management [MGT2-KKM1, PDF page 72].
```

C200 was no longer present.

### Grounded refusal

Question:

```text
What is Microsoft's current stock price?
```

Observed answer:

```text
I don't have enough information in the provided document to answer that.
```

The valid refusal remained intact after the conflicting-refusal cleanup was added.

## Reproducible regression harness

Run:

```powershell
uv run python src/rag_v9_evaluation.py
```

The harness additionally checks the C200 false-positive regression and the course relevance gate.
