# V1.3 release notes

- Project-scoped RAG: PDF/TXT/MD/CSV extraction, chunking, OpenAI embeddings, local SQLite cosine retrieval, cited answers.
- Background index jobs with durable state, integrity receipts, and stale-document blocking.
- Delete Project UI and backend with exact-name confirmation and project isolation.
- Meeting handling preserves main replies when an invalid directed follow-up question is generated.
- Backend and HTTP regression tests for PDF, RAG, deletion, and meeting safety.

**Not yet implemented:** OCR, arbitrary file editing/code execution, real LangGraph engineering workers, document-level multiuser access policies, release pipelines, production deployment, or benchmark/penetration-test evidence.
