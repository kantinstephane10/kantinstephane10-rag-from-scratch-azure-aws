# Example project brief — Enterprise RAG assistant

**AI product type:** RAG assistant

**Project name:** Multi-Cloud Infrastructure RAG Assistant

**Project ideas and requirements:**
Build a Python-based RAG assistant that answers operations questions from our approved AWS, Azure and GCP documentation. Ingestion should extract readable text from PDFs and Markdown, preserve document/page metadata, chunk documents and create searchable embeddings. Retrieval should support dense and keyword queries, hybrid fusion and reranking, with source-grounded answers and citations. Support project-specific permissions, conversation state and audit logs. Use LangChain and LangGraph where they help, plus LangSmith for evaluation/tracing when explicitly configured. Define an offline evaluation corpus measuring retrieval accuracy, groundedness and abstention. Document error handling, test results, security boundaries, deployment prerequisites and the approval process. Do not deploy or claim test success without actual tool evidence.

**Acceptance criteria:** The engineering plan must identify datasets, components, evaluation questions, task ownership, security constraints, approval gates and measurable performance targets. A future engineering runtime must generate source files and execute tests before any handoff can claim code completion.
