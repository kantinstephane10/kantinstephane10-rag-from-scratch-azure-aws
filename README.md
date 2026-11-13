# RAG From Scratch — Azure & AWS

A hands-on Retrieval-Augmented Generation (RAG) project built from scratch with Python, LangChain document utilities, Hugging Face embeddings, ChromaDB, and a small local Hugging Face instruction model.

The project uses a custom Azure and AWS cloud knowledge base and demonstrates the complete RAG learning path from raw text to a grounded answer.

## What I Built

This project implements the full RAG workflow step by step:

```text
Azure/AWS TXT Document
        ↓
Document Loading
        ↓
Text Cleaning
        ↓
Recursive Chunking
        ↓
Hugging Face Embeddings
all-MiniLM-L6-v2
        ↓
ChromaDB Vector Database
        ↓
User Question
        ↓
Question Embedding
        ↓
Semantic Vector Search
        ↓
Top-K Relevant Chunks
        ↓
Context + Question
        ↓
Hugging Face LLM
Qwen2.5-0.5B-Instruct
        ↓
Grounded Answer
```

## Learning Achievement

By building this project from scratch, I practiced and understood:

- document ingestion with LangChain `TextLoader`
- text parsing and cleaning
- recursive text chunking and chunk overlap
- LangChain `Document` objects and metadata
- sentence embeddings and 384-dimensional vectors
- semantic similarity and vector search
- persistent vector storage with ChromaDB
- Top-K retrieval
- context construction
- Retrieval-Augmented Generation
- grounding an LLM response in retrieved source material
- using free/local Hugging Face models without an OpenAI API

## Technology Stack

- Python 3.12+
- uv
- LangChain Community
- LangChain Text Splitters
- Sentence Transformers
- Hugging Face Transformers
- ChromaDB
- PyTorch

### Embedding Model

```text
sentence-transformers/all-MiniLM-L6-v2
```

This model converts each text chunk and user question into a 384-dimensional vector.

### Generation Model

```text
Qwen/Qwen2.5-0.5B-Instruct
```

The small instruction model keeps this learning project accessible on machines where larger local models may be slow.

## Project Structure

```text
.
├── data/
│   └── azure_aws_cloud_rag_training_20_pages.txt
├── src/
│   └── rag_pipeline.py
├── .gitignore
├── pyproject.toml
└── README.md
```

The generated local ChromaDB directory is intentionally excluded from Git. It is recreated by running the pipeline.

## Setup

Clone the repository:

```bash
git clone https://github.com/kantinstephane10/kantinstephane10-rag-from-scratch-azure-aws.git
cd kantinstephane10-rag-from-scratch-azure-aws
```

Install dependencies with uv:

```bash
uv sync
```

Run the RAG pipeline:

```bash
uv run python src/rag_pipeline.py
```

The first run may download the Hugging Face embedding and generation models.

## Example Question

```text
How does Azure provide private connectivity to storage?
```

The system:

1. embeds the question,
2. searches ChromaDB,
3. retrieves the most relevant Azure/AWS chunks,
4. combines them into context,
5. sends only that context and the question to the LLM.

A grounded answer should identify **Azure Private Endpoint** as the direct method described in the training document.

## Key RAG Principle

The LLM does **not** receive the entire knowledge base.

```text
Question
   ↓
Vector Search
   ↓
Relevant Chunks Only
   ↓
LLM
```

This is the core idea behind Retrieval-Augmented Generation.

## Current Scope

This repository demonstrates a foundational local RAG system for learning and experimentation.

Future improvements can include:

- PDF, Word, CSV, Excel, and database ingestion
- source citations in generated answers
- retrieval quality evaluation
- metadata filtering
- reranking
- a Streamlit interface
- Ollama as an alternative generation backend
- LangChain retriever abstractions
- conversational memory
- production API deployment

## Author

Built as a hands-on learning project to understand RAG architecture from first principles rather than relying on a prebuilt end-to-end chain.
