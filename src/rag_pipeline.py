from pathlib import Path
import re

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama


# ============================================================
# RAG V4 - MBA PDF + OLLAMA
#
# PDF
#   -> Clean
#   -> Recursive Chunking
#   -> Noise Filtering
#   -> MiniLM Embeddings
#   -> ChromaDB
#   -> Top-8 Retrieval
#   -> CrossEncoder Reranking
#   -> Best 3
#   -> Ollama Qwen3
#   -> Grounded Answer + Source Pages
# ============================================================


PDF_FILE = Path(
    "data/text_files/pdf/"
    "WGU_MBA_IT_Management_Bound_Edition_Biography_At_Beginning.pdf"
)

CHROMA_PATH = "./chroma_db_mba_v4_ollama"
COLLECTION_NAME = "wgu_mba_rag_v4_ollama"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
OLLAMA_MODEL = "qwen3:4b-instruct"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
RETRIEVE_K = 8
FINAL_K = 3


def clean_text(text):
    text = re.sub(
        r"WGU\s*[|•]\s*MBA IN INFORMATION TECHNOLOGY MANAGEMENT",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"EDITORIAL READER[’']S GUIDE\s*•?\s*"
        r"Original assignment follows unchanged",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"MBA IN INFORMATION TECHNOLOGY MANAGEMENT\s*[|•]\s*KANTIN TONGUE",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = text.replace("\t", " ")
    text = re.sub(r"(?m)^\s*\d{1,3}\s*$", "", text)
    text = re.sub(r"[ ]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def get_course_info(pdf_page):
    if 1 <= pdf_page <= 6:
        return (
            "Front Matter",
            "Biography, Preface, Contents, and Editorial Material",
        )

    if 7 <= pdf_page <= 27:
        return (
            "C200",
            "Managing Organizations and Leading People",
        )

    if 28 <= pdf_page <= 58:
        return (
            "EHM2",
            "Ethical Leadership",
        )

    if 59 <= pdf_page <= 70:
        return (
            "C212-KLM2",
            "Marketing",
        )

    if 71 <= pdf_page <= 95:
        return (
            "MGT2-KKM1",
            "Project Management",
        )

    if 96 <= pdf_page <= 132:
        return (
            "MMT2-BEN1",
            "IT Strategic Solutions",
        )

    if 133 <= pdf_page <= 143:
        return (
            "C207-QUM4",
            "Data-Driven Decision Making",
        )

    return ("Unknown", "Unknown")


def is_low_value_chunk(text):
    text_lower = text.lower()

    cover_phrases = [
        "survey completion date",
        "father of strengths psychology",
        "inventor of cliftonstrengths",
    ]

    legal_phrases = [
        "copyright",
        "all rights reserved",
        "trademark",
        "trademarks",
        "grant a license",
        "patents",
        "any webpage",
    ]

    cover_score = sum(
        phrase in text_lower
        for phrase in cover_phrases
    )

    legal_score = sum(
        phrase in text_lower
        for phrase in legal_phrases
    )

    return cover_score >= 2 or legal_score >= 3


def load_and_chunk_pdf():
    loader = PyMuPDFLoader(str(PDF_FILE))
    documents = loader.load()

    print("Pages loaded:", len(documents))

    for document in documents:
        document.page_content = clean_text(
            document.page_content
        )

        pdf_page = document.metadata["page"] + 1
        course, chapter = get_course_info(pdf_page)

        document.metadata["pdf_page"] = pdf_page
        document.metadata["course"] = course
        document.metadata["chapter"] = chapter
        document.metadata["source"] = PDF_FILE.name

    print("Text cleaned and metadata added")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_number"] = i
        chunk.metadata["chunk_characters"] = len(
            chunk.page_content
        )

    course_chunks = [
        chunk
        for chunk in chunks
        if chunk.metadata["course"] != "Front Matter"
    ]

    rag_chunks = [
        chunk
        for chunk in course_chunks
        if not is_low_value_chunk(chunk.page_content)
    ]

    for i, chunk in enumerate(rag_chunks):
        chunk.metadata["rag_chunk_number"] = i

    print("All chunks:", len(chunks))
    print("Course chunks:", len(course_chunks))
    print("Indexed RAG chunks:", len(rag_chunks))

    return rag_chunks


def build_vector_database(rag_chunks, embedding_model):
    chunk_texts = [
        chunk.page_content
        for chunk in rag_chunks
    ]

    print("Creating embeddings...")

    embeddings = embedding_model.encode(
        chunk_texts,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    print("Embedding shape:", embeddings.shape)

    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    try:
        client.delete_collection(
            COLLECTION_NAME
        )
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine"
        },
    )

    ids = [
        f"mba_chunk_{i:04d}"
        for i in range(len(rag_chunks))
    ]

    metadatas = []

    for chunk in rag_chunks:
        metadatas.append(
            {
                "source": chunk.metadata["source"],
                "pdf_page": chunk.metadata["pdf_page"],
                "course": chunk.metadata["course"],
                "chapter": chunk.metadata["chapter"],
                "original_chunk_number": chunk.metadata[
                    "chunk_number"
                ],
                "rag_chunk_number": chunk.metadata[
                    "rag_chunk_number"
                ],
                "chunk_characters": chunk.metadata[
                    "chunk_characters"
                ],
            }
        )

    collection.add(
        ids=ids,
        documents=chunk_texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
    )

    print(
        "Vectors stored in ChromaDB:",
        collection.count(),
    )

    return collection


def retrieve_and_rerank(
    question,
    collection,
    embedding_model,
    reranker,
):
    question_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True,
    )

    results = collection.query(
        query_embeddings=question_embedding.tolist(),
        n_results=RETRIEVE_K,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    pairs = [
        (question, document)
        for document in documents
    ]

    reranker_scores = reranker.predict(
        pairs
    )

    ranked_results = []

    for i in range(len(documents)):
        ranked_results.append(
            {
                "document": documents[i],
                "metadata": metadatas[i],
                "vector_distance": float(
                    distances[i]
                ),
                "reranker_score": float(
                    reranker_scores[i]
                ),
            }
        )

    ranked_results.sort(
        key=lambda item: item["reranker_score"],
        reverse=True,
    )

    return ranked_results[:FINAL_K]


def build_context(retrieved_results):
    context_parts = []

    for i, result in enumerate(
        retrieved_results,
        start=1,
    ):
        metadata = result["metadata"]

        context_parts.append(
            f"""SOURCE {i}

Course: {metadata["course"]}
Chapter: {metadata["chapter"]}
PDF Page: {metadata["pdf_page"]}

{result["document"]}"""
        )

    return "\n\n".join(context_parts)


def generate_answer(
    question,
    retrieved_results,
    llm,
):
    context = build_context(
        retrieved_results
    )

    system_message = """
You are a question-answering assistant for an MBA academic portfolio.

Answer using ONLY the retrieved document context.

Rules:
1. Do not use outside knowledge.
2. Do not invent information.
3. If the answer is not in the context, say:
   "I don't have enough information in the provided document to answer that."
4. Answer clearly and concisely.
5. Cite the course and PDF page when possible.
6. Use citation format: [C200, PDF page 10]
7. Do not mention embeddings, ChromaDB, vectors, reranking, or chunks.
"""

    user_message = f"""
Retrieved document context:

{context}

Question:

{question}
"""

    response = llm.invoke(
        [
            ("system", system_message),
            ("human", user_message),
        ]
    )

    return response.content.strip()


def ask_rag(
    question,
    collection,
    embedding_model,
    reranker,
    llm,
):
    sources = retrieve_and_rerank(
        question,
        collection,
        embedding_model,
        reranker,
    )

    answer = generate_answer(
        question,
        sources,
        llm,
    )

    return answer, sources


def main():
    if not PDF_FILE.exists():
        raise FileNotFoundError(
            "\nPrivate PDF not found.\n"
            f"Place it locally at:\n{PDF_FILE}\n\n"
            "PDF files are intentionally ignored by Git."
        )

    print("Loading embedding model...")

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    print("Loading reranker...")

    reranker = CrossEncoder(
        RERANKER_MODEL
    )

    print("Connecting to Ollama...")

    llm = ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0,
    )

    rag_chunks = load_and_chunk_pdf()

    collection = build_vector_database(
        rag_chunks,
        embedding_model,
    )

    print("\nRAG is ready.")

    question = input(
        "\nAsk a question about the MBA document: "
    ).strip()

    if not question:
        question = (
            "What are my five CliftonStrengths?"
        )

    answer, sources = ask_rag(
        question,
        collection,
        embedding_model,
        reranker,
        llm,
    )

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(answer)

    print("\n" + "=" * 70)
    print("RETRIEVED SOURCES")
    print("=" * 70)

    for i, result in enumerate(
        sources,
        start=1,
    ):
        metadata = result["metadata"]

        print(
            f"\nSource {i}: "
            f'{metadata["course"]} | '
            f'PDF page {metadata["pdf_page"]} | '
            f'reranker score '
            f'{result["reranker_score"]:.4f}'
        )


if __name__ == "__main__":
    main()
