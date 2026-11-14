from pathlib import Path
import re

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama


# ============================================================
# RAG V8 - QUERY-AWARE RETRIEVAL + DETERMINISTIC CITATIONS
#
# FOCUSED QUESTION
#   -> ChromaDB Top 8
#   -> Neighbor Expansion
#   -> CrossEncoder
#   -> Best 3
#
# BROAD / PORTFOLIO QUESTION
#   -> Search every course separately
#   -> Neighbor Expansion inside each course
#   -> CrossEncoder inside each course
#   -> Best evidence from every course
#
# BOTH PATHS
#   -> Ollama writes answer + [SOURCE N] markers
#   -> Python replaces source markers with real metadata
#   -> Final grounded answer
# ============================================================


# ============================================================
# 1. FILE + MODEL SETTINGS
# ============================================================

PDF_FILE = Path(
    "data/text_files/pdf/"
    "WGU_MBA_IT_Management_Bound_Edition_Biography_At_Beginning.pdf"
)

CHROMA_PATH = "./chroma_db_mba_v8"
COLLECTION_NAME = "wgu_mba_rag_v8"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
OLLAMA_MODEL = "qwen3:4b-instruct"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# Focused question settings
RETRIEVE_K = 8
FINAL_K = 3
NEIGHBOR_RANGE = 1

# Broad question settings
BROAD_PER_COURSE_K = 4
BROAD_FINAL_PER_COURSE = 2

COURSE_CODES = [
    "C200",
    "EHM2",
    "C212-KLM2",
    "MGT2-KKM1",
    "MMT2-BEN1",
    "C207-QUM4",
]


# ============================================================
# 2. CLEAN TEXT
# ============================================================

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


# ============================================================
# 3. COURSE METADATA
# ============================================================

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

    return (
        "Unknown",
        "Unknown",
    )


# ============================================================
# 4. LOW-VALUE CHUNK FILTER
# ============================================================

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

    return (
        cover_score >= 2
        or
        legal_score >= 3
    )


# ============================================================
# 5. LOAD + CHUNK PDF
# ============================================================

def load_and_chunk_pdf():

    loader = PyMuPDFLoader(
        str(PDF_FILE)
    )

    documents = loader.load()

    print(
        "Pages loaded:",
        len(documents),
    )

    for document in documents:

        document.page_content = clean_text(
            document.page_content
        )

        pdf_page = (
            document.metadata["page"]
            + 1
        )

        course, chapter = get_course_info(
            pdf_page
        )

        document.metadata[
            "pdf_page"
        ] = pdf_page

        document.metadata[
            "course"
        ] = course

        document.metadata[
            "chapter"
        ] = chapter

        document.metadata[
            "source"
        ] = PDF_FILE.name

    print(
        "Text cleaned"
    )

    print(
        "Metadata added"
    )

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

    chunks = splitter.split_documents(
        documents
    )

    for i, chunk in enumerate(
        chunks
    ):

        chunk.metadata[
            "chunk_number"
        ] = i

        chunk.metadata[
            "chunk_characters"
        ] = len(
            chunk.page_content
        )

    course_chunks = [
        chunk
        for chunk in chunks
        if chunk.metadata[
            "course"
        ] != "Front Matter"
    ]

    rag_chunks = [
        chunk
        for chunk in course_chunks
        if not is_low_value_chunk(
            chunk.page_content
        )
    ]

    for i, chunk in enumerate(
        rag_chunks
    ):

        chunk.metadata[
            "rag_chunk_number"
        ] = i

    print(
        "All chunks:",
        len(chunks),
    )

    print(
        "Course chunks:",
        len(course_chunks),
    )

    print(
        "Indexed RAG chunks:",
        len(rag_chunks),
    )

    return rag_chunks


# ============================================================
# 6. BUILD CHROMADB
# ============================================================

def build_vector_database(
    rag_chunks,
    embedding_model,
):

    chunk_texts = [
        chunk.page_content
        for chunk in rag_chunks
    ]

    print(
        "Creating embeddings..."
    )

    embeddings = embedding_model.encode(
        chunk_texts,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    print(
        "Embedding shape:",
        embeddings.shape,
    )

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
            "hnsw:space":
                "cosine",
        },
    )

    ids = [
        f"mba_chunk_{i:04d}"
        for i in range(
            len(rag_chunks)
        )
    ]

    metadatas = []

    for chunk in rag_chunks:

        metadatas.append(
            {
                "source":
                    chunk.metadata[
                        "source"
                    ],

                "pdf_page":
                    chunk.metadata[
                        "pdf_page"
                    ],

                "course":
                    chunk.metadata[
                        "course"
                    ],

                "chapter":
                    chunk.metadata[
                        "chapter"
                    ],

                "original_chunk_number":
                    chunk.metadata[
                        "chunk_number"
                    ],

                "rag_chunk_number":
                    chunk.metadata[
                        "rag_chunk_number"
                    ],

                "chunk_characters":
                    chunk.metadata[
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


# ============================================================
# 7. QUERY TYPE ROUTER
# ============================================================

def detect_query_type(
    question,
):

    question_lower = (
        question.lower().strip()
    )

    broad_patterns = [
        r"\ball courses\b",
        r"\bevery course\b",
        r"\bwhich courses\b",
        r"\bwhat courses\b",
        r"\bidentify all\b",
        r"\blist all\b",
        r"\bacross (?:my |the )?(?:mba )?portfolio\b",
        r"\bacross (?:all )?courses\b",
        r"\bthroughout (?:my |the )?(?:mba )?portfolio\b",
        r"\bcompare (?:the )?courses\b",
    ]

    for pattern in broad_patterns:

        if re.search(
            pattern,
            question_lower,
        ):

            return "broad"

    return "focused"


# ============================================================
# 8. EMBED QUESTION
# ============================================================

def embed_question(
    question,
    embedding_model,
):

    return embedding_model.encode(
        [question],
        normalize_embeddings=True,
    )


# ============================================================
# 9. CHROMA RESULT -> CANDIDATES
# ============================================================

def results_to_candidates(
    results,
):

    documents_found = (
        results["documents"][0]
    )

    metadata_found = (
        results["metadatas"][0]
    )

    distances_found = (
        results["distances"][0]
    )

    candidates = []

    for i in range(
        len(documents_found)
    ):

        candidates.append(
            {
                "document":
                    documents_found[i],

                "metadata":
                    metadata_found[i],

                "vector_distance":
                    float(
                        distances_found[i]
                    ),

                "retrieval_source":
                    "semantic",
            }
        )

    return candidates


# ============================================================
# 10. NEIGHBOR EXPANSION
# ============================================================

def expand_candidates_with_neighbors(
    candidates,
    rag_chunk_lookup,
):

    expanded = {}

    for candidate in candidates:

        metadata = candidate[
            "metadata"
        ]

        rag_chunk_number = metadata[
            "rag_chunk_number"
        ]

        for offset in range(
            -NEIGHBOR_RANGE,
            NEIGHBOR_RANGE + 1,
        ):

            neighbor_number = (
                rag_chunk_number
                + offset
            )

            if (
                neighbor_number
                not in rag_chunk_lookup
            ):

                continue

            neighbor_chunk = (
                rag_chunk_lookup[
                    neighbor_number
                ]
            )

            if (
                neighbor_chunk.metadata[
                    "course"
                ]
                !=
                metadata[
                    "course"
                ]
            ):

                continue

            if offset == 0:

                vector_distance = (
                    candidate[
                        "vector_distance"
                    ]
                )

                retrieval_source = (
                    "semantic"
                )

            else:

                vector_distance = None

                retrieval_source = (
                    "neighbor"
                )

            if (
                neighbor_number
                not in expanded
            ):

                expanded[
                    neighbor_number
                ] = {
                    "document":
                        neighbor_chunk.page_content,

                    "metadata": {
                        "source":
                            neighbor_chunk.metadata[
                                "source"
                            ],

                        "pdf_page":
                            neighbor_chunk.metadata[
                                "pdf_page"
                            ],

                        "course":
                            neighbor_chunk.metadata[
                                "course"
                            ],

                        "chapter":
                            neighbor_chunk.metadata[
                                "chapter"
                            ],

                        "rag_chunk_number":
                            neighbor_chunk.metadata[
                                "rag_chunk_number"
                            ],
                    },

                    "vector_distance":
                        vector_distance,

                    "retrieval_source":
                        retrieval_source,
                }

            elif offset == 0:

                expanded[
                    neighbor_number
                ][
                    "vector_distance"
                ] = vector_distance

                expanded[
                    neighbor_number
                ][
                    "retrieval_source"
                ] = "semantic"

    return list(
        expanded.values()
    )


# ============================================================
# 11. RERANK CANDIDATES
# ============================================================

def rerank_candidates(
    question,
    candidates,
    reranker,
):

    if not candidates:

        return []

    pairs = [
        (
            question,
            candidate[
                "document"
            ],
        )
        for candidate
        in candidates
    ]

    scores = reranker.predict(
        pairs
    )

    for i in range(
        len(candidates)
    ):

        candidates[i][
            "reranker_score"
        ] = float(
            scores[i]
        )

    candidates.sort(
        key=lambda item:
            item[
                "reranker_score"
            ],
        reverse=True,
    )

    return candidates


# ============================================================
# 12. FOCUSED RETRIEVAL
# ============================================================

def retrieve_focused(
    question,
    collection,
    embedding_model,
    reranker,
    rag_chunk_lookup,
):

    question_embedding = embed_question(
        question,
        embedding_model,
    )

    results = collection.query(
        query_embeddings=(
            question_embedding.tolist()
        ),
        n_results=RETRIEVE_K,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    candidates = results_to_candidates(
        results
    )

    candidates = expand_candidates_with_neighbors(
        candidates,
        rag_chunk_lookup,
    )

    candidates = rerank_candidates(
        question,
        candidates,
        reranker,
    )

    final_results = candidates[
        :FINAL_K
    ]

    for result in final_results:

        result[
            "query_type"
        ] = "focused"

    return final_results


# ============================================================
# 13. BROAD / COURSE-AWARE RETRIEVAL
# ============================================================

def retrieve_broad(
    question,
    collection,
    embedding_model,
    reranker,
    rag_chunk_lookup,
):

    question_embedding = embed_question(
        question,
        embedding_model,
    )

    final_results = []

    for course in COURSE_CODES:

        results = collection.query(
            query_embeddings=(
                question_embedding.tolist()
            ),
            n_results=BROAD_PER_COURSE_K,
            where={
                "course":
                    course
            },
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        course_candidates = (
            results_to_candidates(
                results
            )
        )

        course_candidates = (
            expand_candidates_with_neighbors(
                course_candidates,
                rag_chunk_lookup,
            )
        )

        course_candidates = (
            rerank_candidates(
                question,
                course_candidates,
                reranker,
            )
        )

        selected = course_candidates[
            :BROAD_FINAL_PER_COURSE
        ]

        for result in selected:

            result[
                "query_type"
            ] = "broad"

            result[
                "course_search"
            ] = course

        final_results.extend(
            selected
        )

    return final_results


# ============================================================
# 14. QUERY-AWARE RETRIEVAL
# ============================================================

def retrieve_for_question(
    question,
    collection,
    embedding_model,
    reranker,
    rag_chunk_lookup,
):

    query_type = detect_query_type(
        question
    )

    if query_type == "broad":

        sources = retrieve_broad(
            question,
            collection,
            embedding_model,
            reranker,
            rag_chunk_lookup,
        )

    else:

        sources = retrieve_focused(
            question,
            collection,
            embedding_model,
            reranker,
            rag_chunk_lookup,
        )

    return (
        query_type,
        sources,
    )


# ============================================================
# 15. BUILD CONTEXT
# ============================================================

def build_context(
    retrieved_results,
):

    context_parts = []

    for i, result in enumerate(
        retrieved_results,
        start=1,
    ):

        metadata = result[
            "metadata"
        ]

        context_parts.append(
            f"""SOURCE {i}

Course: {metadata["course"]}
Course Title: {metadata["chapter"]}
PDF Page: {metadata["pdf_page"]}

{result["document"]}"""
        )

    return "\n\n".join(
        context_parts
    )


# ============================================================
# 16. DETERMINISTIC CITATIONS
# ============================================================

def replace_source_markers(
    answer,
    retrieved_results,
):

    # Remove any direct course/page citation the LLM created.
    answer = re.sub(
        r"\[[A-Za-z0-9/-]+\s*,\s*PDF page\s*\d+\]",
        "",
        answer,
        flags=re.IGNORECASE,
    )

    source_map = {}

    for i, result in enumerate(
        retrieved_results,
        start=1,
    ):

        metadata = result[
            "metadata"
        ]

        source_map[i] = (
            f"[{metadata['course']}, "
            f"PDF page "
            f"{metadata['pdf_page']}]"
        )

    def replace_marker(
        match,
    ):

        source_number = int(
            match.group(1)
        )

        return source_map.get(
            source_number,
            "",
        )

    answer = re.sub(
        r"\[SOURCE\s+(\d+)\]",
        replace_marker,
        answer,
        flags=re.IGNORECASE,
    )

    answer = re.sub(
        r"[ \t]{2,}",
        " ",
        answer,
    )

    return answer.strip()


# ============================================================
# 17. GENERATE ANSWER
# ============================================================

def generate_answer(
    question,
    query_type,
    retrieved_results,
    llm,
):

    context = build_context(
        retrieved_results
    )

    if query_type == "broad":

        task_rules = """
This is a BROAD portfolio question.

Evidence has been retrieved separately from EVERY MBA course.

Rules for broad questions:
- Evaluate each course separately.
- Include a course ONLY when its retrieved text directly supports the user's topic.
- Do not say a course is irrelevant merely because another course is more relevant.
- Do not claim "only one course" unless the supplied evidence from the other courses does not support the topic.
- When listing courses, provide the course code, course title, how the topic is discussed, and the relevant PDF page when requested.
- Put a source marker after each supported course or major claim.
"""

    else:

        task_rules = """
This is a FOCUSED question.

Use the strongest retrieved evidence to answer directly and concisely.
Put a source marker after the factual claim it supports.
"""

    system_message = f"""
You are a question-answering assistant for an MBA academic portfolio.

Answer using ONLY the retrieved document context.

{task_rules}

Citation rules:
1. NEVER create a course/page citation yourself.
2. NEVER write citations like [C200, PDF page 10].
3. Cite evidence ONLY with the exact source marker format:
   [SOURCE 1]
   [SOURCE 2]
   [SOURCE 3]
4. Use only source numbers that appear in the retrieved context.
5. Python will convert the source markers to real course/page citations after you answer.

Grounding rules:
1. Do not use outside knowledge.
2. Do not invent information.
3. If the retrieved context does not contain enough information to answer, say exactly:
   I don't have enough information in the provided document to answer that.
4. Do not mention embeddings, vectors, ChromaDB, reranking, or chunks.
"""

    user_message = f"""
Retrieved document context:

{context}

Question:

{question}
"""

    response = llm.invoke(
        [
            (
                "system",
                system_message,
            ),
            (
                "human",
                user_message,
            ),
        ]
    )

    raw_answer = (
        response.content.strip()
    )

    final_answer = replace_source_markers(
        raw_answer,
        retrieved_results,
    )

    return final_answer


# ============================================================
# 18. COMPLETE RAG FUNCTION
# ============================================================

def ask_rag(
    question,
    collection,
    embedding_model,
    reranker,
    rag_chunk_lookup,
    llm,
):

    query_type, sources = (
        retrieve_for_question(
            question,
            collection,
            embedding_model,
            reranker,
            rag_chunk_lookup,
        )
    )

    answer = generate_answer(
        question,
        query_type,
        sources,
        llm,
    )

    return (
        answer,
        sources,
    )


# ============================================================
# 19. CLI TEST
# ============================================================

def main():

    if not PDF_FILE.exists():

        raise FileNotFoundError(
            "\nMBA PDF not found.\n"
            f"Expected path:\n{PDF_FILE}\n"
        )

    print(
        "Loading embedding model..."
    )

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    print(
        "Loading reranker..."
    )

    reranker = CrossEncoder(
        RERANKER_MODEL
    )

    print(
        "Connecting to Ollama..."
    )

    llm = ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0,
    )

    rag_chunks = load_and_chunk_pdf()

    rag_chunk_lookup = {
        chunk.metadata[
            "rag_chunk_number"
        ]:
            chunk
        for chunk in rag_chunks
    }

    collection = build_vector_database(
        rag_chunks,
        embedding_model,
    )

    print(
        "\nRAG V8 is ready."
    )

    question = input(
        "\nAsk a question: "
    ).strip()

    if not question:

        question = (
            "Identify all courses in my MBA portfolio "
            "that discuss project management. "
            "For each course, provide the course code, "
            "course title, how project management is discussed, "
            "and the PDF page number."
        )

    query_type = detect_query_type(
        question
    )

    print(
        "\nQuery type:",
        query_type,
    )

    answer, sources = ask_rag(
        question,
        collection,
        embedding_model,
        reranker,
        rag_chunk_lookup,
        llm,
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "ANSWER"
    )

    print(
        "=" * 70
    )

    print(
        answer
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "RETRIEVED SOURCES"
    )

    print(
        "=" * 70
    )

    for i, result in enumerate(
        sources,
        start=1,
    ):

        metadata = result[
            "metadata"
        ]

        print(
            f"\nSource {i}: "
            f"{metadata['course']} | "
            f"{metadata['chapter']} | "
            f"PDF page "
            f"{metadata['pdf_page']} | "
            f"{result['retrieval_source']} | "
            f"score "
            f"{result['reranker_score']:.4f}"
        )


if __name__ == "__main__":
    main()
