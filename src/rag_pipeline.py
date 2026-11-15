from pathlib import Path
import os
import re
import time

from pinecone import DenseVectorQuery, Pinecone, SchemaBuilder
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama


# ============================================================
# RAG V11 - PINECONE VECTOR DATABASE
#
# V9:
#   - conversational memory
#   - focused vs broad query routing
#   - neighbor expansion
#   - CrossEncoder reranking
#   - deterministic citations
#
# V10:
#   - SQLite persistent conversational memory in memory_db.py
#
# V11:
#   - Pinecone replaces ChromaDB as the vector database
#   - MiniLM embeddings remain local
#   - retrieval/reranking/generation behavior remains unchanged
# ============================================================


PDF_FILE = Path(
    "data/text_files/pdf/"
    "WGU_MBA_IT_Management_Bound_Edition_Biography_At_Beginning.pdf"
)

PINECONE_INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME",
    "mba-rag-v11",
)

PINECONE_NAMESPACE = os.getenv(
    "PINECONE_NAMESPACE",
    "mba-portfolio",
)

PINECONE_VECTOR_FIELD = "embedding"
PINECONE_DIMENSION = 384

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
OLLAMA_MODEL = "qwen3:4b-instruct"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

RETRIEVE_K = 8
FINAL_K = 3
NEIGHBOR_RANGE = 1

BROAD_PER_COURSE_K = 4
BROAD_FINAL_PER_COURSE = 2

MEMORY_MAX_MESSAGES = 12

REFUSAL_MESSAGE = (
    "I don't have enough information "
    "in the provided document to answer that."
)

COURSE_CODES = [
    "C200",
    "EHM2",
    "C212-KLM2",
    "MGT2-KKM1",
    "MMT2-BEN1",
    "C207-QUM4",
]


# ============================================================
# TEXT + METADATA
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
        r"MBA IN INFORMATION TECHNOLOGY MANAGEMENT\s*"
        r"[|•]\s*KANTIN TONGUE",
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

    return (
        cover_score >= 2
        or legal_score >= 3
    )


def load_and_chunk_pdf():
    if not PDF_FILE.exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_FILE}"
        )

    loader = PyMuPDFLoader(
        str(PDF_FILE)
    )

    documents = loader.load()

    print("Pages loaded:", len(documents))

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

        document.metadata["pdf_page"] = pdf_page
        document.metadata["course"] = course
        document.metadata["chapter"] = chapter
        document.metadata["source"] = PDF_FILE.name

    print("Text cleaned")
    print("Metadata added")

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

    print("All chunks:", len(chunks))

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

    print("Course chunks:", len(course_chunks))

    rag_chunks = [
        chunk
        for chunk in course_chunks
        if not is_low_value_chunk(
            chunk.page_content
        )
    ]

    print("Final indexed chunks:", len(rag_chunks))

    for i, chunk in enumerate(rag_chunks):
        chunk.metadata["rag_chunk_number"] = i

    return rag_chunks


# ============================================================
# PINECONE VECTOR DATABASE
# ============================================================

def get_pinecone_client():
    api_key = os.getenv(
        "PINECONE_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "PINECONE_API_KEY is not set. "
            "Set it before running V11."
        )

    return Pinecone(
        api_key=api_key
    )


def ensure_pinecone_index(
    pinecone_client,
):
    if not pinecone_client.indexes.exists(
        PINECONE_INDEX_NAME
    ):
        print(
            "Creating Pinecone index:",
            PINECONE_INDEX_NAME,
        )

        schema = (
            SchemaBuilder()
            .add_dense_vector_field(
                PINECONE_VECTOR_FIELD,
                dimension=PINECONE_DIMENSION,
                metric="cosine",
            )
            .build()
        )

        # Current Pinecone SDK behavior:
        # without an explicit deployment, this creates
        # a managed/serverless index on AWS us-east-1.
        pinecone_client.indexes.create(
            name=PINECONE_INDEX_NAME,
            schema=schema,
        )

        print(
            "Pinecone index created ✅"
        )

    else:
        print(
            "Pinecone index already exists:",
            PINECONE_INDEX_NAME,
        )

        index_info = (
            pinecone_client.indexes.describe(
                PINECONE_INDEX_NAME
            )
        )

        fields = (
            index_info
            .schema
            .fields
        )

        if (
            PINECONE_VECTOR_FIELD
            not in fields
        ):
            raise RuntimeError(
                "Existing Pinecone index has an "
                "incompatible schema. Expected "
                f"vector field: {PINECONE_VECTOR_FIELD}"
            )

        vector_field = fields[
            PINECONE_VECTOR_FIELD
        ]

        if (
            int(vector_field.dimension)
            !=
            PINECONE_DIMENSION
        ):
            raise RuntimeError(
                "Existing Pinecone index dimension "
                f"is {vector_field.dimension}, "
                f"but V11 requires "
                f"{PINECONE_DIMENSION}."
            )

    return pinecone_client.index(
        PINECONE_INDEX_NAME
    )


def wait_for_pinecone_documents(
    index,
    document_ids,
    timeout_seconds=60,
):
    if not document_ids:
        return

    sample_ids = [
        document_ids[0],
        document_ids[-1],
    ]

    # Remove duplicate when there is only one document.
    sample_ids = list(
        dict.fromkeys(
            sample_ids
        )
    )

    deadline = (
        time.time()
        +
        timeout_seconds
    )

    while time.time() < deadline:
        response = (
            index.documents.fetch(
                namespace=PINECONE_NAMESPACE,
                ids=sample_ids,
            )
        )

        if (
            len(response.documents)
            ==
            len(sample_ids)
        ):
            return

        time.sleep(1)

    raise TimeoutError(
        "Pinecone documents were accepted but "
        "did not become searchable within "
        f"{timeout_seconds} seconds."
    )


def build_vector_database(
    rag_chunks,
    embedding_model,
):
    chunk_texts = [
        chunk.page_content
        for chunk in rag_chunks
    ]

    print(
        "\nCreating embeddings..."
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

    if (
        embeddings.shape[1]
        !=
        PINECONE_DIMENSION
    ):
        raise RuntimeError(
            "Embedding dimension does not match "
            "the Pinecone schema. "
            f"Got {embeddings.shape[1]}, "
            f"expected {PINECONE_DIMENSION}."
        )

    pinecone_client = (
        get_pinecone_client()
    )

    index = ensure_pinecone_index(
        pinecone_client
    )

    ids = [
        f"mba_chunk_{i:04d}"
        for i in range(
            len(rag_chunks)
        )
    ]

    pinecone_documents = []

    for i, chunk in enumerate(
        rag_chunks
    ):
        pinecone_documents.append(
            {
                "_id":
                    ids[i],

                PINECONE_VECTOR_FIELD:
                    embeddings[i].tolist(),

                "text":
                    chunk.page_content,

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

                "rag_chunk_number":
                    chunk.metadata[
                        "rag_chunk_number"
                    ],
            }
        )

    print(
        "Upserting documents to Pinecone..."
    )

    result = (
        index.documents.batch_upsert(
            namespace=PINECONE_NAMESPACE,
            documents=pinecone_documents,
        )
    )

    if result.errors:
        print(
            "Pinecone upsert errors:"
        )

        for error in result.errors:
            print(
                error.batch_index,
                error.error_message,
            )

        raise RuntimeError(
            "Pinecone batch upsert did not "
            "complete successfully."
        )

    wait_for_pinecone_documents(
        index,
        ids,
    )

    print(
        "Pinecone documents ready:",
        len(ids),
    )

    return index


def pinecone_search(
    index,
    query_embedding,
    top_k,
    course=None,
):
    metadata_filter = None

    if course is not None:
        metadata_filter = {
            "course": {
                "$eq": course,
            }
        }

    results = (
        index.documents.search(
            namespace=PINECONE_NAMESPACE,

            top_k=top_k,

            score_by=[
                DenseVectorQuery(
                    field=PINECONE_VECTOR_FIELD,
                    values=query_embedding,
                )
            ],

            include_fields=[
                "text",
                "source",
                "pdf_page",
                "course",
                "chapter",
                "rag_chunk_number",
            ],

            filter=metadata_filter,
        )
    )

    candidates = []

    for match in results.matches:
        candidates.append(
            {
                "document":
                    match.get(
                        "text",
                        "",
                    ),

                "metadata": {
                    "source":
                        match.get(
                            "source",
                            "",
                        ),

                    "pdf_page":
                        int(
                            match.get(
                                "pdf_page",
                                0,
                            )
                        ),

                    "course":
                        match.get(
                            "course",
                            "",
                        ),

                    "chapter":
                        match.get(
                            "chapter",
                            "",
                        ),

                    "rag_chunk_number":
                        int(
                            match.get(
                                "rag_chunk_number",
                                0,
                            )
                        ),
                },

                # Pinecone cosine scores are similarity
                # scores where higher is better.
                "vector_score":
                    float(
                        match.score
                    ),

                "retrieval_source":
                    "semantic",
            }
        )

    return candidates


# ============================================================
# CONVERSATION MEMORY
# ============================================================

def format_conversation_history(
    conversation_history,
):
    if not conversation_history:
        return "No previous conversation."

    recent_messages = conversation_history[
        -MEMORY_MAX_MESSAGES:
    ]

    lines = []

    for message in recent_messages:
        role = message.get("role", "")
        content = message.get("content", "")

        if role == "user":
            label = "USER"
        elif role == "assistant":
            label = "ASSISTANT"
        else:
            continue

        lines.append(
            f"{label}: {content}"
        )

    return "\n".join(lines)


def contextualize_question(
    question,
    conversation_history,
    llm,
):
    if not conversation_history:
        return question

    history_text = format_conversation_history(
        conversation_history
    )

    system_message = """
You rewrite follow-up questions into standalone search questions.

Use the conversation history only to resolve references such as:
it, that, they, those, which one, the first one, the second one,
that course, that project, or that technology.

Rules:
1. Preserve the user's original meaning.
2. Do not answer the question.
3. Do not add facts.
4. If the question is already standalone, return it unchanged.
5. Return ONLY the standalone rewritten question.
"""

    user_message = f"""
Conversation history:

{history_text}

Current question:

{question}

Standalone search question:
"""

    response = llm.invoke(
        [
            ("system", system_message),
            ("human", user_message),
        ]
    )

    standalone_question = (
        response.content
        .strip()
        .strip('"')
    )

    if not standalone_question:
        return question

    return standalone_question


# ============================================================
# QUERY ROUTER
# ============================================================

def detect_query_type(question):
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
# RETRIEVAL HELPERS
# ============================================================

def embed_question(
    question,
    embedding_model,
):
    return embedding_model.encode(
        [question],
        normalize_embeddings=True,
    )


def add_neighbors(
    candidates,
    rag_chunk_lookup,
):
    expanded = {}

    for candidate in candidates:
        metadata = candidate["metadata"]

        current_number = metadata[
            "rag_chunk_number"
        ]

        for offset in range(
            -NEIGHBOR_RANGE,
            NEIGHBOR_RANGE + 1,
        ):
            neighbor_number = (
                current_number
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
                neighbor_chunk.metadata["course"]
                != metadata["course"]
            ):
                continue

            if offset == 0:
                retrieval_source = "semantic"
                vector_distance = candidate[
                    "vector_score"
                ]
            else:
                retrieval_source = "neighbor"
                vector_distance = None

            if neighbor_number not in expanded:
                expanded[neighbor_number] = {
                    "document":
                        neighbor_chunk.page_content,
                    "metadata": {
                        "source":
                            neighbor_chunk.metadata["source"],
                        "pdf_page":
                            neighbor_chunk.metadata["pdf_page"],
                        "course":
                            neighbor_chunk.metadata["course"],
                        "chapter":
                            neighbor_chunk.metadata["chapter"],
                        "rag_chunk_number":
                            neighbor_chunk.metadata[
                                "rag_chunk_number"
                            ],
                    },
                    "vector_score":
                        vector_distance,
                    "retrieval_source":
                        retrieval_source,
                }

            elif offset == 0:
                expanded[
                    neighbor_number
                ]["retrieval_source"] = "semantic"

                expanded[
                    neighbor_number
                ]["vector_score"] = vector_distance

    return list(
        expanded.values()
    )


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
            candidate["document"],
        )
        for candidate in candidates
    ]

    scores = reranker.predict(
        pairs
    )

    for i in range(len(candidates)):
        candidates[i][
            "reranker_score"
        ] = float(scores[i])

    candidates.sort(
        key=lambda item:
            item["reranker_score"],
        reverse=True,
    )

    return candidates


# ============================================================
# FOCUSED RETRIEVAL
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

    candidates = pinecone_search(
        collection,
        question_embedding[0].tolist(),
        RETRIEVE_K,
    )

    candidates = add_neighbors(
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
        result["query_type"] = "focused"

    return final_results


# ============================================================
# BROAD RETRIEVAL
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
        candidates = pinecone_search(
            collection,
            question_embedding[0].tolist(),
            BROAD_PER_COURSE_K,
            course=course,
        )

        candidates = add_neighbors(
            candidates,
            rag_chunk_lookup,
        )

        candidates = rerank_candidates(
            question,
            candidates,
            reranker,
        )

        selected = candidates[
            :BROAD_FINAL_PER_COURSE
        ]

        for result in selected:
            result["query_type"] = "broad"

        final_results.extend(
            selected
        )

    return final_results


def group_sources_by_course(sources):
    grouped = {}

    for source in sources:
        course = source[
            "metadata"
        ]["course"]

        if course not in grouped:
            grouped[course] = []

        grouped[course].append(
            source
        )

    return grouped


def filter_broad_sources_by_relevance(
    question,
    sources,
    llm,
):
    if not sources:
        return []

    grouped = group_sources_by_course(
        sources
    )

    evidence_parts = []

    for course in COURSE_CODES:
        course_sources = grouped.get(
            course,
            [],
        )

        if not course_sources:
            continue

        chapter = course_sources[
            0
        ]["metadata"]["chapter"]

        evidence_parts.append(
            f"""
COURSE CODE: {course}
COURSE TITLE: {chapter}
""".strip()
        )

        for i, source in enumerate(
            course_sources,
            start=1,
        ):
            evidence_parts.append(
                f"""
Evidence {i}
PDF Page: {source["metadata"]["pdf_page"]}

{source["document"]}
""".strip()
            )

    course_evidence = "\n\n".join(
        evidence_parts
    )

    allowed_codes = ", ".join(
        COURSE_CODES
    )

    system_message = f"""
You are a strict evidence relevance classifier.

The user asked a broad question across an MBA portfolio.

Decide which courses SUBSTANTIVELY discuss the topic.

Allowed course codes:

{allowed_codes}

A course is relevant only when the supplied evidence directly or
substantively discusses the requested topic.

Do NOT count a course merely because:
- it contains a generic related word;
- the topic could theoretically apply to that course;
- the course contains transferable skills;
- leadership or communication could be useful in the topic;
- there is only an indirect relationship;
- the topic is mentioned incidentally;
- another project happens to be mentioned.

For example, if the question asks which courses discuss project
management, a leadership course should NOT qualify merely because
leadership or communication can be used during projects.

Return ONLY relevant course codes separated by commas.

If no course is supported, return exactly:

NONE

Do not explain your answer.
"""

    user_message = f"""
USER QUESTION:

{question}

COURSE EVIDENCE:

{course_evidence}

Relevant course codes:
"""

    response = llm.invoke(
        [
            ("system", system_message),
            ("human", user_message),
        ]
    )

    raw_result = (
        response.content
        .strip()
        .upper()
    )

    if raw_result == "NONE":
        return []

    relevant_courses = []

    for course in COURSE_CODES:
        if course.upper() in raw_result:
            relevant_courses.append(
                course
            )

    return [
        source
        for source in sources
        if source["metadata"]["course"]
        in relevant_courses
    ]


def retrieve_for_question(
    standalone_question,
    collection,
    embedding_model,
    reranker,
    rag_chunk_lookup,
    llm,
):
    query_type = detect_query_type(
        standalone_question
    )

    if query_type == "broad":
        sources = retrieve_broad(
            standalone_question,
            collection,
            embedding_model,
            reranker,
            rag_chunk_lookup,
        )

        sources = filter_broad_sources_by_relevance(
            standalone_question,
            sources,
            llm,
        )

    else:
        sources = retrieve_focused(
            standalone_question,
            collection,
            embedding_model,
            reranker,
            rag_chunk_lookup,
        )

    return query_type, sources


# ============================================================
# CONTEXT + CITATIONS
# ============================================================

def build_context(sources):
    if not sources:
        return (
            "No supporting document evidence was found."
        )

    context_parts = []

    for i, source in enumerate(
        sources,
        start=1,
    ):
        metadata = source["metadata"]

        context_parts.append(
            f"""
SOURCE {i}

Course: {metadata["course"]}
Course Title: {metadata["chapter"]}
PDF Page: {metadata["pdf_page"]}

{source["document"]}
""".strip()
        )

    return "\n\n".join(
        context_parts
    )


def deduplicate_citation_blocks(answer):
    citation_pattern = (
        r"\[[A-Za-z0-9/-]+,\s*"
        r"PDF page\s*\d+\]"
    )

    block_pattern = (
        rf"(?:{citation_pattern}\s*){{2,}}"
    )

    def clean_block(match):
        block = match.group(0)

        citations = re.findall(
            citation_pattern,
            block,
            flags=re.IGNORECASE,
        )

        unique = []

        for citation in citations:
            if citation not in unique:
                unique.append(
                    citation
                )

        return " ".join(unique)

    return re.sub(
        block_pattern,
        clean_block,
        answer,
        flags=re.IGNORECASE,
    )


def remove_conflicting_refusal(answer):
    answer = answer.strip()

    if answer == REFUSAL_MESSAGE:
        return answer

    if REFUSAL_MESSAGE in answer:
        answer = answer.replace(
            REFUSAL_MESSAGE,
            "",
        )

        answer = re.sub(
            r"\n{3,}",
            "\n\n",
            answer,
        ).strip()

    return answer


def replace_source_markers(
    answer,
    sources,
):
    answer = re.sub(
        r"\[[A-Za-z0-9/-]+\s*,\s*PDF page\s*\d+\]",
        "",
        answer,
        flags=re.IGNORECASE,
    )

    source_map = {}

    for i, source in enumerate(
        sources,
        start=1,
    ):
        metadata = source["metadata"]

        source_map[i] = (
            f"[{metadata['course']}, "
            f"PDF page "
            f"{metadata['pdf_page']}]"
        )

    def replace_marker(match):
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

    answer = deduplicate_citation_blocks(
        answer
    )

    answer = remove_conflicting_refusal(
        answer
    )

    answer = re.sub(
        r"[ \t]{2,}",
        " ",
        answer,
    )

    return answer.strip()


# ============================================================
# GENERATION
# ============================================================

def generate_answer(
    question,
    standalone_question,
    query_type,
    sources,
    conversation_history,
    llm,
):
    if (
        query_type == "broad"
        and not sources
    ):
        return REFUSAL_MESSAGE

    document_context = build_context(
        sources
    )

    conversation_text = format_conversation_history(
        conversation_history
    )

    if query_type == "broad":
        retrieval_rules = """
This is a BROAD portfolio question.

A separate evidence relevance check already removed courses that
do not substantively discuss the requested topic.

Discuss ONLY courses present in the retrieved context.
Do not add indirect relationships.
Do not infer another course might be related.
Do not append the refusal after a supported answer.
Use [SOURCE N] after supported factual statements.
"""
    else:
        retrieval_rules = """
This is a FOCUSED question.

Answer directly using the strongest retrieved evidence.
Use [SOURCE N] after supported factual statements.
Do not append the refusal after a supported answer.
"""

    system_message = f"""
You are a conversational question-answering assistant for an MBA
academic portfolio.

Conversation history helps interpret follow-up questions, but
factual answers must be supported by newly retrieved document
context.

{retrieval_rules}

CITATION RULES:
1. Never create course/page citations yourself.
2. Never write citations such as [C200, PDF page 10].
3. Cite evidence only with [SOURCE N] markers.
4. Use only SOURCE numbers present in the context.
5. Python converts source markers to trusted course/page metadata.

REFUSAL RULE:
Use this sentence ONLY when no retrieved evidence can answer:

{REFUSAL_MESSAGE}

GROUNDING RULES:
1. Do not use outside knowledge.
2. Do not invent facts.
3. Use retrieved document context for factual claims.
4. Answer the current user question.
"""

    user_message = f"""
CONVERSATION HISTORY:

{conversation_text}

CURRENT USER QUESTION:

{question}

STANDALONE RETRIEVAL QUESTION:

{standalone_question}

RETRIEVED DOCUMENT CONTEXT:

{document_context}

Answer the current user question:
"""

    response = llm.invoke(
        [
            ("system", system_message),
            ("human", user_message),
        ]
    )

    return replace_source_markers(
        response.content.strip(),
        sources,
    )


# ============================================================
# COMPLETE CONVERSATIONAL RAG
# ============================================================

def ask_rag(
    question,
    conversation_history,
    collection,
    embedding_model,
    reranker,
    rag_chunk_lookup,
    llm,
):
    standalone_question = contextualize_question(
        question,
        conversation_history,
        llm,
    )

    query_type, sources = retrieve_for_question(
        standalone_question,
        collection,
        embedding_model,
        reranker,
        rag_chunk_lookup,
        llm,
    )

    answer = generate_answer(
        question,
        standalone_question,
        query_type,
        sources,
        conversation_history,
        llm,
    )

    return (
        answer,
        sources,
        query_type,
        standalone_question,
    )


def initialize_rag():
    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    reranker = CrossEncoder(
        RERANKER_MODEL
    )

    llm = ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0,
    )

    rag_chunks = load_and_chunk_pdf()

    rag_chunk_lookup = {
        chunk.metadata["rag_chunk_number"]: chunk
        for chunk in rag_chunks
    }

    collection = build_vector_database(
        rag_chunks,
        embedding_model,
    )

    return {
        "embedding_model": embedding_model,
        "reranker": reranker,
        "llm": llm,
        "rag_chunks": rag_chunks,
        "rag_chunk_lookup": rag_chunk_lookup,
        "collection": collection,
    }


# ============================================================
# CLI CONVERSATION
# ============================================================

def main():
    print("\nLoading RAG V11...")

    rag = initialize_rag()

    print("\nRAG V11 ready.")
    print("Type 'exit' to stop.")
    print("Type 'clear' to erase conversation memory.")

    conversation_history = []

    while True:
        question = input(
            "\nYou: "
        ).strip()

        if not question:
            continue

        if question.lower() == "exit":
            break

        if question.lower() == "clear":
            conversation_history = []
            print("Conversation memory cleared.")
            continue

        (
            answer,
            sources,
            query_type,
            standalone_question,
        ) = ask_rag(
            question,
            conversation_history,
            rag["collection"],
            rag["embedding_model"],
            rag["reranker"],
            rag["rag_chunk_lookup"],
            rag["llm"],
        )

        print(
            "\nStandalone retrieval question:"
        )
        print(standalone_question)

        print(
            "\nRetrieval mode:",
            query_type,
        )

        print("\nAssistant:")
        print(answer)

        conversation_history.append(
            {
                "role": "user",
                "content": question,
            }
        )

        conversation_history.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )


if __name__ == "__main__":
    main()
