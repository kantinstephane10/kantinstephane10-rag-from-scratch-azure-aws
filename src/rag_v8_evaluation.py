import re

from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_ollama import ChatOllama

from rag_pipeline import (
    EMBEDDING_MODEL,
    RERANKER_MODEL,
    OLLAMA_MODEL,
    COURSE_CODES,
    detect_query_type,
    replace_source_markers,
    load_and_chunk_pdf,
    build_vector_database,
    ask_rag,
)


# ============================================================
# RAG V8 REGRESSION EVALUATION
#
# Tests the two issues found during V7:
#
# 1. Broad questions must use course-aware retrieval.
# 2. Citations must come from Python metadata, not LLM guesses.
# ============================================================


def extract_citations(
    answer,
):

    pattern = (
        r"\["
        r"([^,\]]+)"
        r",\s*PDF page\s*"
        r"(\d+)"
        r"\]"
    )

    matches = re.findall(
        pattern,
        answer,
        flags=re.IGNORECASE,
    )

    return [
        (
            course.strip().upper(),
            int(page),
        )
        for course, page
        in matches
    ]


def citations_are_grounded(
    answer,
    sources,
):

    citations = extract_citations(
        answer
    )

    valid_pairs = {
        (
            source["metadata"][
                "course"
            ].upper(),
            int(
                source["metadata"][
                    "pdf_page"
                ]
            ),
        )
        for source
        in sources
    }

    return all(
        citation in valid_pairs
        for citation in citations
    )


def print_result(
    name,
    passed,
):

    print(
        f"{name}: "
        f"{'PASS ✅' if passed else 'FAIL ❌'}"
    )


print(
    "\n"
    + "=" * 70
)

print(
    "V8 FAST UNIT TESTS"
)

print(
    "=" * 70
)


focused_question = (
    "What are my five CliftonStrengths?"
)

broad_question = (
    "Identify all courses in my MBA portfolio "
    "that discuss project management."
)


focused_route_pass = (
    detect_query_type(
        focused_question
    )
    == "focused"
)


broad_route_pass = (
    detect_query_type(
        broad_question
    )
    == "broad"
)


fake_sources = [
    {
        "metadata": {
            "course":
                "MMT2-BEN1",

            "pdf_page":
                115,
        }
    }
]


fake_answer = (
    "The architecture uses Microsoft Foundry "
    "and Azure AI Search. "
    "[C200, PDF page 115] "
    "[SOURCE 1]"
)


fixed_answer = replace_source_markers(
    fake_answer,
    fake_sources,
)


citation_replacement_pass = (
    "[MMT2-BEN1, PDF page 115]"
    in fixed_answer
    and
    "[C200, PDF page 115]"
    not in fixed_answer
)


print_result(
    "Focused query routing",
    focused_route_pass,
)

print_result(
    "Broad query routing",
    broad_route_pass,
)

print_result(
    "Deterministic citation replacement",
    citation_replacement_pass,
)


print(
    "\nLoading embedding model..."
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
    "\n"
    + "=" * 70
)

print(
    "TEST 1 - AGENTIC RAG CITATION"
)

print(
    "=" * 70
)


question_1 = (
    "What technologies were proposed "
    "for the Agentic RAG architecture?"
)


answer_1, sources_1 = ask_rag(
    question_1,
    collection,
    embedding_model,
    reranker,
    rag_chunk_lookup,
    llm,
)


print(
    "\nAnswer:"
)

print(
    answer_1
)


facts_pass = (
    "microsoft foundry"
    in answer_1.lower()
    and
    "azure ai search"
    in answer_1.lower()
)


correct_citation_pass = (
    "[MMT2-BEN1, PDF page 115]"
    in answer_1
)


old_wrong_citation_removed = (
    "[C200, PDF page 115]"
    not in answer_1
)


citation_grounding_pass = (
    citations_are_grounded(
        answer_1,
        sources_1,
    )
)


print_result(
    "Expected technologies",
    facts_pass,
)

print_result(
    "Correct MMT2-BEN1 citation",
    correct_citation_pass,
)

print_result(
    "Old C200 citation removed",
    old_wrong_citation_removed,
)

print_result(
    "All citations exist in retrieved metadata",
    citation_grounding_pass,
)


print(
    "\n"
    + "=" * 70
)

print(
    "TEST 2 - BROAD PORTFOLIO RETRIEVAL"
)

print(
    "=" * 70
)


question_2 = (
    "Identify all courses in my MBA portfolio "
    "that discuss project management. "
    "For each course, provide the course code, "
    "course title, how project management is discussed, "
    "and the PDF page number."
)


answer_2, sources_2 = ask_rag(
    question_2,
    collection,
    embedding_model,
    reranker,
    rag_chunk_lookup,
    llm,
)


print(
    "\nAnswer:"
)

print(
    answer_2
)


retrieved_courses = {
    source["metadata"][
        "course"
    ]
    for source
    in sources_2
}


all_courses_covered_pass = (
    retrieved_courses
    == set(
        COURSE_CODES
    )
)


project_management_course_pass = (
    "MGT2-KKM1"
    in answer_2
)


broad_citation_grounding_pass = (
    citations_are_grounded(
        answer_2,
        sources_2,
    )
)


print(
    "\nCourses represented in broad retrieval:"
)

for course in COURSE_CODES:

    print(
        "-",
        course,
        "✅"
        if course in retrieved_courses
        else "❌",
    )


print_result(
    "All six courses searched",
    all_courses_covered_pass,
)

print_result(
    "Project Management course identified",
    project_management_course_pass,
)

print_result(
    "Broad-answer citations grounded",
    broad_citation_grounding_pass,
)


print(
    "\n"
    + "=" * 70
)

print(
    "TEST 3 - GROUNDED REFUSAL"
)

print(
    "=" * 70
)


question_3 = (
    "What is Microsoft's current stock price?"
)


answer_3, sources_3 = ask_rag(
    question_3,
    collection,
    embedding_model,
    reranker,
    rag_chunk_lookup,
    llm,
)


print(
    "\nAnswer:"
)

print(
    answer_3
)


refusal_pass = (
    "i don't have enough information "
    "in the provided document to answer that"
    in answer_3.lower()
)


print_result(
    "Grounded refusal",
    refusal_pass,
)


all_tests = [
    focused_route_pass,
    broad_route_pass,
    citation_replacement_pass,
    facts_pass,
    correct_citation_pass,
    old_wrong_citation_removed,
    citation_grounding_pass,
    all_courses_covered_pass,
    project_management_course_pass,
    broad_citation_grounding_pass,
    refusal_pass,
]


passed = sum(
    all_tests
)


total = len(
    all_tests
)


print(
    "\n"
    + "=" * 70
)

print(
    "V8 FINAL RESULT"
)

print(
    "=" * 70
)


print(
    f"Tests passed: "
    f"{passed}/{total}"
)


print(
    f"Accuracy: "
    f"{passed / total * 100:.1f}%"
)


if passed == total:

    print(
        "V8 regression evaluation: PASS ✅"
    )

else:

    print(
        "V8 regression evaluation: "
        "NEEDS REVIEW ⚠️"
    )
