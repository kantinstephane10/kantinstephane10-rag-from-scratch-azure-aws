import re

from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_ollama import ChatOllama

from rag_pipeline import (
    EMBEDDING_MODEL,
    RERANKER_MODEL,
    OLLAMA_MODEL,
    load_and_chunk_pdf,
    build_vector_database,
    ask_rag,
)


# ============================================================
# RAG V6 - END-TO-END ANSWER EVALUATION
#
# Evaluates:
# 1. Answer correctness
# 2. Correct source retrieval
# 3. Citation validity
# 4. Grounded refusal
# ============================================================


def normalize_for_match(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def check_answer_evidence(answer, evidence_groups):
    normalized_answer = normalize_for_match(answer)

    matched_groups = []
    missing_groups = []

    for group_number, alternatives in enumerate(
        evidence_groups,
        start=1,
    ):
        found = False

        for alternative in alternatives:
            if normalize_for_match(
                alternative
            ) in normalized_answer:
                found = True
                break

        if found:
            matched_groups.append(group_number)
        else:
            missing_groups.append(group_number)

    return (
        len(missing_groups) == 0,
        matched_groups,
        missing_groups,
    )


def extract_citations(answer):
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
        for course, page in matches
    ]


def validate_citations(answer, sources):
    citations = extract_citations(answer)

    source_pairs = {
        (
            source["metadata"]["course"].upper(),
            int(source["metadata"]["pdf_page"]),
        )
        for source in sources
    }

    valid_citations = [
        citation
        for citation in citations
        if citation in source_pairs
    ]

    return (
        len(valid_citations) > 0,
        citations,
        valid_citations,
    )


def check_refusal(answer):
    expected_text = (
        "i don't have enough information "
        "in the provided document to answer that"
    )

    return expected_text in normalize_for_match(
        answer
    )


evaluation_questions = [
    {
        "question": "What are my five CliftonStrengths?",
        "expected_course": "C200",
        "expected_evidence": [
            ["learner"],
            ["strategic"],
            ["belief"],
            ["context"],
            ["deliberative"],
        ],
        "answerable": True,
    },
    {
        "question": "What two ethical traits did I identify in Satya Nadella?",
        "expected_course": "EHM2",
        "expected_evidence": [
            ["empathy"],
            ["accountability"],
        ],
        "answerable": True,
    },
    {
        "question": "Which country did I recommend as the first international market for Renew Vision Signature?",
        "expected_course": "C212-KLM2",
        "expected_evidence": [
            ["canada"],
        ],
        "answerable": True,
    },
    {
        "question": "What was the main schedule concern identified by the earned value analysis?",
        "expected_course": "MGT2-KKM1",
        "expected_evidence": [
            [
                "behind schedule",
                "schedule performance",
            ],
            [
                "spi",
                "0.71",
            ],
        ],
        "answerable": True,
    },
    {
        "question": "What emerging technology solution did I propose for cloud operations?",
        "expected_course": "MMT2-BEN1",
        "expected_evidence": [
            [
                "agentic rag knowledge assistant",
                "agentic retrieval-augmented generation",
            ],
            ["cloud operations"],
        ],
        "answerable": True,
    },
    {
        "question": "What technologies were proposed for the Agentic RAG architecture?",
        "expected_course": "MMT2-BEN1",
        "expected_evidence": [
            [
                "microsoft foundry agent service",
                "microsoft foundry",
            ],
            ["azure ai search"],
        ],
        "answerable": True,
    },
    {
        "question": "What relationship was examined in the linear regression analysis?",
        "expected_course": "C207-QUM4",
        "expected_evidence": [
            ["nurse"],
            ["participation"],
            ["attrition"],
        ],
        "answerable": True,
    },
    {
        "question": "Which three strategies were compared in the pharmaceutical decision tree?",
        "expected_course": "C207-QUM4",
        "expected_evidence": [
            [
                "new drug",
                "develop a new drug",
            ],
            [
                "modify",
                "existing drug",
            ],
            [
                "no changes",
                "current drug",
            ],
        ],
        "answerable": True,
    },
    {
        "question": "What is Microsoft's current stock price?",
        "expected_course": None,
        "expected_evidence": [],
        "answerable": False,
    },
]


def main():
    print("\nLoading embedding model...")

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    print("Embedding model loaded")

    print("\nLoading reranker...")

    reranker = CrossEncoder(
        RERANKER_MODEL
    )

    print("Reranker loaded")

    print("\nConnecting to Ollama...")

    llm = ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0,
    )

    print("Ollama connected")

    rag_chunks = load_and_chunk_pdf()

    rag_chunk_lookup = {
        chunk.metadata["rag_chunk_number"]: chunk
        for chunk in rag_chunks
    }

    collection = build_vector_database(
        rag_chunks,
        embedding_model,
    )

    print("\n\n" + "=" * 70)
    print("RAG V6 - END-TO-END ANSWER EVALUATION")
    print("=" * 70)

    evaluation_results = []

    for number, test in enumerate(
        evaluation_questions,
        start=1,
    ):
        question = test["question"]

        print("\n")
        print("-" * 70)
        print(f"QUESTION {number}")
        print(question)

        answer, sources = ask_rag(
            question,
            collection,
            embedding_model,
            reranker,
            rag_chunk_lookup,
            llm,
        )

        print("\nOLLAMA ANSWER:")
        print(answer)

        if test["answerable"]:
            (
                evidence_pass,
                matched_groups,
                missing_groups,
            ) = check_answer_evidence(
                answer,
                test["expected_evidence"],
            )

            (
                citation_pass,
                citations,
                valid_citations,
            ) = validate_citations(
                answer,
                sources,
            )

            source_courses = [
                source["metadata"]["course"]
                for source in sources
            ]

            source_pass = (
                test["expected_course"]
                in source_courses
            )

            passed = (
                evidence_pass
                and citation_pass
                and source_pass
            )

            result = {
                "question": question,
                "answer": answer,
                "answerable": True,
                "evidence_pass": evidence_pass,
                "citation_pass": citation_pass,
                "source_pass": source_pass,
                "matched_groups": matched_groups,
                "missing_groups": missing_groups,
                "citations": citations,
                "valid_citations": valid_citations,
                "passed": passed,
            }

            print(
                "\nAnswer content correct:",
                evidence_pass,
            )
            print(
                "Correct course retrieved:",
                source_pass,
            )
            print(
                "Valid source citation:",
                citation_pass,
            )
            print(
                "Citations found:",
                citations,
            )
            print(
                "Validated citations:",
                valid_citations,
            )

        else:
            refusal_pass = check_refusal(
                answer
            )

            passed = refusal_pass

            result = {
                "question": question,
                "answer": answer,
                "answerable": False,
                "refusal_pass": refusal_pass,
                "passed": passed,
            }

            print(
                "\nCorrect refusal:",
                refusal_pass,
            )

        print(
            "RESULT: PASS ✅"
            if passed
            else "RESULT: FAIL ❌"
        )

        evaluation_results.append(
            result
        )

    total_tests = len(
        evaluation_results
    )

    total_passed = sum(
        result["passed"]
        for result in evaluation_results
    )

    answerable_results = [
        result
        for result in evaluation_results
        if result["answerable"]
    ]

    answerable_count = len(
        answerable_results
    )

    answer_correct_count = sum(
        result["evidence_pass"]
        for result in answerable_results
    )

    citation_correct_count = sum(
        result["citation_pass"]
        for result in answerable_results
    )

    source_correct_count = sum(
        result["source_pass"]
        for result in answerable_results
    )

    refusal_results = [
        result
        for result in evaluation_results
        if not result["answerable"]
    ]

    refusal_count = len(
        refusal_results
    )

    refusal_pass_count = sum(
        result["refusal_pass"]
        for result in refusal_results
    )

    print("\n\n" + "=" * 70)
    print("FINAL END-TO-END RAG RESULTS")
    print("=" * 70)

    print(
        "Total tests:",
        total_tests,
    )

    print(
        "Total passed:",
        total_passed,
    )

    print("\nAnswer correctness:")
    print(
        f"{answer_correct_count}/{answerable_count} "
        f"({answer_correct_count / answerable_count * 100:.1f}%)"
    )

    print("\nCorrect source retrieval:")
    print(
        f"{source_correct_count}/{answerable_count} "
        f"({source_correct_count / answerable_count * 100:.1f}%)"
    )

    print("\nValid citation accuracy:")
    print(
        f"{citation_correct_count}/{answerable_count} "
        f"({citation_correct_count / answerable_count * 100:.1f}%)"
    )

    if refusal_count > 0:
        print("\nGrounded refusal accuracy:")
        print(
            f"{refusal_pass_count}/{refusal_count} "
            f"({refusal_pass_count / refusal_count * 100:.1f}%)"
        )

    print("\nOverall end-to-end accuracy:")
    print(
        f"{total_passed}/{total_tests} "
        f"({total_passed / total_tests * 100:.1f}%)"
    )

    failed_results = [
        result
        for result in evaluation_results
        if not result["passed"]
    ]

    print("\n" + "=" * 70)
    print("FAILED END-TO-END TESTS")
    print("=" * 70)

    if not failed_results:
        print("No failed tests. ✅")
    else:
        for result in failed_results:
            print("\nQuestion:")
            print(result["question"])
            print("\nAnswer:")
            print(result["answer"])


if __name__ == "__main__":
    main()
