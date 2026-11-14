from rag_pipeline import (
    COURSE_CODES,
    REFUSAL_MESSAGE,
    detect_query_type,
    initialize_rag,
    ask_rag,
)


# ============================================================
# RAG V9 REGRESSION EVALUATION
#
# This harness checks:
# - focused retrieval
# - conversational follow-up rewriting
# - deterministic citations
# - broad retrieval
# - broad course relevance filtering
# - grounded refusal
# ============================================================


def print_test(name, passed):
    print(
        f"{name}: "
        f"{'PASS ✅' if passed else 'FAIL ❌'}"
    )


rag = initialize_rag()

conversation_history = []


def run_turn(question):
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

    return (
        answer,
        sources,
        query_type,
        standalone_question,
    )


print(
    "\n"
    + "=" * 70
)

print(
    "V9 REGRESSION EVALUATION"
)

print(
    "=" * 70
)


# ============================================================
# TEST 1 - CLIFTONSTRENGTHS
# ============================================================

answer_1, sources_1, type_1, search_1 = run_turn(
    "What are my five CliftonStrengths?"
)

expected_strengths = [
    "learner",
    "strategic",
    "belief",
    "context",
    "deliberative",
]

strengths_pass = all(
    strength in answer_1.lower()
    for strength in expected_strengths
)


# ============================================================
# TEST 2 - MEMORY FOLLOW-UP
# ============================================================

answer_2, sources_2, type_2, search_2 = run_turn(
    "Which one was ranked highest?"
)

memory_rewrite_1_pass = (
    "clifton"
    in search_2.lower()
    or "strength"
    in search_2.lower()
)

memory_answer_1_pass = (
    "learner"
    in answer_2.lower()
)


# ============================================================
# TEST 3 - AGENTIC RAG
# ============================================================

answer_3, sources_3, type_3, search_3 = run_turn(
    "What technologies were proposed "
    "for the Agentic RAG architecture?"
)

agentic_pass = (
    "microsoft foundry"
    in answer_3.lower()
    and "azure ai search"
    in answer_3.lower()
)

citation_pass = (
    "[MMT2-BEN1, PDF page 115]"
    in answer_3
    and "[C200, PDF page 115]"
    not in answer_3
)


# ============================================================
# TEST 4 - SECOND MEMORY FOLLOW-UP
# ============================================================

answer_4, sources_4, type_4, search_4 = run_turn(
    "Which one provides the search and retrieval layer?"
)

memory_rewrite_2_pass = (
    "agentic"
    in search_4.lower()
    or "azure ai search"
    in search_4.lower()
    or "foundry"
    in search_4.lower()
)

memory_answer_2_pass = (
    "azure ai search"
    in answer_4.lower()
)


# ============================================================
# TEST 5 - BROAD QUERY + RELEVANCE GATE
# ============================================================

answer_5, sources_5, type_5, search_5 = run_turn(
    "Identify all courses in my MBA portfolio "
    "that discuss project management."
)

broad_router_pass = (
    type_5 == "broad"
)

project_management_pass = (
    "MGT2-KKM1"
    in answer_5
)

c200_false_positive_removed = (
    "C200"
    not in answer_5
)

filtered_courses = {
    source["metadata"]["course"]
    for source in sources_5
}

relevance_gate_pass = (
    "MGT2-KKM1"
    in filtered_courses
    and "C200"
    not in filtered_courses
)


# ============================================================
# TEST 6 - GROUNDED REFUSAL
# ============================================================

answer_6, sources_6, type_6, search_6 = run_turn(
    "What is Microsoft's current stock price?"
)

refusal_pass = (
    answer_6.strip()
    == REFUSAL_MESSAGE
)


# ============================================================
# RESULTS
# ============================================================

checks = {
    "Five CliftonStrengths":
        strengths_pass,

    "Follow-up 1 rewritten with memory":
        memory_rewrite_1_pass,

    "Follow-up 1 answer = Learner":
        memory_answer_1_pass,

    "Agentic RAG technologies":
        agentic_pass,

    "Agentic RAG citation grounded":
        citation_pass,

    "Follow-up 2 rewritten with memory":
        memory_rewrite_2_pass,

    "Follow-up 2 answer = Azure AI Search":
        memory_answer_2_pass,

    "Broad router":
        broad_router_pass,

    "Project Management course found":
        project_management_pass,

    "C200 false positive removed":
        c200_false_positive_removed,

    "Course relevance gate":
        relevance_gate_pass,

    "Grounded refusal":
        refusal_pass,
}


for name, passed in checks.items():
    print_test(
        name,
        passed,
    )


passed_count = sum(
    checks.values()
)

total_count = len(
    checks
)

accuracy = (
    passed_count
    / total_count
    * 100
)


print(
    "\n"
    + "=" * 70
)

print(
    "V9 FINAL RESULT"
)

print(
    "=" * 70
)

print(
    "Tests passed:",
    f"{passed_count}/{total_count}",
)

print(
    "Accuracy:",
    f"{accuracy:.1f}%",
)


if passed_count == total_count:
    print(
        "V9 REGRESSION TEST: PASS ✅"
    )
else:
    print(
        "V9 REGRESSION TEST: NEEDS REVIEW ⚠️"
    )
