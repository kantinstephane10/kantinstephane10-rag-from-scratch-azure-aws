from pathlib import Path

from memory_db import (
    initialize_database,
    create_new_chat,
    list_conversations,
    get_conversation,
    conversation_exists,
    save_message,
    load_messages,
    load_recent_memory,
    count_messages,
    rename_conversation,
    clear_conversation,
    delete_conversation,
    auto_title_conversation,
)


# ============================================================
# RAG V10 - SQLITE MEMORY REGRESSION TEST
# ============================================================


TEST_DB = Path(
    "data/memory/rag_v10_repo_test.db"
)


if TEST_DB.exists():
    TEST_DB.unlink()


checks = {}


def check(
    name,
    condition,
):
    checks[name] = bool(condition)

    print(
        f"{name}: "
        f"{'PASS ✅' if condition else 'FAIL ❌'}"
    )


# ============================================================
# DATABASE + CONVERSATION
# ============================================================

initialize_database(
    TEST_DB
)

check(
    "Database created",
    TEST_DB.exists(),
)


conversation_id = create_new_chat(
    TEST_DB
)

check(
    "Conversation created",
    conversation_exists(
        conversation_id,
        TEST_DB,
    ),
)

check(
    "Conversation starts empty",
    count_messages(
        conversation_id,
        TEST_DB,
    )
    == 0,
)


# ============================================================
# SAVE + RELOAD
# ============================================================

save_message(
    conversation_id=
        conversation_id,
    role="user",
    content=(
        "What are my five "
        "CliftonStrengths?"
    ),
    db_path=TEST_DB,
)


sample_sources = [
    {
        "document":
            "Sample retrieved evidence",

        "metadata": {
            "course": "C200",
            "chapter":
                "Managing Organizations and Leading People",
            "pdf_page": 10,
        },

        "retrieval_source":
            "semantic",

        "reranker_score":
            7.5,
    }
]


save_message(
    conversation_id=
        conversation_id,
    role="assistant",
    content=(
        "Learner, Strategic, Belief, "
        "Context, and Deliberative."
    ),
    query_type="focused",
    standalone_question=(
        "What are my five "
        "CliftonStrengths?"
    ),
    sources=sample_sources,
    db_path=TEST_DB,
)


messages = load_messages(
    conversation_id,
    TEST_DB,
)


check(
    "Two messages persisted",
    len(messages) == 2,
)

check(
    "User message restored",
    (
        messages[0]["role"] == "user"
        and
        "CliftonStrengths"
        in messages[0]["content"]
    ),
)

check(
    "Assistant message restored",
    (
        messages[1]["role"]
        == "assistant"
        and
        "Learner"
        in messages[1]["content"]
    ),
)

check(
    "Query metadata restored",
    messages[1]["query_type"]
    == "focused",
)

check(
    "Standalone question restored",
    (
        "CliftonStrengths"
        in messages[1][
            "standalone_question"
        ]
    ),
)

check(
    "Retrieved source restored",
    (
        len(
            messages[1]["sources"]
        )
        == 1
        and
        messages[1][
            "sources"
        ][0]["metadata"]["course"]
        == "C200"
    ),
)


# ============================================================
# AUTO TITLE
# ============================================================

auto_title_conversation(
    conversation_id,
    (
        "What are my five "
        "CliftonStrengths?"
    ),
    TEST_DB,
)


conversation = get_conversation(
    conversation_id,
    TEST_DB,
)


check(
    "Auto title created",
    (
        conversation["title"]
        !=
        "New conversation"
    ),
)


# ============================================================
# RECENT MEMORY
# ============================================================

recent = load_recent_memory(
    conversation_id,
    TEST_DB,
)


check(
    "Recent memory loads",
    len(recent) == 2,
)


# ============================================================
# SECOND CONVERSATION
# ============================================================

conversation_2 = create_new_chat(
    TEST_DB
)


check(
    "Second conversation created",
    conversation_exists(
        conversation_2,
        TEST_DB,
    ),
)


check(
    "Conversations are independent",
    count_messages(
        conversation_2,
        TEST_DB,
    )
    == 0,
)


check(
    "Both conversations listed",
    len(
        list_conversations(
            TEST_DB
        )
    )
    == 2,
)


# ============================================================
# RENAME
# ============================================================

rename_conversation(
    conversation_2,
    "Agentic RAG Discussion",
    TEST_DB,
)


check(
    "Conversation renamed",
    (
        get_conversation(
            conversation_2,
            TEST_DB,
        )["title"]
        ==
        "Agentic RAG Discussion"
    ),
)


# ============================================================
# CLEAR
# ============================================================

clear_conversation(
    conversation_id,
    TEST_DB,
)


check(
    "Clear removes messages",
    count_messages(
        conversation_id,
        TEST_DB,
    )
    == 0,
)


check(
    "Clear keeps conversation",
    conversation_exists(
        conversation_id,
        TEST_DB,
    ),
)


# ============================================================
# DELETE
# ============================================================

delete_conversation(
    conversation_2,
    TEST_DB,
)


check(
    "Delete removes conversation",
    not conversation_exists(
        conversation_2,
        TEST_DB,
    ),
)


# ============================================================
# RESULT
# ============================================================

passed_count = sum(
    checks.values()
)

total_count = len(
    checks
)

accuracy = (
    passed_count
    /
    total_count
    *
    100
)


print(
    "\n"
    + "=" * 70
)

print(
    "RAG V10 REPO MEMORY TEST"
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
        "V10 REPO MEMORY TEST: PASS ✅"
    )
else:
    print(
        "V10 REPO MEMORY TEST: "
        "NEEDS REVIEW ⚠️"
    )


if TEST_DB.exists():
    TEST_DB.unlink()
