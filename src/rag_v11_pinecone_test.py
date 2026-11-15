import os

from rag_pipeline import (
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
    PINECONE_VECTOR_FIELD,
    PINECONE_DIMENSION,
    get_pinecone_client,
    initialize_rag,
    embed_question,
    pinecone_search,
    retrieve_focused,
)


# ============================================================
# RAG V11 - PINECONE VECTOR DATABASE TEST
# ============================================================


checks = {}


def check(
    name,
    condition,
):
    checks[name] = bool(
        condition
    )

    print(
        f"{name}: "
        f"{'PASS ✅' if condition else 'FAIL ❌'}"
    )


# ============================================================
# API KEY
# ============================================================

api_key_present = bool(
    os.getenv(
        "PINECONE_API_KEY"
    )
)


check(
    "Pinecone API key available",
    api_key_present,
)


if not api_key_present:
    raise RuntimeError(
        "PINECONE_API_KEY is not set. "
        "Set it in PowerShell before running this test."
    )


# ============================================================
# INITIALIZE COMPLETE RAG
#
# This:
# - loads the PDF
# - creates MiniLM embeddings
# - creates/reuses the Pinecone index
# - upserts the 351 chunks
# - waits until Pinecone can read the documents
# ============================================================

rag = initialize_rag()


# ============================================================
# CONTROL-PLANE CHECKS
# ============================================================

pc = get_pinecone_client()


check(
    "Pinecone index exists",
    pc.indexes.exists(
        PINECONE_INDEX_NAME
    ),
)


index_info = (
    pc.indexes.describe(
        PINECONE_INDEX_NAME
    )
)


fields = (
    index_info
    .schema
    .fields
)


check(
    "Dense vector field exists",
    (
        PINECONE_VECTOR_FIELD
        in fields
    ),
)


vector_field = fields[
    PINECONE_VECTOR_FIELD
]


check(
    "Pinecone dimension = 384",
    (
        int(
            vector_field.dimension
        )
        ==
        PINECONE_DIMENSION
    ),
)


check(
    "Pinecone metric = cosine",
    (
        str(
            vector_field.metric
        ).lower()
        ==
        "cosine"
    ),
)


# ============================================================
# FETCH STORED DOCUMENTS
# ============================================================

index = rag[
    "collection"
]


sample_ids = [
    "mba_chunk_0000",
    "mba_chunk_0350",
]


fetch_response = (
    index.documents.fetch(
        namespace=
            PINECONE_NAMESPACE,

        ids=
            sample_ids,

        include_fields=[
            "text",
            "source",
            "pdf_page",
            "course",
            "chapter",
            "rag_chunk_number",
        ],
    )
)


check(
    "Pinecone documents persisted",
    (
        len(
            fetch_response.documents
        )
        ==
        2
    ),
)


# ============================================================
# DIRECT SEMANTIC SEARCH
# ============================================================

question = (
    "What are my five "
    "CliftonStrengths?"
)


question_embedding = (
    embed_question(
        question,
        rag[
            "embedding_model"
        ],
    )
)


direct_results = (
    pinecone_search(
        index,
        question_embedding[
            0
        ].tolist(),
        top_k=8,
    )
)


check(
    "Pinecone semantic search returns results",
    (
        len(
            direct_results
        )
        > 0
    ),
)


required_metadata = {
    "source",
    "pdf_page",
    "course",
    "chapter",
    "rag_chunk_number",
}


check(
    "Pinecone metadata restored",
    all(
        required_metadata.issubset(
            result[
                "metadata"
            ].keys()
        )

        for result
        in direct_results
    ),
)


check(
    "Pinecone similarity score returned",
    all(
        "vector_score"
        in result

        for result
        in direct_results
    ),
)


# ============================================================
# METADATA FILTER TEST
# ============================================================

filtered_results = (
    pinecone_search(
        index,
        question_embedding[
            0
        ].tolist(),
        top_k=4,
        course="C200",
    )
)


check(
    "Pinecone metadata filter returns results",
    (
        len(
            filtered_results
        )
        > 0
    ),
)


check(
    "Pinecone metadata filter stays in C200",
    all(
        result[
            "metadata"
        ][
            "course"
        ]
        ==
        "C200"

        for result
        in filtered_results
    ),
)


# ============================================================
# FULL FOCUSED RETRIEVAL TEST
#
# Pinecone -> Neighbor Expansion -> CrossEncoder -> Best 3
# ============================================================

focused_results = (
    retrieve_focused(
        question,
        index,
        rag[
            "embedding_model"
        ],
        rag[
            "reranker"
        ],
        rag[
            "rag_chunk_lookup"
        ],
    )
)


check(
    "Focused retrieval returns Best 3",
    (
        len(
            focused_results
        )
        ==
        3
    ),
)


check(
    "Focused retrieval includes C200 page 10",
    any(
        (
            result[
                "metadata"
            ][
                "course"
            ]
            ==
            "C200"
        )
        and
        (
            result[
                "metadata"
            ][
                "pdf_page"
            ]
            ==
            10
        )

        for result
        in focused_results
    ),
)


check(
    "CrossEncoder reranking still works",
    all(
        "reranker_score"
        in result

        for result
        in focused_results
    ),
)


# ============================================================
# FINAL RESULT
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
    "RAG V11 PINECONE VECTOR DATABASE TEST"
)


print(
    "=" * 70
)


print(
    "Index:",
    PINECONE_INDEX_NAME,
)


print(
    "Namespace:",
    PINECONE_NAMESPACE,
)


print(
    "Tests passed:",
    f"{passed_count}/{total_count}",
)


print(
    "Accuracy:",
    f"{accuracy:.1f}%",
)


if (
    passed_count
    ==
    total_count
):

    print(
        "V11 PINECONE VECTOR DATABASE: PASS ✅"
    )

else:

    print(
        "V11 PINECONE VECTOR DATABASE: "
        "NEEDS REVIEW ⚠️"
    )
