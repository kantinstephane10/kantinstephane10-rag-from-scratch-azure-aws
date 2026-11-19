# ============================================================
# RAG V13 - PRODUCTION BACKEND
#
# Generated from the validated V13 Jupyter implementation.
#
# Architecture:
#
# Dense Pinecone retrieval
# + BM25 lexical retrieval
# + Reciprocal Rank Fusion
# + document-safe neighbor expansion
# + evidence filtering
# + CrossEncoder reranking
# + provider-balanced comparison retrieval
# + Ollama grounded generation
# + deterministic citation validation
# + SQLite conversational memory
# ============================================================

from pathlib import Path

import json
import os
import pickle
import re
import time

import requests

from pinecone import (
    Pinecone,
    DenseVectorQuery,
)

from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder,
)

from memory_db import (
    initialize_database,
    create_new_chat,
    get_conversation,
    auto_title_conversation,
    load_messages,
    load_recent_memory,
    save_message,
    count_messages,
)


# ============================================================
# REPOSITORY PATHS
# ============================================================

SRC_ROOT = Path(__file__).resolve().parent

REPO_ROOT = SRC_ROOT.parent

DATA_ROOT = (
    REPO_ROOT
    / "data"
)

CACHE_ROOT = (
    DATA_ROOT
    / "cache"
)

MEMORY_ROOT = (
    DATA_ROOT
    / "memory"
)


MEMORY_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


BACKEND_CACHE = (
    CACHE_ROOT
    / "v13_backend_state.pkl"
)


DEFAULT_MEMORY_DB = (
    MEMORY_ROOT
    / "rag_v13_streamlit.db"
)


# ============================================================
# V13 CONFIGURATION
# ============================================================

PINECONE_INDEX_NAME = 'multicloud-rag-v12'

PINECONE_NAMESPACE = 'cloud-documents'

PINECONE_VECTOR_FIELD = 'embedding'

PINECONE_DIMENSION = 384


OLLAMA_BASE_URL = 'http://localhost:11434'

OLLAMA_MODEL = 'qwen3:4b-instruct'


DENSE_TOP_K = 20

BM25_TOP_K = 20

RRF_K = 60

FUSION_TOP_K = 25

NEIGHBOR_RANGE = 1

FINAL_TOP_K = 5

MIN_EVIDENCE_CHARS = 120


COMPARISON_DENSE_TOP_K = 30

COMPARISON_BM25_TOP_K = 30

COMPARISON_FUSION_TOP_K = 40

COMPARISON_FINAL_TOP_K = 5


MEMORY_MAX_MESSAGES = 12


REFUSAL_MESSAGE = "Sorry , but I don't have enough information in the provided documents to answer that question, Thank you !!."


SOURCE_PATTERN = re.compile(
    r"\[SOURCE\s+(\d+)\]",
    re.IGNORECASE,
)




# ============================================================
# BM25 TOKENIZATION PATTERN
#
# Exported from the validated V13 notebook.
#
# Preserves identifiers such as:
#
# SEC11-BP06
# active.htm
# HTTP-502
# us-east-1
# ============================================================

TOKEN_PATTERN = re.compile(
    '[A-Za-z0-9]+(?:[-_.:/][A-Za-z0-9]+)*'
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

RERANKER_MODEL_NAME = (
    "cross-encoder/ms-marco-MiniLM-L6-v2"
)


# ============================================================
# LOAD VALIDATED LOCAL RETRIEVAL CACHE
# ============================================================

if not BACKEND_CACHE.exists():

    raise FileNotFoundError(

        "V13 backend cache does not exist: "
        f"{BACKEND_CACHE}"

    )


with open(

    BACKEND_CACHE,

    "rb",

) as cache_file:

    _backend_state = pickle.load(
        cache_file
    )


chunks = (
    _backend_state[
        "chunks"
    ]
)


chunk_lookup = (
    _backend_state[
        "chunk_lookup"
    ]
)


bm25 = (
    _backend_state[
        "bm25"
    ]
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

embedding_model = (

    SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

)


# ============================================================
# LOAD CROSSENCODER RERANKER
# ============================================================

reranker = (

    CrossEncoder(
        RERANKER_MODEL_NAME
    )

)


# ============================================================
# CONNECT TO PINECONE
#
# SECURITY:
#
# API key is NOT stored in this file.
# It must exist in the environment as:
#
# PINECONE_API_KEY
# ============================================================

PINECONE_API_KEY = (

    os.getenv(
        "PINECONE_API_KEY"
    )

)


if not PINECONE_API_KEY:

    raise RuntimeError(

        "PINECONE_API_KEY is not available "
        "in the Streamlit process environment."

    )


pinecone_client = (

    Pinecone(

        api_key=
            PINECONE_API_KEY

    )

)


pinecone_index = (

    pinecone_client.Index(

        PINECONE_INDEX_NAME

    )

)


# ============================================================
# INITIALIZE PERSISTENT MEMORY
# ============================================================

initialize_database(
    DEFAULT_MEMORY_DB
)



def tokenize_for_bm25(
    text,
):

    return [

        token.lower()

        for token
        in TOKEN_PATTERN.findall(
            text
        )

    ]


def dense_search(

    question,

    top_k=20,

    provider=None,

    document_id=None,

):

    query_vector = (

        embedding_model.encode(

            [
                question
            ],

            normalize_embeddings=True,

        )[0]

    )


    filters = []


    if provider is not None:

        filters.append(

            {

                "provider": {

                    "$eq":
                        provider

                }

            }

        )


    if document_id is not None:

        filters.append(

            {

                "document_id": {

                    "$eq":
                        document_id

                }

            }

        )


    if len(
        filters
    ) == 0:

        metadata_filter = None


    elif len(
        filters
    ) == 1:

        metadata_filter = (
            filters[0]
        )


    else:

        metadata_filter = {

            "$and":
                filters

        }


    response = (

        pinecone_index
        .documents
        .search(

            namespace=
                PINECONE_NAMESPACE,

            top_k=
                top_k,

            score_by=[

                DenseVectorQuery(

                    field=
                        PINECONE_VECTOR_FIELD,

                    values=
                        query_vector.tolist(),

                )

            ],

            include_fields=[

                "text",
                "provider",
                "document_id",
                "document_title",
                "source",
                "pdf_page",
                "document_chunk_number",
                "global_chunk_number",

            ],

            filter=
                metadata_filter,

        )

    )


    results = []


    for rank, match in enumerate(

        response.matches,

        start=1,

    ):

        results.append(

            {

                "chunk_id":

                    match.get(
                        "_id",
                        "",
                    ),

                "provider":

                    match.get(
                        "provider",
                        "",
                    ),

                "document_id":

                    match.get(
                        "document_id",
                        "",
                    ),

                "document_title":

                    match.get(
                        "document_title",
                        "",
                    ),

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

                "document_chunk_number":

                    int(
                        match.get(
                            "document_chunk_number",
                            0,
                        )
                    ),

                "global_chunk_number":

                    int(
                        match.get(
                            "global_chunk_number",
                            0,
                        )
                    ),

                "text":

                    match.get(
                        "text",
                        "",
                    ),

                "dense_score":

                    float(
                        match.score
                    ),

                "dense_rank":
                    rank,

            }

        )


    return results


def bm25_search(

    question,

    top_k=20,

    provider=None,

    document_id=None,

):

    query_tokens = (

        tokenize_for_bm25(
            question
        )

    )


    scores = (

        bm25.get_scores(
            query_tokens
        )

    )


    eligible_indices = []


    for index_number, chunk in enumerate(
        chunks
    ):

        metadata = chunk.metadata


        if (

            provider is not None

            and

            metadata[
                "provider"
            ]
            !=
            provider

        ):

            continue


        if (

            document_id is not None

            and

            metadata[
                "document_id"
            ]
            !=
            document_id

        ):

            continue


        eligible_indices.append(
            index_number
        )


    ranked_indices = sorted(

        eligible_indices,

        key=lambda i:
            scores[i],

        reverse=True,

    )[
        :top_k
    ]


    results = []


    for rank, index_number in enumerate(

        ranked_indices,

        start=1,

    ):

        chunk = chunks[
            index_number
        ]


        results.append(

            {

                "chunk_id":

                    chunk.metadata[
                        "chunk_id"
                    ],

                "provider":

                    chunk.metadata[
                        "provider"
                    ],

                "document_id":

                    chunk.metadata[
                        "document_id"
                    ],

                "document_title":

                    chunk.metadata[
                        "document_title"
                    ],

                "source":

                    chunk.metadata[
                        "source"
                    ],

                "pdf_page":

                    chunk.metadata[
                        "pdf_page"
                    ],

                "document_chunk_number":

                    chunk.metadata[
                        "document_chunk_number"
                    ],

                "global_chunk_number":

                    chunk.metadata[
                        "global_chunk_number"
                    ],

                "text":

                    chunk.page_content,

                "bm25_score":

                    float(
                        scores[
                            index_number
                        ]
                    ),

                "bm25_rank":
                    rank,

            }

        )


    return results


def reciprocal_rank_fusion(

    dense_results,

    bm25_results,

    rrf_k=60,

):

    fused = {}


    # --------------------------------------------------------
    # DENSE
    # --------------------------------------------------------

    for rank, result in enumerate(

        dense_results,

        start=1,

    ):

        chunk_id = (
            result[
                "chunk_id"
            ]
        )


        if chunk_id not in fused:

            fused[
                chunk_id
            ] = dict(
                result
            )


        fused[
            chunk_id
        ][
            "rrf_score"
        ] = (

            fused[
                chunk_id
            ].get(
                "rrf_score",
                0.0,
            )

            +

            1.0
            /
            (
                rrf_k
                +
                rank
            )

        )


        fused[
            chunk_id
        ][
            "dense_rank"
        ] = rank


    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    for rank, result in enumerate(

        bm25_results,

        start=1,

    ):

        chunk_id = (
            result[
                "chunk_id"
            ]
        )


        if chunk_id not in fused:

            fused[
                chunk_id
            ] = dict(
                result
            )


        fused[
            chunk_id
        ][
            "rrf_score"
        ] = (

            fused[
                chunk_id
            ].get(
                "rrf_score",
                0.0,
            )

            +

            1.0
            /
            (
                rrf_k
                +
                rank
            )

        )


        fused[
            chunk_id
        ][
            "bm25_rank"
        ] = rank


    ranked = sorted(

        fused.values(),

        key=lambda item:

            item[
                "rrf_score"
            ],

        reverse=True,

    )


    return ranked


def hybrid_candidates(

    question,

    dense_top_k=
        DENSE_TOP_K,

    bm25_top_k=
        BM25_TOP_K,

    fusion_top_k=
        FUSION_TOP_K,

    provider=None,

    document_id=None,

):

    dense_results = (

        dense_search(

            question=

                question,

            top_k=

                dense_top_k,

            provider=

                provider,

            document_id=

                document_id,

        )

    )


    sparse_results = (

        bm25_search(

            question=

                question,

            top_k=

                bm25_top_k,

            provider=

                provider,

            document_id=

                document_id,

        )

    )


    fused_results = (

        reciprocal_rank_fusion(

            dense_results,

            sparse_results,

            rrf_k=
                RRF_K,

        )

    )


    return fused_results[
        :fusion_top_k
    ]


def get_document_neighbors(

    document_id,

    document_chunk_number,

    neighbor_range=1,

):

    neighbors = []


    for offset in range(

        -neighbor_range,

        neighbor_range
        +
        1,

    ):

        key = (

            document_id,

            document_chunk_number
            +
            offset,

        )


        chunk = (

            chunk_lookup.get(
                key
            )

        )


        if chunk is not None:

            neighbors.append(
                chunk
            )


    return neighbors


def expand_hybrid_neighbors(
    hybrid_results,
):

    expanded = []


    for result in hybrid_results:

        neighbors = (

            get_document_neighbors(

                result[
                    "document_id"
                ],

                result[
                    "document_chunk_number"
                ],

                NEIGHBOR_RANGE,

            )

        )


        for neighbor in neighbors:

            expanded.append(

                {

                    "chunk_id":

                        neighbor.metadata[
                            "chunk_id"
                        ],

                    "provider":

                        neighbor.metadata[
                            "provider"
                        ],

                    "document_id":

                        neighbor.metadata[
                            "document_id"
                        ],

                    "document_title":

                        neighbor.metadata[
                            "document_title"
                        ],

                    "source":

                        neighbor.metadata[
                            "source"
                        ],

                    "pdf_page":

                        neighbor.metadata[
                            "pdf_page"
                        ],

                    "document_chunk_number":

                        neighbor.metadata[
                            "document_chunk_number"
                        ],

                    "global_chunk_number":

                        neighbor.metadata[
                            "global_chunk_number"
                        ],

                    "text":

                        neighbor.page_content,

                    "rrf_score":

                        result[
                            "rrf_score"
                        ],

                    "dense_rank":

                        result.get(
                            "dense_rank"
                        ),

                    "bm25_rank":

                        result.get(
                            "bm25_rank"
                        ),

                }

            )


    return expanded


def deduplicate_results(
    results,
):

    unique = {}


    for result in results:

        chunk_id = (
            result[
                "chunk_id"
            ]
        )


        if chunk_id not in unique:

            unique[
                chunk_id
            ] = result


        elif (

            result[
                "rrf_score"
            ]

            >

            unique[
                chunk_id
            ][
                "rrf_score"
            ]

        ):

            unique[
                chunk_id
            ] = result


    return list(
        unique.values()
    )


def filter_informative_results(
    results,
):

    filtered = [

        result

        for result
        in results

        if len(
            result[
                "text"
            ].strip()
        )
        >=
        MIN_EVIDENCE_CHARS

    ]


    if filtered:

        return filtered


    return results


def crossencoder_rerank(

    question,

    results,

    final_top_k=
        FINAL_TOP_K,

):

    if not results:

        return []


    results = (

        filter_informative_results(
            results
        )

    )


    pairs = [

        [

            question,

            result[
                "text"
            ],

        ]

        for result
        in results

    ]


    scores = (

        reranker.predict(
            pairs
        )

    )


    for result, score in zip(

        results,

        scores,

    ):

        result[
            "rerank_score"
        ] = float(
            score
        )


    ranked = sorted(

        results,

        key=lambda item:

            item[
                "rerank_score"
            ],

        reverse=True,

    )


    return ranked[
        :final_top_k
    ]


def hybrid_retrieve(

    question,

    provider=None,

    document_id=None,

    final_top_k=
        FINAL_TOP_K,

):

    # 1. Dense + BM25 + RRF

    fused = (

        hybrid_candidates(

            question=

                question,

            provider=

                provider,

            document_id=

                document_id,

        )

    )


    # 2. Same-document neighbors

    expanded = (

        expand_hybrid_neighbors(
            fused
        )

    )


    # 3. Deduplicate

    unique = (

        deduplicate_results(
            expanded
        )

    )


    # 4. Remove weak evidence

    informative = (

        filter_informative_results(
            unique
        )

    )


    # 5. CrossEncoder

    final_results = (

        crossencoder_rerank(

            question=

                question,

            results=

                informative,

            final_top_k=

                final_top_k,

        )

    )


    return final_results


def build_comparison_retrieval_query(

    question,

    provider,

):

    retrieval_query = f"""
{provider} reliability best practices for building reliable cloud workloads.

Find specific architectural principles, design practices,
reliability patterns, resilience techniques, availability guidance,
failure recovery practices, redundancy strategies,
fault tolerance recommendations, and workload reliability guidance.

Original comparison topic:

{question}
""".strip()


    return retrieval_query


def hybrid_retrieve_for_comparison(

    question,

    provider,

    final_top_k=
        COMPARISON_FINAL_TOP_K,

):

    # --------------------------------------------------------
    # A. CREATE PROVIDER-SPECIFIC RETRIEVAL QUERY
    # --------------------------------------------------------

    retrieval_query = (

        build_comparison_retrieval_query(

            question=
                question,

            provider=
                provider,

        )

    )


    # --------------------------------------------------------
    # B. RUN DENSE + BM25 + RRF
    #
    # We deliberately retrieve more candidates here.
    # --------------------------------------------------------

    fused_results = (

        hybrid_candidates(

            question=
                retrieval_query,

            dense_top_k=
                COMPARISON_DENSE_TOP_K,

            bm25_top_k=
                COMPARISON_BM25_TOP_K,

            fusion_top_k=
                COMPARISON_FUSION_TOP_K,

            provider=
                provider,

        )

    )


    # --------------------------------------------------------
    # C. EXPAND ±1 NEIGHBORS
    #
    # Neighbor expansion remains document-safe.
    # --------------------------------------------------------

    expanded_results = (

        expand_hybrid_neighbors(

            fused_results

        )

    )


    # --------------------------------------------------------
    # D. REMOVE DUPLICATES
    # --------------------------------------------------------

    unique_results = (

        deduplicate_results(

            expanded_results

        )

    )


    # --------------------------------------------------------
    # E. REMOVE VERY WEAK / SHORT CHUNKS
    # --------------------------------------------------------

    informative_results = (

        filter_informative_results(

            unique_results

        )

    )


    # --------------------------------------------------------
    # F. CROSSENCODER RERANK
    #
    # Important:
    #
    # We rerank against the provider-specific query,
    # not the original broad comparison question.
    # --------------------------------------------------------

    final_results = (

        crossencoder_rerank(

            question=
                retrieval_query,

            results=
                informative_results,

            final_top_k=
                final_top_k,

        )

    )


    return final_results


def hybrid_cross_provider_retrieve(

    question,

    providers,

    per_provider_top_k=3,

):

    # --------------------------------------------------------
    # Validate provider list
    # --------------------------------------------------------

    if not providers:

        return []


    combined_results = []

    seen_chunks = set()


    # ========================================================
    # STEP 2
    # RETRIEVE EVIDENCE FOR EACH PROVIDER SEPARATELY
    # ========================================================

    for provider in providers:

        provider_results = (

            hybrid_retrieve_for_comparison(

                question=
                    question,

                provider=
                    provider,

                # Retrieve enough so we can keep
                # the best requested amount.
                final_top_k=
                    max(
                        per_provider_top_k,
                        5,
                    ),

            )

        )


        # ====================================================
        # STEP 3
        # KEEP A PROVIDER QUOTA
        #
        # Example:
        #
        # AWS   → best 3
        # Azure → best 3
        #
        # This prevents one provider from taking all
        # available evidence slots.
        # ====================================================

        selected_results = (

            provider_results[
                :per_provider_top_k
            ]

        )


        # ====================================================
        # STEP 4
        # ADD RESULTS WITHOUT DUPLICATES
        # ====================================================

        for result in selected_results:

            chunk_key = (

                result.get(
                    "chunk_id"
                )

                or

                (
                    result[
                        "document_id"
                    ],

                    result[
                        "document_chunk_number"
                    ],

                )

            )


            if chunk_key in seen_chunks:

                continue


            seen_chunks.add(
                chunk_key
            )


            # Make a copy so we do not accidentally
            # modify the original retrieval result.
            result_copy = dict(
                result
            )


            result_copy[
                "comparison_provider"
            ] = provider


            combined_results.append(
                result_copy
            )


    return combined_results


def build_context(
    evidence,
):

    blocks = []


    for source_number, result in enumerate(

        evidence,

        start=1,

    ):

        blocks.append(

            f"""
[SOURCE {source_number}]
Provider: {result["provider"]}
Document: {result["document_title"]}
PDF page: {result["pdf_page"]}

Evidence:
{result["text"]}
""".strip()

        )


    return (

        "\n\n"
        "---------------------------------------"
        "\n\n"

    ).join(
        blocks
    )


def build_grounded_prompt(

    question,

    evidence,

    required_providers=None,

):

    # --------------------------------------------------------
    # Default:
    # no provider requirement
    # --------------------------------------------------------

    if required_providers is None:

        required_providers = []


    # ========================================================
    # VALID SOURCE NUMBERS
    # ========================================================

    valid_sources = " ".join(

        f"[SOURCE {number}]"

        for number in range(

            1,

            len(evidence) + 1,

        )

    )


    # ========================================================
    # BUILD THE EVIDENCE CONTEXT
    # ========================================================

    context = (

        build_context(
            evidence
        )

    )


    # ========================================================
    # PROVIDER REQUIREMENT
    #
    # Normal question:
    #
    # provider_instruction = ""
    #
    # Comparison:
    #
    # provider_instruction explains that evidence from
    # every requested provider must appear in the answer.
    # ========================================================

    provider_instruction = ""


    if required_providers:

        provider_list = ", ".join(
            required_providers
        )


        provider_instruction = f"""
IMPORTANT CROSS-PROVIDER COMPARISON RULE:

This question requires comparison across these providers:

{provider_list}

You MUST:

1. Discuss every required provider.
2. Use evidence from every required provider.
3. Cite at least one valid SOURCE for every required provider.
4. Clearly explain similarities and differences supported by the evidence.
5. Do not refuse merely because the providers use different terminology.
6. If one provider truly has no usable evidence, state which provider lacks evidence.
"""


    # ========================================================
    # FINAL PROMPT
    # ========================================================

    prompt = f"""
You are a grounded multi-cloud RAG assistant.

Answer ONLY from the supplied evidence.

VALID SOURCE MARKERS:

{valid_sources}

{provider_instruction}

GENERAL RULES:

1. Use only the supplied evidence.

2. Do not use outside knowledge.

3. Do not invent facts.

4. Cite factual claims using [SOURCE N].

5. Never invent a SOURCE number.

6. Use only SOURCE numbers listed above.

7. Synthesize the evidence rather than simply copying it.

8. When comparing providers, organize the answer clearly.

9. If the supplied evidence truly does not support an answer,
   reply exactly:

{REFUSAL_MESSAGE}


QUESTION:

{question}


EVIDENCE:

{context}


ANSWER:
""".strip()


    return prompt


def stream_grounded_ollama_tokens(

    question,

    evidence,

    required_providers=None,

):

    # ========================================================
    # BUILD GROUNDED PROMPT
    # ========================================================

    prompt = (

        build_grounded_prompt(

            question=
                question,

            evidence=
                evidence,

            required_providers=
                required_providers,

        )

    )


    # ========================================================
    # OLLAMA REQUEST
    # ========================================================

    payload = {

        "model":
            OLLAMA_MODEL,

        "messages": [

            {

                "role":
                    "user",

                "content":
                    prompt,

            }

        ],

        "stream":
            True,

        "options": {

            "temperature":
                0.1,

        },

    }


    # ========================================================
    # OPEN STREAMING CONNECTION
    # ========================================================

    with requests.post(

        f"{OLLAMA_BASE_URL}/api/chat",

        json=
            payload,

        stream=
            True,

        timeout=
            300,

    ) as response:

        response.raise_for_status()


        # ====================================================
        # READ EACH STREAM EVENT
        # ====================================================

        for line in response.iter_lines():

            if not line:

                continue


            # ------------------------------------------------
            # Ollama sends JSON objects one line at a time.
            # ------------------------------------------------

            data = (

                json.loads(

                    line.decode(
                        "utf-8"
                    )

                )

            )


            # ------------------------------------------------
            # Extract generated text.
            # ------------------------------------------------

            text = (

                data
                .get(
                    "message",
                    {}
                )
                .get(
                    "content",
                    ""
                )

            )


            # ------------------------------------------------
            # Yield token/chunk to whoever called us.
            #
            # Jupyter can print it.
            #
            # Streamlit can display it.
            # ------------------------------------------------

            if text:

                yield text


def stream_grounded_ollama_answer(

    question,

    evidence,

    required_providers=None,

):

    # ========================================================
    # STEP 1
    # BUILD THE GROUNDED PROMPT
    #
    # The important update is:
    #
    # required_providers
    #
    # Example:
    #
    # ["AWS", "Azure"]
    # ========================================================

    prompt = (

        build_grounded_prompt(

            question=
                question,

            evidence=
                evidence,

            required_providers=
                required_providers,

        )

    )


    # ========================================================
    # STEP 2
    # PREPARE OLLAMA REQUEST
    # ========================================================

    payload = {

        "model":
            OLLAMA_MODEL,

        "messages": [

            {

                "role":
                    "user",

                "content":
                    prompt,

            }

        ],

        "stream":
            True,

        "options": {

            "temperature":
                0.1,

        },

    }


    # ========================================================
    # STEP 3
    # START TIMING
    # ========================================================

    start_time = (

        time.perf_counter()

    )


    first_token_time = None

    response_parts = []


    print()

    print("=" * 80)
    print("LIVE GROUNDED RAG RESPONSE")
    print("=" * 80)

    print()


    # ========================================================
    # STEP 4
    # STREAM OLLAMA RESPONSE
    # ========================================================

    with requests.post(

        f"{OLLAMA_BASE_URL}/api/chat",

        json=
            payload,

        stream=
            True,

        timeout=
            300,

    ) as response:

        response.raise_for_status()


        for line in response.iter_lines():

            if not line:

                continue


            data = (

                json.loads(

                    line.decode(
                        "utf-8"
                    )

                )

            )


            text = (

                data
                .get(
                    "message",
                    {}
                )
                .get(
                    "content",
                    ""
                )

            )


            if text:

                # --------------------------------------------
                # Capture time to first visible token
                # --------------------------------------------

                if first_token_time is None:

                    first_token_time = (

                        time.perf_counter()
                        -
                        start_time

                    )


                # --------------------------------------------
                # Print live in Jupyter
                # --------------------------------------------

                print(

                    text,

                    end="",

                    flush=True,

                )


                # --------------------------------------------
                # Save token for final answer
                # --------------------------------------------

                response_parts.append(
                    text
                )


    # ========================================================
    # STEP 5
    # ASSEMBLE COMPLETE RAW ANSWER
    # ========================================================

    raw_answer = (

        "".join(
            response_parts
        )

    )


    total_time = (

        time.perf_counter()
        -
        start_time

    )


    print()
    print()


    # ========================================================
    # STEP 6
    # RETURN COMPLETE STREAMING RESULT
    # ========================================================

    return {

        "raw_answer":
            raw_answer,

        "time_to_first_token":
            first_token_time,

        "total_time":
            total_time,

    }


def validate_source_numbers(

    answer,

    evidence,

):

    # --------------------------------------------------------
    # Extract all SOURCE numbers from the LLM answer
    # --------------------------------------------------------

    source_numbers = [

        int(
            number
        )

        for number
        in SOURCE_PATTERN.findall(
            answer
        )

    ]


    # --------------------------------------------------------
    # Valid source numbers are:
    #
    # 1 through len(evidence)
    #
    # Example:
    #
    # 5 evidence chunks
    #
    # valid:
    # 1 2 3 4 5
    # --------------------------------------------------------

    valid_source_numbers = set(

        range(

            1,

            len(
                evidence
            )
            +
            1,

        )

    )


    # --------------------------------------------------------
    # Find invented / invalid source numbers
    # --------------------------------------------------------

    invalid_source_numbers = [

        number

        for number
        in source_numbers

        if number
        not in
        valid_source_numbers

    ]


    return (

        source_numbers,

        invalid_source_numbers,

    )


def create_citation_label(
    result,
):

    return (

        f"[{result['provider']} | "

        f"{result['document_title']} | "

        f"PDF page {result['pdf_page']}]"

    )


def replace_source_markers(

    answer,

    evidence,

):

    citations_used = []


    def replacement(
        match,
    ):

        source_number = int(
            match.group(
                1
            )
        )


        evidence_index = (

            source_number
            -
            1

        )


        if (

            0
            <=
            evidence_index
            <
            len(
                evidence
            )

        ):

            citation = (

                create_citation_label(

                    evidence[
                        evidence_index
                    ]

                )

            )


            citations_used.append(
                citation
            )


            return citation


        # ----------------------------------------------------
        # Invalid SOURCE numbers are removed.
        # ----------------------------------------------------

        return ""


    final_answer = (

        SOURCE_PATTERN.sub(

            replacement,

            answer,

        )

    )


    # --------------------------------------------------------
    # Remove duplicate citations from our citation list.
    # --------------------------------------------------------

    citations_used = list(

        dict.fromkeys(
            citations_used
        )

    )


    return (

        final_answer,

        citations_used,

    )


def format_conversation_history(
    messages,
):

    recent_messages = (

        messages[
            -MEMORY_MAX_MESSAGES:
        ]

    )


    formatted_lines = []


    for message in recent_messages:

        role = (

            message.get(
                "role",
                ""
            )

        )


        content = (

            message.get(
                "content",
                ""
            )

        )


        formatted_lines.append(

            f"{role.upper()}: "
            f"{content}"

        )


    return "\n".join(
        formatted_lines
    )


def rewrite_followup_question(

    question,

    history,

):

    # --------------------------------------------------------
    # If there is no conversation history,
    # the question is already standalone.
    # --------------------------------------------------------

    if not history:

        return question


    # --------------------------------------------------------
    # FORMAT THE HISTORY
    # --------------------------------------------------------

    formatted_history = (

        format_conversation_history(
            history
        )

    )


    # --------------------------------------------------------
    # BUILD THE REWRITE PROMPT
    # --------------------------------------------------------

    rewrite_prompt = f"""
You are a query rewriting assistant for a RAG system.

Your ONLY job is to rewrite the current user question into
a complete standalone retrieval question.

Use the conversation history only to resolve references such as:

- it
- that
- this
- they
- those
- the previous one
- the first one
- compare with that

Rules:

1. Do NOT answer the question.
2. Do NOT explain anything.
3. Do NOT add facts that are not in the conversation.
4. Preserve important cloud providers, products, and concepts.
5. Return ONLY the rewritten standalone question.
6. If the question is already standalone, return it unchanged.


CONVERSATION HISTORY:

{formatted_history}


CURRENT QUESTION:

{question}


STANDALONE QUESTION:
""".strip()


    # --------------------------------------------------------
    # CALL OLLAMA
    #
    # We do NOT need streaming here because the output
    # should only be one short rewritten question.
    # --------------------------------------------------------

    payload = {

        "model":
            OLLAMA_MODEL,

        "messages": [

            {

                "role":
                    "user",

                "content":
                    rewrite_prompt,

            }

        ],

        "stream":
            False,

        "options": {

            "temperature":
                0.0,

        },

    }


    response = requests.post(

        f"{OLLAMA_BASE_URL}/api/chat",

        json=
            payload,

        timeout=
            120,

    )


    response.raise_for_status()


    rewritten_question = (

        response
        .json()
        .get(
            "message",
            {}
        )
        .get(
            "content",
            ""
        )
        .strip()

    )


    # --------------------------------------------------------
    # REMOVE ACCIDENTAL QUOTES
    #
    # Example:
    #
    # "How does Azure compare..."
    #
    # becomes:
    #
    # How does Azure compare...
    # --------------------------------------------------------

    rewritten_question = (

        rewritten_question
        .strip('"')
        .strip("'")
        .strip()

    )


    # --------------------------------------------------------
    # SAFETY FALLBACK
    #
    # If Ollama somehow returns nothing,
    # use the original question.
    # --------------------------------------------------------

    if not rewritten_question:

        rewritten_question = (
            question
        )


    return rewritten_question


def evidence_for_memory(
    evidence,
):

    memory_sources = []


    for result in evidence:

        memory_sources.append(

            {

                "provider":
                    result[
                        "provider"
                    ],

                "document_id":
                    result[
                        "document_id"
                    ],

                "document_title":
                    result[
                        "document_title"
                    ],

                "source":
                    result[
                        "source"
                    ],

                "pdf_page":
                    result[
                        "pdf_page"
                    ],

                "document_chunk_number":
                    result[
                        "document_chunk_number"
                    ],

                "text":
                    result[
                        "text"
                    ],

                "rrf_score":
                    result.get(
                        "rrf_score"
                    ),

                "rerank_score":
                    result.get(
                        "rerank_score"
                    ),

            }

        )


    return memory_sources


def ask_v13_with_memory(

    conversation_id,

    question,

    mode="global",

    provider=None,

    providers=None,

    document_id=None,

):

    # ========================================================
    # STEP 1
    # LOAD RECENT SQLITE MEMORY
    # ========================================================

    history = (

        load_recent_memory(

            conversation_id,

            DEFAULT_MEMORY_DB,

        )

    )


    print()

    print("=" * 70)
    print("CONVERSATION MEMORY")
    print("=" * 70)

    print()

    print(
        "Messages loaded:",
        len(history),
    )


    # ========================================================
    # STEP 2
    # REWRITE FOLLOW-UP INTO STANDALONE QUESTION
    # ========================================================

    standalone_question = (

        rewrite_followup_question(

            question=
                question,

            history=
                history,

        )

    )


    print()

    print("=" * 70)
    print("QUERY REWRITING")
    print("=" * 70)

    print()

    print(
        "Original question:"
    )

    print(
        question
    )

    print()

    print(
        "Standalone question:"
    )

    print(
        standalone_question
    )


    # ========================================================
    # STEP 3
    # RETRIEVAL ROUTING
    # ========================================================

    required_providers = []


    # --------------------------------------------------------
    # PROVIDER MODE
    # --------------------------------------------------------

    if mode == "provider":

        if not provider:

            raise ValueError(
                "provider mode requires provider="
            )


        evidence = (

            hybrid_retrieve(

                question=
                    standalone_question,

                provider=
                    provider,

            )

        )


        required_providers = [
            provider
        ]


    # --------------------------------------------------------
    # CROSS-PROVIDER MODE
    #
    # IMPORTANT:
    #
    # This now uses our NEW Cell 59 retriever:
    #
    # provider-specific query
    # deep Dense + BM25
    # RRF
    # CrossEncoder
    # balanced provider quota
    # --------------------------------------------------------

    elif mode == "cross_provider":

        if not providers:

            raise ValueError(

                "cross_provider mode requires "
                "providers=[...]"

            )


        required_providers = list(
            providers
        )


        evidence = (

            hybrid_cross_provider_retrieve(

                question=
                    standalone_question,

                providers=
                    required_providers,

                per_provider_top_k=
                    3,

            )

        )


    # --------------------------------------------------------
    # DOCUMENT MODE
    # --------------------------------------------------------

    elif mode == "document":

        if not document_id:

            raise ValueError(
                "document mode requires document_id="
            )


        evidence = (

            hybrid_retrieve(

                question=
                    standalone_question,

                document_id=
                    document_id,

            )

        )


    # --------------------------------------------------------
    # GLOBAL MODE
    # --------------------------------------------------------

    else:

        evidence = (

            hybrid_retrieve(

                question=
                    standalone_question,

            )

        )


    # ========================================================
    # STEP 4
    # DETERMINE WHICH PROVIDERS WERE RETRIEVED
    # ========================================================

    retrieved_providers = sorted(

        {

            item[
                "provider"
            ]

            for item
            in evidence

        }

    )


    print()

    print(
        "Required providers:",
        required_providers
    )


    print(
        "Retrieved providers:",
        retrieved_providers
    )


    # ========================================================
    # STEP 5
    # NO-EVIDENCE CASE
    # ========================================================

    if not evidence:

        raw_answer = (
            REFUSAL_MESSAGE
        )

        final_answer = (
            REFUSAL_MESSAGE
        )

        citations_used = []

        source_numbers = []

        invalid_source_numbers = []

        citation_valid = False

        time_to_first_token = None

        total_time = None


    # ========================================================
    # STEP 6
    # GROUNDED STREAMING GENERATION
    # ========================================================

    else:

        stream_result = (

            stream_grounded_ollama_answer(

                question=
                    standalone_question,

                evidence=
                    evidence,

                required_providers=(
                    required_providers
                    if mode
                    ==
                    "cross_provider"
                    else None
                ),

            )

        )


        raw_answer = (

            stream_result[
                "raw_answer"
            ]

        )


        time_to_first_token = (

            stream_result[
                "time_to_first_token"
            ]

        )


        total_time = (

            stream_result[
                "total_time"
            ]

        )


        # ====================================================
        # SOURCE NUMBER VALIDATION
        # ====================================================

        (
            source_numbers,
            invalid_source_numbers,

        ) = (

            validate_source_numbers(

                answer=
                    raw_answer,

                evidence=
                    evidence,

            )

        )


        # ====================================================
        # DETERMINISTIC CITATION REPLACEMENT
        # ====================================================

        (
            final_answer,
            citations_used,

        ) = (

            replace_source_markers(

                answer=
                    raw_answer,

                evidence=
                    evidence,

            )

        )


        citation_valid = (

            len(
                source_numbers
            )
            >
            0

            and

            len(
                invalid_source_numbers
            )
            ==
            0

            and

            len(
                citations_used
            )
            >
            0

        )


    # ========================================================
    # STEP 7
    # SAVE USER MESSAGE
    # ========================================================

    save_message(

        conversation_id=
            conversation_id,

        role=
            "user",

        content=
            question,

        query_type=
            mode,

        standalone_question=
            standalone_question,

        sources=
            [],

        db_path=
            DEFAULT_MEMORY_DB,

    )


    # ========================================================
    # STEP 8
    # TITLE CONVERSATION
    # ========================================================

    auto_title_conversation(

        conversation_id,

        question,

        DEFAULT_MEMORY_DB,

    )


    # ========================================================
    # STEP 9
    # SAVE ASSISTANT ANSWER
    # ========================================================

    save_message(

        conversation_id=
            conversation_id,

        role=
            "assistant",

        content=
            final_answer,

        query_type=
            mode,

        standalone_question=
            standalone_question,

        sources=
            evidence_for_memory(
                evidence
            ),

        db_path=
            DEFAULT_MEMORY_DB,

    )


    # ========================================================
    # STEP 10
    # RETURN COMPLETE CURRENT V13 RESULT
    # ========================================================

    return {

        "conversation_id":
            conversation_id,

        "question":
            question,

        "standalone_question":
            standalone_question,

        "mode":
            mode,

        # NEW / REQUIRED BY CELL 60
        "required_providers":
            required_providers,

        # NEW / REQUIRED BY CELL 60
        "retrieved_providers":
            retrieved_providers,

        "answer":
            final_answer,

        "raw_answer":
            raw_answer,

        "evidence":
            evidence,

        "citations":
            citations_used,

        "source_numbers":
            source_numbers,

        "invalid_source_numbers":
            invalid_source_numbers,

        "citation_valid":
            citation_valid,

        "time_to_first_token":
            time_to_first_token,

        "total_time":
            total_time,

    }



# ============================================================
# BACKEND HEALTH INFORMATION
# ============================================================

def get_v13_backend_info():

    return {

        "pinecone_index":
            PINECONE_INDEX_NAME,

        "pinecone_namespace":
            PINECONE_NAMESPACE,

        "embedding_model":
            EMBEDDING_MODEL_NAME,

        "reranker_model":
            RERANKER_MODEL_NAME,

        "ollama_model":
            OLLAMA_MODEL,

        "chunks":
            len(
                chunks
            ),

        "chunk_lookup_entries":
            len(
                chunk_lookup
            ),

        "memory_database":
            str(
                DEFAULT_MEMORY_DB
            ),

    }




# ============================================================
# STREAMLIT / UI STREAMING BRIDGE
# ============================================================

import contextvars


# ============================================================
# REQUEST-LOCAL TOKEN CALLBACK
# ============================================================

_STREAM_TOKEN_CALLBACK = contextvars.ContextVar(

    "v13_stream_token_callback",

    default=None,

)


# ============================================================
# CALLBACK-AWARE GROUNDED OLLAMA STREAM
#
# IMPORTANT:
#
# This intentionally redefines stream_grounded_ollama_answer()
# later in this module.
#
# ask_v13_with_memory() performs a runtime global lookup, so it
# automatically uses this callback-aware implementation.
#
# Existing CLI / notebook behavior remains compatible because
# the callback defaults to None.
# ============================================================

def stream_grounded_ollama_answer(

    question,

    evidence,

    required_providers=None,

):

    # --------------------------------------------------------
    # Build the same grounded prompt used by V13.
    # --------------------------------------------------------

    prompt = (

        build_grounded_prompt(

            question=
                question,

            evidence=
                evidence,

            required_providers=
                required_providers,

        )

    )


    # --------------------------------------------------------
    # Ollama request
    # --------------------------------------------------------

    payload = {

        "model":
            OLLAMA_MODEL,

        "messages": [

            {

                "role":
                    "user",

                "content":
                    prompt,

            }

        ],

        "stream":
            True,

        "options": {

            "temperature":
                0.1,

        },

    }


    # --------------------------------------------------------
    # Timing
    # --------------------------------------------------------

    start_time = (

        time.perf_counter()

    )


    first_token_time = None

    response_parts = []


    # --------------------------------------------------------
    # Get request-local UI callback.
    #
    # None when running from CLI / ordinary backend code.
    # --------------------------------------------------------

    token_callback = (

        _STREAM_TOKEN_CALLBACK.get()

    )


    # --------------------------------------------------------
    # Stream Ollama
    # --------------------------------------------------------

    with requests.post(

        f"{OLLAMA_BASE_URL}/api/chat",

        json=
            payload,

        stream=
            True,

        timeout=
            300,

    ) as response:

        response.raise_for_status()


        for line in response.iter_lines():

            if not line:

                continue


            data = (

                json.loads(

                    line.decode(
                        "utf-8"
                    )

                )

            )


            text = (

                data
                .get(
                    "message",
                    {}
                )
                .get(
                    "content",
                    ""
                )

            )


            if not text:

                continue


            # ------------------------------------------------
            # First-token latency
            # ------------------------------------------------

            if first_token_time is None:

                first_token_time = (

                    time.perf_counter()
                    -
                    start_time

                )


            # ------------------------------------------------
            # Save complete response
            # ------------------------------------------------

            response_parts.append(
                text
            )


            # ------------------------------------------------
            # LIVE UI CALLBACK
            #
            # Streamlit will provide this callback.
            #
            # Example:
            #
            # def on_token(token):
            #     ...
            #
            # If there is no UI callback, nothing happens here.
            # ------------------------------------------------

            if token_callback is not None:

                token_callback(
                    text
                )


    # --------------------------------------------------------
    # Final raw answer
    # --------------------------------------------------------

    raw_answer = (

        "".join(
            response_parts
        )

    )


    total_time = (

        time.perf_counter()
        -
        start_time

    )


    return {

        "raw_answer":
            raw_answer,

        "time_to_first_token":
            first_token_time,

        "total_time":
            total_time,

    }


# ============================================================
# STREAMING VERSION OF THE COMPLETE V13 TURN
#
# This function does NOT duplicate:
#
# - memory rewriting
# - retrieval
# - cross-provider logic
# - citation validation
# - SQLite persistence
#
# All of that remains inside the already-tested
# ask_v13_with_memory().
# ============================================================

def ask_v13_with_memory_streaming(

    conversation_id,

    question,

    mode="global",

    provider=None,

    providers=None,

    document_id=None,

    token_callback=None,

):

    # --------------------------------------------------------
    # Associate callback with THIS request.
    # --------------------------------------------------------

    callback_context_token = (

        _STREAM_TOKEN_CALLBACK.set(

            token_callback

        )

    )


    try:

        # ----------------------------------------------------
        # Call the already-tested V13 conversational pipeline.
        # ----------------------------------------------------

        result = (

            ask_v13_with_memory(

                conversation_id=
                    conversation_id,

                question=
                    question,

                mode=
                    mode,

                provider=
                    provider,

                providers=
                    providers,

                document_id=
                    document_id,

            )

        )


        return result


    finally:

        # ----------------------------------------------------
        # Always remove callback after the request.
        # ----------------------------------------------------

        _STREAM_TOKEN_CALLBACK.reset(

            callback_context_token

        )






# ============================================================
# V13 CELL 75 PRODUCTION POLISH OVERRIDES
# ============================================================


# ============================================================
# GENERIC PROVIDER-SPECIFIC COMPARISON RETRIEVAL
# ============================================================

def build_comparison_retrieval_query(

    question,

    provider,

):

    return f"""
{provider} official cloud guidance relevant to this comparison.

Retrieve specific evidence from {provider} that directly helps answer
the comparison question below.

Original comparison question:
{question}

Focus ONLY on concepts relevant to the actual question.

Depending on the question, useful evidence may involve:

- cloud platform capabilities
- architecture
- compute
- storage
- networking
- security
- identity
- reliability
- resilience
- availability
- deployment
- infrastructure as code
- governance
- operations
- monitoring
- cost optimization
- application architecture
- migration
- cloud adoption

Do not force reliability concepts into the retrieval unless
reliability is actually part of the question.

Prefer substantive architectural or technical guidance over:

- table of contents pages
- copyright pages
- revision histories
- generic introductions

Provider being researched:
{provider}
""".strip()


# ============================================================
# IMPROVED GROUNDED PROMPT
# ============================================================

def build_grounded_prompt(

    question,

    evidence,

    required_providers=None,

):

    context = (

        build_context(
            evidence
        )

    )


    rules = f"""
You are a grounded enterprise cloud RAG assistant.

Answer the user's question using ONLY the evidence supplied below.

GROUNDING RULES:

1. Use only information supported by the supplied evidence.

2. Every factual claim derived from the documents must include
   one or more citations using exactly this format:

   [SOURCE N]

3. N must correspond to a source that actually appears in the
   evidence.

4. Never invent source numbers.

5. Never invent cloud services, recommendations, capabilities,
   frameworks, or conclusions that are not supported by the evidence.

6. If the retrieved evidence does not establish something,
   say:

   "The retrieved evidence does not establish that point."

   Do NOT convert missing evidence into a claim that a provider
   does not have a product, framework, feature, capability,
   recommendation, or practice.

7. Do not infer that one provider is better, stronger, weaker,
   more mature, or more comprehensive merely because one provider
   has more retrieved chunks or more detailed text in the current
   evidence set.

8. Compare providers only on dimensions supported by the evidence.

9. Be concise and avoid unnecessary repetition.

10. Do not produce both a "Conclusion" and a separate
    "Final Answer" that repeat the same information.

11. If there is genuinely not enough evidence to answer the
    question, respond with exactly:

    {REFUSAL_MESSAGE}
""".strip()


    if required_providers:

        provider_list = (

            ", ".join(
                required_providers
            )

        )


        comparison_rules = f"""

CROSS-PROVIDER COMPARISON RULES:

The requested providers are:

{provider_list}

You must:

- discuss every requested provider when usable evidence exists
- cite evidence from each requested provider
- identify meaningful similarities when supported
- identify meaningful differences when supported
- keep the comparison balanced
- distinguish "not found in retrieved evidence" from
  "the provider does not have it"
- never rank one provider as superior solely because its retrieved
  documentation is more detailed
- if evidence for one comparison dimension exists for only one
  provider, clearly state that the current evidence does not support
  a direct comparison for that dimension

Do not refuse merely because different providers use different
terminology.
""".strip()


        rules = (

            rules
            +
            "\n\n"
            +
            comparison_rules

        )


    prompt = f"""
{rules}

QUESTION:

{question}


EVIDENCE:

{context}


ANSWER:
""".strip()


    return prompt


# ============================================================
# ADJACENT DUPLICATE CITATION CLEANUP
# ============================================================

INLINE_CITATION_PATTERN = re.compile(

    r"("
    r"\[[^\[\]\n]+"
    r"\|[^\[\]\n]+"
    r"\| PDF page \d+\]"
    r")"
    r"(?:\s*\1)+"

)


def deduplicate_adjacent_inline_citations(

    answer,

):

    if not answer:

        return answer


    previous_answer = None

    cleaned_answer = answer


    while (

        previous_answer
        !=
        cleaned_answer

    ):

        previous_answer = (

            cleaned_answer

        )


        cleaned_answer = (

            INLINE_CITATION_PATTERN.sub(

                r"\1",

                cleaned_answer,

            )

        )


    return cleaned_answer


# ============================================================
# SAVE PRE-POLISH REPLACEMENT FUNCTION
# ============================================================

_V13_PRE_POLISH_REPLACE_SOURCE_MARKERS = (

    replace_source_markers

)


# ============================================================
# TRUSTED REPLACEMENT + INLINE DEDUP
# ============================================================

def replace_source_markers(

    answer,

    evidence,

):

    (
        trusted_answer,
        citations,

    ) = (

        _V13_PRE_POLISH_REPLACE_SOURCE_MARKERS(

            answer=
                answer,

            evidence=
                evidence,

        )

    )


    trusted_answer = (

        deduplicate_adjacent_inline_citations(

            trusted_answer

        )

    )


    return (

        trusted_answer,

        citations,

    )






# ============================================================
# V13 CELL 76 CITATION INTEGRITY HARDENING
# ============================================================

COMBINED_SOURCE_MARKER_PATTERN = re.compile(

    r"\[SOURCE\s+([^\]]+)\]",

    re.IGNORECASE,

)


STRICT_SOURCE_MARKER_PATTERN = re.compile(

    r"\[SOURCE\s+(-?\d+)\]",

    re.IGNORECASE,

)


ANY_SOURCE_MARKER_PATTERN = re.compile(

    r"\[SOURCE[^\]]*\]",

    re.IGNORECASE,

)


def normalize_source_markers(

    answer,

):

    if not answer:

        return answer


    def normalize_match(
        match,
    ):

        content = (

            match.group(
                1
            )

        )


        numbers = (

            re.findall(

                r"-?\d+",

                content,

            )

        )


        if not numbers:

            return match.group(
                0
            )


        return "".join(

            f"[SOURCE {number}]"

            for number in numbers

        )


    return (

        COMBINED_SOURCE_MARKER_PATTERN.sub(

            normalize_match,

            answer,

        )

    )


def validate_source_numbers(

    answer,

    evidence,

):

    normalized_answer = (

        normalize_source_markers(
            answer
        )

    )


    source_numbers = [

        int(
            match.group(
                1
            )
        )

        for match in (

            STRICT_SOURCE_MARKER_PATTERN
            .finditer(
                normalized_answer
            )

        )

    ]


    all_source_markers = (

        ANY_SOURCE_MARKER_PATTERN.findall(

            normalized_answer

        )

    )


    malformed_markers = [

        marker

        for marker in all_source_markers

        if not (

            STRICT_SOURCE_MARKER_PATTERN
            .fullmatch(
                marker
            )

        )

    ]


    valid_source_numbers = []

    invalid_source_numbers = []


    for source_number in source_numbers:

        if (

            1
            <=
            source_number
            <=
            len(
                evidence
            )

        ):

            if (

                source_number
                not in
                valid_source_numbers

            ):

                valid_source_numbers.append(
                    source_number
                )


        else:

            if (

                source_number
                not in
                invalid_source_numbers

            ):

                invalid_source_numbers.append(
                    source_number
                )


    if malformed_markers:

        if 0 not in invalid_source_numbers:

            invalid_source_numbers.append(
                0
            )


    return (

        valid_source_numbers,

        invalid_source_numbers,

    )


def replace_source_markers(

    answer,

    evidence,

):

    normalized_answer = (

        normalize_source_markers(
            answer
        )

    )


    citations_used = []


    def replace_match(
        match,
    ):

        source_number = (

            int(
                match.group(
                    1
                )
            )

        )


        evidence_index = (

            source_number
            -
            1

        )


        if (

            0
            <=
            evidence_index
            <
            len(
                evidence
            )

        ):

            citation = (

                create_citation_label(

                    evidence[
                        evidence_index
                    ]

                )

            )


            if citation not in citations_used:

                citations_used.append(
                    citation
                )


            return citation


        return "[citation unavailable]"


    trusted_answer = (

        STRICT_SOURCE_MARKER_PATTERN.sub(

            replace_match,

            normalized_answer,

        )

    )


    trusted_answer = (

        ANY_SOURCE_MARKER_PATTERN.sub(

            "[citation unavailable]",

            trusted_answer,

        )

    )


    if (

        "deduplicate_adjacent_inline_citations"
        in globals()

    ):

        trusted_answer = (

            deduplicate_adjacent_inline_citations(

                trusted_answer

            )

        )


    return (

        trusted_answer,

        citations_used,

    )


_V13_PRE_CITATION_HARDENING_PROMPT = (

    build_grounded_prompt

)


def build_grounded_prompt(

    question,

    evidence,

    required_providers=None,

):

    original_prompt = (

        _V13_PRE_CITATION_HARDENING_PROMPT(

            question=
                question,

            evidence=
                evidence,

            required_providers=
                required_providers,

        )

    )


    citation_rule = """

STRICT CITATION FORMAT:

When citing more than one source, write each source marker separately.

CORRECT:

[SOURCE 1][SOURCE 2]

INCORRECT:

[SOURCE 1, SOURCE 2]

INCORRECT:

[SOURCE 1, 2]

Never use negative source numbers.
Never use a source number that is not present in the supplied evidence.
""".strip()


    if "\n\nANSWER:" in original_prompt:

        return (

            original_prompt.replace(

                "\n\nANSWER:",

                "\n\n"
                +
                citation_rule
                +
                "\n\nANSWER:",

                1,

            )

        )


    return (

        original_prompt
        +
        "\n\n"
        +
        citation_rule

    )






# ============================================================
# V13 CELL 77 COMPARISON-AWARE GLOBAL ROUTING
# ============================================================

V13_PROVIDERS = [

    "AWS",

    "Azure",

    "GCP",

    "Oracle",

    "IBM",

]


PROVIDER_PATTERNS = {

    "AWS": [

        r"\baws\b",

        r"\bamazon web services\b",

    ],

    "Azure": [

        r"\bazure\b",

        r"\bmicrosoft azure\b",

    ],

    "GCP": [

        r"\bgcp\b",

        r"\bgoogle cloud\b",

        r"\bgoogle cloud platform\b",

    ],

    "Oracle": [

        r"\boracle cloud\b",

        r"\boci\b",

    ],

    "IBM": [

        r"\bibm cloud\b",

        r"\bibm\b",

    ],

}


COMPARISON_PHRASES = [

    "compare",

    "comparison",

    "compared to",

    "compared with",

    "versus",

    " vs ",

    "side by side",

    "difference between",

    "differences between",

]


PROVIDER_SCOPE_PHRASES = [

    "cloud provider",

    "cloud providers",

    "other cloud",

    "different cloud",

    "across clouds",

    "across cloud",

    "multi-cloud",

    "multicloud",

    "other provider",

    "other providers",

    "different providers",

]


def extract_cloud_providers(

    text,

):

    if not text:

        return []


    providers = []


    for provider in V13_PROVIDERS:

        for pattern in PROVIDER_PATTERNS[
            provider
        ]:

            if re.search(

                pattern,

                text,

                re.IGNORECASE,

            ):

                providers.append(
                    provider
                )

                break


    return providers


def has_comparison_language(

    question,

):

    question_lower = (

        question.lower()

    )


    return any(

        phrase
        in
        question_lower

        for phrase in COMPARISON_PHRASES

    )


def has_provider_scope_language(

    question,

):

    question_lower = (

        question.lower()

    )


    return any(

        phrase
        in
        question_lower

        for phrase in PROVIDER_SCOPE_PHRASES

    )


def get_latest_provider_context(

    messages,

):

    if not messages:

        return []


    for message in reversed(
        messages
    ):

        if isinstance(
            message,
            dict,
        ):

            content = (

                message.get(
                    "content",
                    ""
                )

            )

        else:

            content = str(
                message
            )


        providers = (

            extract_cloud_providers(
                content
            )

        )


        if providers:

            return providers


    return []


def resolve_global_comparison_route(

    question,

    messages=None,

):

    messages = (

        messages

        or []

    )


    current_providers = (

        extract_cloud_providers(
            question
        )

    )


    previous_providers = (

        get_latest_provider_context(
            messages
        )

    )


    comparison_language = (

        has_comparison_language(
            question
        )

    )


    provider_scope_language = (

        has_provider_scope_language(
            question
        )

    )


    question_lower = (

        question.lower()

    )


    if (

        len(
            current_providers
        )
        >=
        2

        and

        comparison_language

    ):

        return {

            "auto_route":
                True,

            "mode":
                "cross_provider",

            "providers":
                current_providers,

            "reason":
                "Multiple providers explicitly named.",

        }


    if (

        len(
            current_providers
        )
        ==
        1

        and

        comparison_language

    ):

        combined = []


        for provider in (

            previous_providers
            +
            current_providers

        ):

            if provider not in combined:

                combined.append(
                    provider
                )


        if len(
            combined
        ) >= 2:

            return {

                "auto_route":
                    True,

                "mode":
                    "cross_provider",

                "providers":
                    combined,

                "reason":
                    "Current provider combined with conversation context.",

            }


        if provider_scope_language:

            return {

                "auto_route":
                    True,

                "mode":
                    "cross_provider",

                "providers":
                    list(
                        V13_PROVIDERS
                    ),

                "reason":
                    "Generic multi-cloud comparison requested.",

            }


    if (

        provider_scope_language

        and

        (

            comparison_language

            or

            "other"
            in
            question_lower

            or

            "different"
            in
            question_lower

        )

    ):

        return {

            "auto_route":
                True,

            "mode":
                "cross_provider",

            "providers":
                list(
                    V13_PROVIDERS
                ),

            "reason":
                "Generic multi-cloud comparison requested.",

        }


    if (

        comparison_language

        and

        len(
            previous_providers
        )
        >=
        2

    ):

        return {

            "auto_route":
                True,

            "mode":
                "cross_provider",

            "providers":
                previous_providers,

            "reason":
                "Comparison resolved from conversation context.",

        }


    return {

        "auto_route":
            False,

        "mode":
            "global",

        "providers":
            [],

        "reason":
            "Normal global retrieval.",

    }


# ============================================================
# PRESERVE THE ALREADY-TESTED V13 ORCHESTRATOR
# ============================================================

_V13_PRE_GLOBAL_ROUTING_ASK = (

    ask_v13_with_memory

)


# ============================================================
# ROUTING-AWARE ORCHESTRATOR
# ============================================================

def ask_v13_with_memory(

    conversation_id,

    question,

    mode="global",

    provider=None,

    providers=None,

    document_id=None,

):

    requested_mode = (

        mode

    )


    resolved_mode = (

        mode

    )


    resolved_provider = (

        provider

    )


    resolved_providers = (

        list(
            providers
        )

        if providers

        else []

    )


    auto_route = False

    routing_reason = (

        "User-selected retrieval mode."

    )


    # ========================================================
    # ONLY AUTO-ROUTE WHEN USER SELECTED GLOBAL MODE
    # ========================================================

    if mode == "global":

        previous_messages = (

            load_messages(

                conversation_id,

                DEFAULT_MEMORY_DB,

            )

        )


        route = (

            resolve_global_comparison_route(

                question=
                    question,

                messages=
                    previous_messages,

            )

        )


        if route[
            "auto_route"
        ]:

            auto_route = True


            resolved_mode = (

                route[
                    "mode"
                ]

            )


            resolved_provider = None


            resolved_providers = (

                route[
                    "providers"
                ]

            )


            routing_reason = (

                route[
                    "reason"
                ]

            )


    # ========================================================
    # CALL THE EXISTING TESTED PIPELINE
    # ========================================================

    result = (

        _V13_PRE_GLOBAL_ROUTING_ASK(

            conversation_id=
                conversation_id,

            question=
                question,

            mode=
                resolved_mode,

            provider=
                resolved_provider,

            providers=
                resolved_providers,

            document_id=
                document_id,

        )

    )


    # ========================================================
    # ATTACH ROUTING METADATA
    # ========================================================

    result[
        "requested_mode"
    ] = requested_mode


    result[
        "resolved_mode"
    ] = resolved_mode


    result[
        "auto_routed"
    ] = auto_route


    result[
        "routing_reason"
    ] = routing_reason


    result[
        "routing_providers"
    ] = resolved_providers


    return result


