import streamlit as st

from rag_v12_pipeline import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CROSS_PROVIDER_FINAL_TOP_K,
    CROSS_PROVIDER_SEMANTIC_TOP_K,
    EMBEDDING_MODEL,
    EXPECTED_PROVIDERS,
    FINAL_TOP_K,
    MEMORY_MAX_MESSAGES,
    NEIGHBOR_RANGE,
    OLLAMA_MODEL,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
    RERANKER_MODEL,
    SEMANTIC_TOP_K,
    ask_rag,
    evidence_for_memory,
    initialize_rag,
)

from memory_db import (
    MEMORY_DB_PATH,
    auto_title_conversation,
    clear_conversation,
    count_messages,
    create_new_chat,
    delete_conversation,
    get_conversation,
    get_or_create_latest_conversation,
    initialize_database,
    list_conversations,
    load_messages,
    load_recent_memory,
    rename_conversation,
    save_message,
)


# ============================================================
# RAG V12 - MULTI-CLOUD STREAMLIT APPLICATION
# ============================================================

st.set_page_config(
    page_title="Multi-Cloud RAG Assistant V12",
    page_icon="☁️",
    layout="wide",
)

st.title("☁️ Multi-Cloud RAG Assistant V12")

st.caption(
    "AWS • Azure • GCP • Oracle • IBM | "
    "Pinecone + MiniLM + CrossEncoder + Ollama + SQLite Memory"
)


# ============================================================
# STATUS MESSAGES
# ============================================================

if "rename_success_message" in st.session_state:
    st.success(
        st.session_state.pop(
            "rename_success_message"
        )
    )


# ============================================================
# INITIALIZE SQLITE
# ============================================================

initialize_database()


# ============================================================
# LOAD V12 ONCE
# ============================================================

@st.cache_resource(show_spinner=False)
def load_rag():
    return initialize_rag()


try:
    with st.spinner(
        "Loading V12 multi-cloud RAG..."
    ):
        rag = load_rag()

except Exception as error:
    st.error(
        "RAG V12 could not start."
    )
    st.exception(error)
    st.stop()


# ============================================================
# ACTIVE CONVERSATION
# ============================================================

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = (
        get_or_create_latest_conversation()
    )


# ============================================================
# HELPERS
# ============================================================

def switch_conversation(conversation_id):
    st.session_state.conversation_id = conversation_id

    if "conversation_selector" in st.session_state:
        del st.session_state["conversation_selector"]


def source_display_values(source):
    # V12 source format.
    if "provider" in source:
        return {
            "provider": source.get("provider", ""),
            "document_title": source.get("document_title", ""),
            "source": source.get("source", ""),
            "pdf_page": source.get("pdf_page", ""),
            "semantic_score": source.get("semantic_score"),
            "rerank_score": source.get("rerank_score"),
            "text": source.get("text", ""),
        }

    # Backward compatibility for older V10/V11 saved messages.
    metadata = source.get("metadata", {})

    return {
        "provider": metadata.get("course", "Legacy"),
        "document_title": metadata.get("chapter", "Legacy source"),
        "source": metadata.get("source", ""),
        "pdf_page": metadata.get("pdf_page", ""),
        "semantic_score": source.get("vector_score"),
        "rerank_score": source.get("reranker_score"),
        "text": source.get("document", ""),
    }


def show_sources(sources):
    if not sources:
        return

    with st.expander(
        f"View retrieved evidence ({len(sources)})"
    ):
        for i, source in enumerate(
            sources,
            start=1,
        ):
            item = source_display_values(source)

            st.markdown(
                f"### Source {i}"
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.write(
                    "**Provider:**",
                    item["provider"],
                )

            with col2:
                st.write(
                    "**PDF page:**",
                    item["pdf_page"],
                )

            with col3:
                rerank_score = item[
                    "rerank_score"
                ]

                if rerank_score is not None:
                    st.write(
                        "**Rerank:**",
                        round(
                            rerank_score,
                            4,
                        ),
                    )

            st.write(
                "**Document:**",
                item["document_title"],
            )

            st.write(
                "**File:**",
                item["source"],
            )

            semantic_score = item[
                "semantic_score"
            ]

            if semantic_score is not None:
                st.write(
                    "**Semantic score:**",
                    round(
                        semantic_score,
                        4,
                    ),
                )

            st.text(
                item["text"]
            )

            if i < len(sources):
                st.divider()


def document_label(document):
    return (
        f"{document['provider']} | "
        f"{document['document_title']}"
    )


# ============================================================
# SIDEBAR - CONVERSATIONS
# ============================================================

with st.sidebar:
    st.header("💬 Conversations")

    if st.button(
        "➕ New Conversation",
        use_container_width=True,
    ):
        new_conversation_id = create_new_chat()
        switch_conversation(new_conversation_id)
        st.rerun()

    st.divider()

    conversations = list_conversations()

    conversation_map = {
        conversation["id"]: conversation
        for conversation in conversations
    }

    conversation_ids = [
        conversation["id"]
        for conversation in conversations
    ]

    if conversation_ids:
        current_id = st.session_state.conversation_id

        if current_id not in conversation_ids:
            current_id = conversation_ids[0]
            st.session_state.conversation_id = current_id

        current_index = conversation_ids.index(
            current_id
        )

        selected_conversation_id = st.selectbox(
            "Previous conversations",
            options=conversation_ids,
            index=current_index,
            format_func=lambda conversation_id: (
                conversation_map[
                    conversation_id
                ]["title"]
            ),
            key="conversation_selector",
        )

        if (
            selected_conversation_id
            != st.session_state.conversation_id
        ):
            st.session_state.conversation_id = (
                selected_conversation_id
            )
            st.rerun()

    st.divider()

    current_conversation_id = (
        st.session_state.conversation_id
    )

    current_conversation = get_conversation(
        current_conversation_id
    )

    if current_conversation is None:
        current_conversation_id = create_new_chat()
        switch_conversation(current_conversation_id)
        current_conversation = get_conversation(
            current_conversation_id
        )

    st.write("**Current conversation**")
    st.write(current_conversation["title"])

    st.caption(
        f"{count_messages(current_conversation_id)} "
        "stored messages"
    )

    rename_input_key = (
        "rename_input_"
        + current_conversation_id
    )

    rename_form_key = (
        "rename_form_"
        + current_conversation_id
    )

    with st.form(
        key=rename_form_key,
        clear_on_submit=False,
    ):
        rename_value = st.text_input(
            "Rename conversation",
            value=current_conversation[
                "title"
            ],
            key=rename_input_key,
        )

        rename_submitted = (
            st.form_submit_button(
                "Rename",
                use_container_width=True,
            )
        )

    if rename_submitted:
        new_title = rename_value.strip()

        if not new_title:
            st.warning(
                "Conversation title cannot be empty."
            )

        elif (
            new_title
            == current_conversation["title"]
        ):
            st.info(
                "The conversation already has that title."
            )

        else:
            rename_conversation(
                current_conversation_id,
                new_title,
            )

            st.session_state[
                "rename_success_message"
            ] = (
                f"Conversation renamed to: {new_title}"
            )

            st.rerun()

    if st.button(
        "🧹 Clear Conversation",
        use_container_width=True,
    ):
        clear_conversation(
            current_conversation_id
        )
        st.rerun()

    if st.button(
        "🗑️ Delete Conversation",
        use_container_width=True,
    ):
        delete_conversation(
            current_conversation_id
        )

        remaining = list_conversations()

        if remaining:
            next_id = remaining[0]["id"]
        else:
            next_id = create_new_chat()

        switch_conversation(next_id)
        st.rerun()

    st.divider()


    # ========================================================
    # RETRIEVAL CONTROLS
    # ========================================================

    st.header("🔎 Retrieval")

    retrieval_mode = st.selectbox(
        "Mode",
        options=[
            "Auto",
            "All Providers",
            "Single Provider",
            "Compare Providers",
            "Specific Document",
        ],
        help=(
            "Auto detects provider names after "
            "conversation-memory rewriting."
        ),
    )

    selected_provider = None
    selected_providers = []
    selected_document_id = None

    if retrieval_mode == "Single Provider":
        selected_provider = st.selectbox(
            "Provider",
            options=EXPECTED_PROVIDERS,
        )

    elif retrieval_mode == "Compare Providers":
        selected_providers = st.multiselect(
            "Providers to compare",
            options=EXPECTED_PROVIDERS,
            default=["AWS", "Azure"],
        )

        if len(selected_providers) < 2:
            st.warning(
                "Select at least two providers."
            )

    elif retrieval_mode == "Specific Document":
        document_provider = st.selectbox(
            "Document provider",
            options=EXPECTED_PROVIDERS,
        )

        provider_documents = [
            document
            for document in rag[
                "document_catalog"
            ]
            if document["provider"]
            == document_provider
        ]

        document_map = {
            document["document_id"]: document
            for document in provider_documents
        }

        selected_document_id = st.selectbox(
            "Document",
            options=list(
                document_map.keys()
            ),
            format_func=lambda document_id: (
                document_label(
                    document_map[document_id]
                )
            ),
        )

    st.divider()


    # ========================================================
    # MEMORY STATUS
    # ========================================================

    st.subheader("Persistent Memory")

    st.write(
        "**Storage:** SQLite"
    )

    st.write(
        "**Recent messages sent to Ollama:**",
        MEMORY_MAX_MESSAGES,
    )

    st.code(
        str(MEMORY_DB_PATH)
    )

    st.success(
        "Conversation history survives app restarts."
    )

    st.divider()


    # ========================================================
    # RAG CONFIGURATION
    # ========================================================

    with st.expander(
        "RAG Configuration"
    ):
        st.write("**Documents**")
        st.code(str(len(rag["pdf_records"])))

        st.write("**Pages**")
        st.code(str(len(rag["pages"])))

        st.write("**Chunks**")
        st.code(str(len(rag["chunks"])))

        st.write("**Embedding model**")
        st.code(EMBEDDING_MODEL)

        st.write("**Reranker**")
        st.code(RERANKER_MODEL)

        st.write("**Generation model**")
        st.code(OLLAMA_MODEL)

        st.write(
            f"Chunk size: {CHUNK_SIZE}"
        )

        st.write(
            f"Chunk overlap: {CHUNK_OVERLAP}"
        )

        st.write(
            f"Semantic Top K: {SEMANTIC_TOP_K}"
        )

        st.write(
            f"Neighbor range: ±{NEIGHBOR_RANGE}"
        )

        st.write(
            f"Final Top K: {FINAL_TOP_K}"
        )

        st.write(
            "Cross-provider candidates/provider: "
            f"{CROSS_PROVIDER_SEMANTIC_TOP_K}"
        )

        st.write(
            "Cross-provider final/provider: "
            f"{CROSS_PROVIDER_FINAL_TOP_K}"
        )

        st.write("**Vector database**")
        st.code("Pinecone")

        st.write("**Pinecone index**")
        st.code(PINECONE_INDEX_NAME)

        st.write("**Pinecone namespace**")
        st.code(PINECONE_NAMESPACE)


# ============================================================
# CURRENT CONVERSATION
# ============================================================

current_conversation_id = (
    st.session_state.conversation_id
)

current_conversation = get_conversation(
    current_conversation_id
)

if current_conversation is None:
    current_conversation_id = create_new_chat()
    switch_conversation(
        current_conversation_id
    )
    current_conversation = get_conversation(
        current_conversation_id
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.subheader(
    current_conversation["title"]
)

st.caption(
    f"Conversation ID: {current_conversation_id}"
)

st.success(
    "RAG V12 ready — "
    f"{len(rag['pdf_records'])} PDFs, "
    f"{len(rag['pages'])} pages, "
    f"{len(rag['chunks'])} chunks."
)


# ============================================================
# EXAMPLE QUESTIONS
# ============================================================

example_question = None

with st.expander(
    "Example Questions"
):
    example_questions = [
        (
            "What does AWS recommend for building "
            "reliable cloud workloads?"
        ),
        (
            "How does Azure compare with that?"
        ),
        (
            "Compare AWS and Azure guidance for "
            "building reliable cloud workloads."
        ),
        (
            "What identity and access management "
            "best practices does Oracle recommend?"
        ),
        (
            "How does Google Cloud protect "
            "cloud infrastructure and workloads?"
        ),
        (
            "What security principles apply to "
            "IBM hybrid cloud environments?"
        ),
    ]

    for i, example in enumerate(
        example_questions,
        start=1,
    ):
        if st.button(
            example,
            use_container_width=True,
            key=(
                f"example_{i}_"
                f"{current_conversation_id}"
            ),
        ):
            example_question = example


# ============================================================
# DISPLAY PERSISTED HISTORY
# ============================================================

messages = load_messages(
    current_conversation_id
)

for message in messages:
    with st.chat_message(
        message["role"]
    ):
        st.markdown(
            message["content"]
        )

        if message["role"] == "assistant":
            query_type = message.get(
                "query_type"
            )

            standalone_question = message.get(
                "standalone_question"
            )

            if query_type:
                st.caption(
                    f"Retrieval mode: {query_type}"
                )

            if standalone_question:
                with st.expander(
                    "Conversation memory / interpreted question"
                ):
                    st.write(
                        "**Standalone retrieval question:**"
                    )
                    st.write(
                        standalone_question
                    )

            show_sources(
                message.get(
                    "sources",
                    [],
                )
            )


# ============================================================
# USER INPUT
# ============================================================

question = st.chat_input(
    "Ask about AWS, Azure, GCP, Oracle, or IBM..."
)

if question is None and example_question is not None:
    question = example_question


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:
    conversation_history = load_recent_memory(
        current_conversation_id
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner(
            "Using conversation memory, retrieving evidence, "
            "reranking, and generating a grounded answer..."
        ):
            try:
                result = ask_rag(
                    rag=rag,
                    question=question,
                    conversation_history=(
                        conversation_history
                    ),
                    mode=retrieval_mode,
                    provider=selected_provider,
                    providers=selected_providers,
                    document_id=(
                        selected_document_id
                    ),
                )

            except Exception as error:
                st.error(
                    "The question could not be processed."
                )
                st.exception(error)
                st.stop()

        st.caption(
            "Retrieval mode: "
            f"{result['query_type']}"
        )

        if result["valid"]:
            st.markdown(
                result["answer"]
            )
        else:
            st.warning(
                "The generated answer failed "
                "citation validation."
            )
            st.markdown(
                result["answer"]
            )

        with st.expander(
            "Conversation memory / interpreted question"
        ):
            st.write("**Original question:**")
            st.write(question)

            st.write(
                "**Standalone retrieval question:**"
            )
            st.write(
                result[
                    "standalone_question"
                ]
            )

            st.write("**Resolved mode:**")
            st.write(
                result[
                    "resolved_mode"
                ]
            )

            if result[
                "resolved_provider"
            ]:
                st.write("**Provider:**")
                st.write(
                    result[
                        "resolved_provider"
                    ]
                )

            if result[
                "resolved_providers"
            ]:
                st.write(
                    "**Compared providers:**"
                )
                st.write(
                    ", ".join(
                        result[
                            "resolved_providers"
                        ]
                    )
                )

            st.write(
                "**Citation attempts:**",
                result["attempts"],
            )

        show_sources(
            result["evidence"]
        )

    # --------------------------------------------------------
    # SAVE USER MESSAGE
    # --------------------------------------------------------

    save_message(
        conversation_id=(
            current_conversation_id
        ),
        role="user",
        content=question,
        query_type=result[
            "query_type"
        ],
        standalone_question=result[
            "standalone_question"
        ],
        sources=[],
    )

    # --------------------------------------------------------
    # SAVE ASSISTANT MESSAGE + EVIDENCE
    # --------------------------------------------------------

    save_message(
        conversation_id=(
            current_conversation_id
        ),
        role="assistant",
        content=result["answer"],
        query_type=result[
            "query_type"
        ],
        standalone_question=result[
            "standalone_question"
        ],
        sources=evidence_for_memory(
            result["evidence"]
        ),
    )

    auto_title_conversation(
        current_conversation_id,
        question,
    )

    if "conversation_selector" in st.session_state:
        del st.session_state[
            "conversation_selector"
        ]

    st.rerun()