import streamlit as st

from rag_pipeline import (
    EMBEDDING_MODEL,
    RERANKER_MODEL,
    OLLAMA_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    RETRIEVE_K,
    FINAL_K,
    NEIGHBOR_RANGE,
    BROAD_PER_COURSE_K,
    BROAD_FINAL_PER_COURSE,
    MEMORY_MAX_MESSAGES,
    initialize_rag,
    ask_rag,
)

from memory_db import (
    MEMORY_DB_PATH,
    initialize_database,
    list_conversations,
    get_conversation,
    get_or_create_latest_conversation,
    create_new_chat,
    rename_conversation,
    clear_conversation,
    delete_conversation,
    load_messages,
    load_recent_memory,
    save_message,
    count_messages,
    auto_title_conversation,
)


# ============================================================
# RAG V10 - STREAMLIT PERSISTENT CONVERSATIONAL MEMORY
#
# V9 RAG backend stays unchanged.
#
# V10 adds:
# - SQLite persistent memory
# - multiple conversations
# - new/resume chat
# - automatic chat titles
# - rename
# - clear
# - delete
# - restart persistence
# ============================================================


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="MBA RAG Assistant V10",
    page_icon="📚",
    layout="wide",
)


st.title(
    "📚 MBA RAG Assistant V10"
)

st.caption(
    "Persistent SQLite Memory + Multi-Conversation Chat + "
    "Query-Aware Retrieval + CrossEncoder Reranking + "
    "Deterministic Citations"
)


# ============================================================
# INITIALIZE PERSISTENT MEMORY
# ============================================================

initialize_database()


# ============================================================
# LOAD RAG ONCE
# ============================================================

@st.cache_resource(
    show_spinner=False
)
def load_rag():
    return initialize_rag()


try:
    with st.spinner(
        "Loading RAG V10..."
    ):
        rag = load_rag()

except Exception as error:
    st.error(
        "RAG V10 could not start."
    )
    st.exception(error)
    st.stop()


# ============================================================
# ACTIVE CONVERSATION
# ============================================================

if (
    "conversation_id"
    not in st.session_state
):
    st.session_state.conversation_id = (
        get_or_create_latest_conversation()
    )


# ============================================================
# HELPERS
# ============================================================

def switch_conversation(
    conversation_id,
):
    st.session_state.conversation_id = (
        conversation_id
    )

    if (
        "conversation_selector"
        in st.session_state
    ):
        del st.session_state[
            "conversation_selector"
        ]


def show_sources(
    sources,
):
    if not sources:
        return

    with st.expander(
        "View retrieved sources"
    ):
        for i, source in enumerate(
            sources,
            start=1,
        ):
            metadata = source.get(
                "metadata",
                {},
            )

            st.markdown(
                f"### Source {i}"
            )

            st.write(
                "**Course:**",
                metadata.get(
                    "course",
                    "",
                ),
            )

            st.write(
                "**Course title:**",
                metadata.get(
                    "chapter",
                    "",
                ),
            )

            st.write(
                "**PDF page:**",
                metadata.get(
                    "pdf_page",
                    "",
                ),
            )

            st.write(
                "**Retrieval source:**",
                source.get(
                    "retrieval_source",
                    "",
                ),
            )

            reranker_score = source.get(
                "reranker_score"
            )

            if reranker_score is not None:
                st.write(
                    "**Reranker score:**",
                    round(
                        reranker_score,
                        4,
                    ),
                )

            st.text(
                source.get(
                    "document",
                    "",
                )
            )

            if i < len(sources):
                st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header(
        "💬 Conversations"
    )

    # --------------------------------------------------------
    # NEW CHAT
    # --------------------------------------------------------

    if st.button(
        "➕ New Conversation",
        use_container_width=True,
    ):
        new_conversation_id = (
            create_new_chat()
        )

        switch_conversation(
            new_conversation_id
        )

        st.rerun()


    # --------------------------------------------------------
    # SAVED CHATS
    # --------------------------------------------------------

    st.divider()

    conversations = (
        list_conversations()
    )

    conversation_map = {
        conversation["id"]:
            conversation
        for conversation
        in conversations
    }

    conversation_ids = [
        conversation["id"]
        for conversation
        in conversations
    ]

    if conversation_ids:
        current_id = (
            st.session_state.conversation_id
        )

        if current_id not in conversation_ids:
            current_id = (
                conversation_ids[0]
            )

            st.session_state.conversation_id = (
                current_id
            )

        current_index = (
            conversation_ids.index(
                current_id
            )
        )

        selected_conversation_id = (
            st.selectbox(
                "Previous conversations",
                options=conversation_ids,
                index=current_index,
                format_func=lambda conversation_id:
                    conversation_map[
                        conversation_id
                    ]["title"],
                key="conversation_selector",
            )
        )

        if (
            selected_conversation_id
            !=
            st.session_state.conversation_id
        ):
            st.session_state.conversation_id = (
                selected_conversation_id
            )

            st.rerun()


    # --------------------------------------------------------
    # CURRENT CHAT CONTROLS
    # --------------------------------------------------------

    st.divider()

    current_conversation_id = (
        st.session_state.conversation_id
    )

    current_conversation = (
        get_conversation(
            current_conversation_id
        )
    )

    if current_conversation is None:
        current_conversation_id = (
            create_new_chat()
        )

        switch_conversation(
            current_conversation_id
        )

        current_conversation = (
            get_conversation(
                current_conversation_id
            )
        )

    st.write(
        "**Current conversation**"
    )

    st.write(
        current_conversation[
            "title"
        ]
    )

    st.caption(
        f"{count_messages(current_conversation_id)} "
        "stored messages"
    )


    # --------------------------------------------------------
    # RENAME
    # --------------------------------------------------------

    rename_value = st.text_input(
        "Rename conversation",
        value=current_conversation[
            "title"
        ],
        key=(
            "rename_"
            + current_conversation_id
        ),
    )

    if st.button(
        "Rename",
        use_container_width=True,
    ):
        rename_conversation(
            current_conversation_id,
            rename_value,
        )

        if (
            "conversation_selector"
            in st.session_state
        ):
            del st.session_state[
                "conversation_selector"
            ]

        st.rerun()


    # --------------------------------------------------------
    # CLEAR
    # --------------------------------------------------------

    if st.button(
        "🧹 Clear Conversation",
        use_container_width=True,
    ):
        clear_conversation(
            current_conversation_id
        )

        st.rerun()


    # --------------------------------------------------------
    # DELETE
    # --------------------------------------------------------

    if st.button(
        "🗑️ Delete Conversation",
        use_container_width=True,
    ):
        delete_conversation(
            current_conversation_id
        )

        remaining = (
            list_conversations()
        )

        if remaining:
            next_id = (
                remaining[0]["id"]
            )
        else:
            next_id = (
                create_new_chat()
            )

        switch_conversation(
            next_id
        )

        st.rerun()


    # --------------------------------------------------------
    # MEMORY STATUS
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "Persistent Memory"
    )

    st.write(
        f"**Recent messages sent to LLM:** "
        f"{MEMORY_MAX_MESSAGES}"
    )

    st.write(
        "**Storage:** SQLite"
    )

    st.code(
        str(
            MEMORY_DB_PATH
        )
    )

    st.success(
        "Conversation history survives "
        "application restarts."
    )


    # --------------------------------------------------------
    # RAG SETTINGS
    # --------------------------------------------------------

    st.divider()

    with st.expander(
        "RAG Configuration"
    ):
        st.write(
            "**Embedding model**"
        )
        st.code(
            EMBEDDING_MODEL
        )

        st.write(
            "**Reranker**"
        )
        st.code(
            RERANKER_MODEL
        )

        st.write(
            "**Generation model**"
        )
        st.code(
            OLLAMA_MODEL
        )

        st.write(
            f"Chunk size: "
            f"{CHUNK_SIZE}"
        )

        st.write(
            f"Chunk overlap: "
            f"{CHUNK_OVERLAP}"
        )

        st.write(
            f"Focused candidates: "
            f"{RETRIEVE_K}"
        )

        st.write(
            f"Neighbor range: "
            f"±{NEIGHBOR_RANGE}"
        )

        st.write(
            f"Focused final context: "
            f"{FINAL_K}"
        )

        st.write(
            f"Broad candidates/course: "
            f"{BROAD_PER_COURSE_K}"
        )

        st.write(
            f"Broad final/course: "
            f"{BROAD_FINAL_PER_COURSE}"
        )

        st.write(
            "Course relevance gate: enabled"
        )


# ============================================================
# CURRENT CONVERSATION
# ============================================================

current_conversation_id = (
    st.session_state.conversation_id
)

current_conversation = (
    get_conversation(
        current_conversation_id
    )
)

if current_conversation is None:
    current_conversation_id = (
        create_new_chat()
    )

    switch_conversation(
        current_conversation_id
    )

    current_conversation = (
        get_conversation(
            current_conversation_id
        )
    )


# ============================================================
# HEADER
# ============================================================

st.subheader(
    current_conversation[
        "title"
    ]
)

st.caption(
    f"Conversation ID: "
    f"{current_conversation_id}"
)

st.success(
    f"RAG V10 ready — "
    f"{len(rag['rag_chunks'])} indexed chunks."
)


# ============================================================
# EXAMPLE QUESTIONS
# ============================================================

example_question = None

with st.expander(
    "Example Questions"
):
    example_questions = [
        "What are my five CliftonStrengths?",
        "Which one was ranked highest?",
        (
            "What technologies were proposed "
            "for the Agentic RAG architecture?"
        ),
        (
            "Which one provides the search "
            "and retrieval layer?"
        ),
        (
            "Identify all courses in my MBA portfolio "
            "that discuss project management."
        ),
        "What is Microsoft's current stock price?",
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
# DISPLAY PERSISTED CHAT HISTORY
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

        if (
            message["role"]
            ==
            "assistant"
        ):
            query_type = message.get(
                "query_type"
            )

            standalone_question = (
                message.get(
                    "standalone_question"
                )
            )

            if query_type:
                st.caption(
                    f"Retrieval mode: "
                    f"{query_type}"
                )

            if standalone_question:
                with st.expander(
                    "Conversation memory / "
                    "interpreted question"
                ):
                    st.write(
                        "**Standalone retrieval question:**"
                    )

                    st.write(
                        standalone_question
                    )

            sources = message.get(
                "sources"
            )

            if sources:
                show_sources(
                    sources
                )


# ============================================================
# USER INPUT
# ============================================================

question = st.chat_input(
    "Ask a question about the MBA portfolio..."
)

if (
    question is None
    and example_question is not None
):
    question = (
        example_question
    )


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:
    # SQLite is the source of truth.
    # Load only the most recent memory window for the LLM.
    conversation_history = (
        load_recent_memory(
            current_conversation_id
        )
    )

    with st.chat_message(
        "user"
    ):
        st.markdown(
            question
        )

    with st.chat_message(
        "assistant"
    ):
        with st.spinner(
            "Loading persistent memory and "
            "searching the MBA portfolio..."
        ):
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

        st.caption(
            f"Retrieval mode: "
            f"{query_type}"
        )

        st.markdown(
            answer
        )

        with st.expander(
            "Conversation memory / "
            "interpreted question"
        ):
            st.write(
                "**Original question:**"
            )

            st.write(
                question
            )

            st.write(
                "**Standalone retrieval question:**"
            )

            st.write(
                standalone_question
            )

        show_sources(
            sources
        )

    # Save current turn after generation so it becomes
    # memory for the NEXT question.
    save_message(
        conversation_id=
            current_conversation_id,
        role="user",
        content=question,
    )

    save_message(
        conversation_id=
            current_conversation_id,
        role="assistant",
        content=answer,
        query_type=query_type,
        standalone_question=
            standalone_question,
        sources=sources,
    )

    auto_title_conversation(
        current_conversation_id,
        question,
    )

    if (
        "conversation_selector"
        in st.session_state
    ):
        del st.session_state[
            "conversation_selector"
        ]

    st.rerun()
