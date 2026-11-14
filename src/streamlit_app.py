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


# ============================================================
# RAG V9 - STREAMLIT CONVERSATIONAL MEMORY
#
# Streamlit chat
#   -> session conversation history
#   -> follow-up question rewriting
#   -> query router
#   -> focused or broad retrieval
#   -> broad-query course relevance gate
#   -> CrossEncoder reranking
#   -> Ollama
#   -> deterministic Python citations
#   -> final answer
# ============================================================


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="MBA RAG Assistant V9",
    page_icon="📚",
    layout="wide",
)


st.title(
    "📚 MBA RAG Assistant V9"
)

st.caption(
    "Conversational Memory + Query-Aware Retrieval + "
    "Course Relevance Filtering + Neighbor Expansion + "
    "CrossEncoder Reranking + Deterministic Citations"
)


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
        "Loading RAG V9..."
    ):
        rag = load_rag()

except Exception as error:
    st.error(
        "RAG V9 could not start."
    )
    st.exception(error)
    st.stop()


# ============================================================
# SESSION MEMORY
# ============================================================

if (
    "messages"
    not in st.session_state
):
    st.session_state.messages = []


# ============================================================
# SOURCE DISPLAY
# ============================================================

def show_sources(sources):
    if not sources:
        return

    with st.expander(
        "View retrieved sources"
    ):
        for i, source in enumerate(
            sources,
            start=1,
        ):
            metadata = source[
                "metadata"
            ]

            st.markdown(
                f"### Source {i}"
            )

            st.write(
                "**Course:**",
                metadata["course"],
            )

            st.write(
                "**Course title:**",
                metadata["chapter"],
            )

            st.write(
                "**PDF page:**",
                metadata["pdf_page"],
            )

            st.write(
                "**Retrieval source:**",
                source[
                    "retrieval_source"
                ],
            )

            st.write(
                "**Reranker score:**",
                round(
                    source[
                        "reranker_score"
                    ],
                    4,
                ),
            )

            st.text(
                source["document"]
            )

            if i < len(sources):
                st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header(
        "RAG V9 Configuration"
    )

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

    st.divider()

    st.write(
        f"**Chunk size:** "
        f"{CHUNK_SIZE}"
    )

    st.write(
        f"**Chunk overlap:** "
        f"{CHUNK_OVERLAP}"
    )

    st.write(
        f"**Focused candidates:** "
        f"{RETRIEVE_K}"
    )

    st.write(
        f"**Neighbor range:** "
        f"±{NEIGHBOR_RANGE}"
    )

    st.write(
        f"**Focused final chunks:** "
        f"{FINAL_K}"
    )

    st.divider()

    st.write(
        "**Broad retrieval**"
    )

    st.write(
        f"{BROAD_PER_COURSE_K} "
        "candidates/course"
    )

    st.write(
        f"{BROAD_FINAL_PER_COURSE} "
        "final chunks/course"
    )

    st.write(
        "Course relevance gate: enabled"
    )

    st.divider()

    st.write(
        "**Conversation memory**"
    )

    st.write(
        f"Recent messages used: "
        f"{MEMORY_MAX_MESSAGES}"
    )

    st.write(
        f"Current chat messages: "
        f"{len(st.session_state.messages)}"
    )

    st.info(
        "V9 remembers recent conversation turns "
        "during the current Streamlit session."
    )

    if st.button(
        "Clear conversation"
    ):
        st.session_state.messages = []
        st.rerun()


# ============================================================
# READY STATUS
# ============================================================

st.success(
    f"RAG V9 ready — "
    f"{len(rag['rag_chunks'])} indexed chunks."
)


# ============================================================
# EXAMPLE QUESTIONS
# ============================================================

st.subheader(
    "Example Questions"
)

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

example_question = None

for example in example_questions:
    if st.button(
        example,
        use_container_width=True,
    ):
        example_question = example


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:
    with st.chat_message(
        message["role"]
    ):
        st.markdown(
            message["content"]
        )

        if (
            message["role"]
            == "assistant"
        ):
            query_type = message.get(
                "query_type"
            )

            standalone_question = message.get(
                "standalone_question"
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
    question = example_question


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:
    # Existing conversation only.
    # The current question is added after the answer.
    conversation_history = list(
        st.session_state.messages
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
            "Using conversation memory and searching "
            "the MBA portfolio..."
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
            "Conversation memory / interpreted question"
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

    # Save after generation so the current turn becomes
    # memory for the NEXT question.
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "query_type": query_type,
            "standalone_question":
                standalone_question,
            "sources": sources,
        }
    )
