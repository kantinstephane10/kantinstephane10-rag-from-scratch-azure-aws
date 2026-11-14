import streamlit as st

from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_ollama import ChatOllama

from rag_pipeline import (
    PDF_FILE,
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
    load_and_chunk_pdf,
    build_vector_database,
    detect_query_type,
    ask_rag,
)


# ============================================================
# RAG V8 - STREAMLIT UI
# ============================================================


st.set_page_config(
    page_title="MBA RAG Assistant V8",
    page_icon="📚",
    layout="wide",
)


st.title(
    "📚 MBA RAG Assistant — V8"
)

st.caption(
    "Query-aware local RAG with MiniLM, ChromaDB, "
    "neighbor expansion, CrossEncoder reranking, "
    "Ollama, and deterministic citations."
)


@st.cache_resource(
    show_spinner=False
)
def initialize_rag():

    if not PDF_FILE.exists():

        raise FileNotFoundError(
            "\nMBA PDF not found.\n"
            f"Expected path:\n{PDF_FILE}\n"
        )

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

    return {
        "embedding_model":
            embedding_model,

        "reranker":
            reranker,

        "llm":
            llm,

        "rag_chunks":
            rag_chunks,

        "rag_chunk_lookup":
            rag_chunk_lookup,

        "collection":
            collection,
    }


def show_sources(
    sources,
):

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
                f"**Course:** "
                f"{metadata['course']}"
            )

            st.write(
                f"**Course title:** "
                f"{metadata['chapter']}"
            )

            st.write(
                f"**PDF page:** "
                f"{metadata['pdf_page']}"
            )

            st.write(
                f"**Retrieval source:** "
                f"{source['retrieval_source']}"
            )

            st.write(
                f"**Reranker score:** "
                f"{source['reranker_score']:.4f}"
            )

            st.text(
                source[
                    "document"
                ]
            )

            if i < len(
                sources
            ):

                st.divider()


with st.sidebar:

    st.header(
        "RAG Configuration"
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
        f"**Focused semantic candidates:** "
        f"{RETRIEVE_K}"
    )

    st.write(
        f"**Neighbor range:** "
        f"±{NEIGHBOR_RANGE}"
    )

    st.write(
        f"**Focused final context:** "
        f"{FINAL_K}"
    )

    st.divider()

    st.write(
        "**Broad retrieval**"
    )

    st.write(
        f"{BROAD_PER_COURSE_K} semantic candidates "
        "per course"
    )

    st.write(
        f"{BROAD_FINAL_PER_COURSE} final chunks "
        "per course"
    )

    st.divider()

    st.info(
        "V8 automatically routes focused questions "
        "and broad portfolio questions through "
        "different retrieval strategies."
    )

    if st.button(
        "Clear chat"
    ):

        st.session_state.messages = []

        st.rerun()


try:

    with st.spinner(
        "Loading local RAG system..."
    ):

        rag = initialize_rag()

except Exception as error:

    st.error(
        "The RAG system could not start."
    )

    st.exception(
        error
    )

    st.stop()


st.success(
    f"RAG V8 ready — "
    f"{len(rag['rag_chunks'])} indexed chunks."
)


st.subheader(
    "Example questions"
)

example_questions = [
    "What are my five CliftonStrengths?",
    "What technologies were proposed for the Agentic RAG architecture?",
    (
        "Identify all courses in my MBA portfolio that discuss "
        "project management. For each course, provide the course code, "
        "course title, how project management is discussed, "
        "and the PDF page number."
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


if (
    "messages"
    not in st.session_state
):

    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(
        message[
            "role"
        ]
    ):

        st.markdown(
            message[
                "content"
            ]
        )

        if (
            message[
                "role"
            ] == "assistant"
        ):

            query_type = message.get(
                "query_type"
            )

            if query_type:

                st.caption(
                    f"Retrieval mode: "
                    f"{query_type}"
                )

            if message.get(
                "sources"
            ):

                show_sources(
                    message[
                        "sources"
                    ]
                )


question = st.chat_input(
    "Ask a question about the MBA portfolio..."
)

if (
    question is None
    and
    example_question is not None
):

    question = example_question


if question:

    query_type = detect_query_type(
        question
    )

    st.session_state.messages.append(
        {
            "role":
                "user",

            "content":
                question,
        }
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

        st.caption(
            f"Retrieval mode: "
            f"{query_type}"
        )

        with st.spinner(
            "Searching the portfolio and generating an answer..."
        ):

            answer, sources = ask_rag(
                question,
                rag[
                    "collection"
                ],
                rag[
                    "embedding_model"
                ],
                rag[
                    "reranker"
                ],
                rag[
                    "rag_chunk_lookup"
                ],
                rag[
                    "llm"
                ],
            )

        st.markdown(
            answer
        )

        show_sources(
            sources
        )

    st.session_state.messages.append(
        {
            "role":
                "assistant",

            "content":
                answer,

            "query_type":
                query_type,

            "sources":
                sources,
        }
    )
