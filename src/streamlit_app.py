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
    load_and_chunk_pdf,
    build_vector_database,
    ask_rag,
)


# ============================================================
# RAG V7 - STREAMLIT UI
#
# Browser UI
#   -> User question
#   -> Validated V5.2/V6 RAG backend
#   -> Grounded answer
#   -> Retrieved source details
#
# The PDF, embeddings, ChromaDB, reranker, and Ollama
# all remain local.
# ============================================================


# ============================================================
# 1. PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="MBA RAG Assistant",
    page_icon="📚",
    layout="wide",
)


# ============================================================
# 2. TITLE
# ============================================================

st.title("📚 MBA RAG Assistant")

st.caption(
    "Local Retrieval-Augmented Generation with MiniLM, ChromaDB, "
    "CrossEncoder reranking, neighbor expansion, and Ollama."
)


# ============================================================
# 3. INITIALIZE RAG ONCE
#
# Streamlit reruns the script after user interactions.
# @st.cache_resource prevents reloading models and rebuilding
# the vector database on every rerun.
# ============================================================

@st.cache_resource(show_spinner=False)
def initialize_rag():

    if not PDF_FILE.exists():

        raise FileNotFoundError(
            "\nPrivate PDF not found.\n"
            f"Place it locally at:\n{PDF_FILE}\n"
        )


    # --------------------------------------------------------
    # Embedding model
    # --------------------------------------------------------

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL
    )


    # --------------------------------------------------------
    # CrossEncoder reranker
    # --------------------------------------------------------

    reranker = CrossEncoder(
        RERANKER_MODEL
    )


    # --------------------------------------------------------
    # Local Ollama LLM
    # --------------------------------------------------------

    llm = ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0,
    )


    # --------------------------------------------------------
    # Load + chunk PDF
    # --------------------------------------------------------

    rag_chunks = load_and_chunk_pdf()


    # --------------------------------------------------------
    # Build neighbor lookup
    # --------------------------------------------------------

    rag_chunk_lookup = {

        chunk.metadata[
            "rag_chunk_number"
        ]:
            chunk

        for chunk in rag_chunks

    }


    # --------------------------------------------------------
    # Build local ChromaDB collection
    # --------------------------------------------------------

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


# ============================================================
# 4. SOURCE DISPLAY FUNCTION
# ============================================================

def show_sources(sources):


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
                f"**Chapter:** "
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


# ============================================================
# 5. SIDEBAR
# ============================================================

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
        f"**Chunk size:** {CHUNK_SIZE}"
    )


    st.write(
        f"**Chunk overlap:** {CHUNK_OVERLAP}"
    )


    st.write(
        f"**Semantic candidates:** {RETRIEVE_K}"
    )


    st.write(
        f"**Neighbor range:** ±{NEIGHBOR_RANGE}"
    )


    st.write(
        f"**Final context chunks:** {FINAL_K}"
    )


    st.divider()


    st.info(
        "The private PDF stays on your computer. "
        "It is not committed to GitHub."
    )


    if st.button(
        "Clear chat"
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# 6. LOAD LOCAL RAG
# ============================================================

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


# ============================================================
# 7. READY STATUS
# ============================================================

st.success(
    f"RAG ready — "
    f"{len(rag['rag_chunks'])} indexed chunks."
)


# ============================================================
# 8. EXAMPLE QUESTIONS
# ============================================================

st.subheader(
    "Example questions"
)


example_questions = [

    "What are my five CliftonStrengths?",

    "What two ethical traits did I identify in Satya Nadella?",

    "What technologies were proposed for the Agentic RAG architecture?",

    "What relationship was examined in the linear regression analysis?",

]


selected_example = st.selectbox(

    "Choose an example or type your own question below:",

    [
        "Select an example..."
    ]
    + example_questions,

)


# ============================================================
# 9. CHAT HISTORY
# ============================================================

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
            ]
            == "assistant"
            and
            message.get(
                "sources"
            )
        ):


            show_sources(
                message[
                    "sources"
                ]
            )


# ============================================================
# 10. USER QUESTION
# ============================================================

question = st.chat_input(
    "Ask a question about the MBA portfolio..."
)


# ------------------------------------------------------------
# Use selected example when the chat box itself is empty.
# ------------------------------------------------------------

if (
    question is None
    and
    selected_example
    != "Select an example..."
):


    question = selected_example


# ============================================================
# 11. RUN RAG
# ============================================================

if question:


    # --------------------------------------------------------
    # User message
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Assistant answer
    # --------------------------------------------------------

    with st.chat_message(
        "assistant"
    ):


        with st.spinner(
            "Searching the document and generating an answer..."
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


    # --------------------------------------------------------
    # Store assistant answer + evidence
    # --------------------------------------------------------

    st.session_state.messages.append(

        {

            "role":
                "assistant",

            "content":
                answer,

            "sources":
                sources,

        }

    )
