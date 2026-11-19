"""Streamlit production UI for Multi-Cloud RAG V14."""

import streamlit as st

from rag_v14 import (
    EXPECTED_PROVIDERS,
    EXPECTED_TOTAL_CHUNKS,
    EXPECTED_TOTAL_PDFS,
    OLLAMA_MODEL,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
    REFUSAL_MESSAGE,
    ask_v14_with_memory,
    initialize_v14,
)

from memory_db import (
    MEMORY_DB_PATH,
    clear_conversation,
    count_messages,
    create_new_chat,
    delete_conversation,
    get_conversation,
    get_or_create_latest_conversation,
    initialize_database,
    list_conversations,
    load_messages,
    rename_conversation,
)


st.set_page_config(
    page_title="Multi-Cloud RAG V14",
    page_icon="☁️",
    layout="wide",
)

st.title("☁️ Multi-Cloud RAG V14")
st.caption(
    "V14 controlled metadata + local Dense/BM25 hybrid retrieval + "
    "CrossEncoder reranking + Ollama + persistent SQLite memory"
)

initialize_database()


@st.cache_resource(show_spinner=False)
def load_v14():
    return initialize_v14()


try:
    with st.spinner("Loading V14 backend..."):
        rag = load_v14()
except Exception as exc:
    st.error("V14 could not start.")
    st.exception(exc)
    st.stop()


if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = get_or_create_latest_conversation()


def switch_conversation(conversation_id):
    st.session_state.conversation_id = conversation_id
    if "conversation_selector" in st.session_state:
        del st.session_state["conversation_selector"]


def document_label(document):
    return f"{document['provider']} | {document['document_title']}"


def show_sources(sources):
    if not sources:
        return

    with st.expander(f"View retrieved evidence ({len(sources)})"):
        for index, source in enumerate(sources, start=1):
            st.markdown(f"### Source {index}")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("**Provider:**", source.get("provider", ""))
            with col2:
                st.write("**PDF page:**", source.get("pdf_page", ""))
            with col3:
                score = source.get("rerank_score")
                if score is not None:
                    st.write("**Rerank:**", round(score, 4))

            st.write("**Document:**", source.get("document_title", ""))
            st.write("**File:**", source.get("source", ""))
            st.write("**Fallback stage:**", source.get("metadata_fallback_stage", "strict"))
            st.text(source.get("text", ""))

            if index < len(sources):
                st.divider()


with st.sidebar:
    st.header("💬 Conversations")

    if st.button("➕ New Conversation", use_container_width=True):
        new_id = create_new_chat()
        switch_conversation(new_id)
        st.rerun()

    conversations = list_conversations()
    conversation_map = {item["id"]: item for item in conversations}
    conversation_ids = [item["id"] for item in conversations]

    if conversation_ids:
        current_id = st.session_state.conversation_id
        if current_id not in conversation_ids:
            current_id = conversation_ids[0]
            st.session_state.conversation_id = current_id

        selected_id = st.selectbox(
            "Previous conversations",
            options=conversation_ids,
            index=conversation_ids.index(current_id),
            format_func=lambda cid: conversation_map[cid]["title"],
            key="conversation_selector",
        )
        if selected_id != st.session_state.conversation_id:
            st.session_state.conversation_id = selected_id
            st.rerun()

    current_id = st.session_state.conversation_id
    current_conversation = get_conversation(current_id)
    if current_conversation is None:
        current_id = create_new_chat()
        switch_conversation(current_id)
        current_conversation = get_conversation(current_id)

    st.divider()
    st.write("**Current conversation**")
    st.write(current_conversation["title"])
    st.caption(f"{count_messages(current_id)} stored messages")

    with st.form(f"rename_form_{current_id}"):
        new_title = st.text_input("Rename conversation", value=current_conversation["title"])
        rename_submitted = st.form_submit_button("Rename", use_container_width=True)

    if rename_submitted:
        new_title = new_title.strip()
        if new_title and new_title != current_conversation["title"]:
            rename_conversation(current_id, new_title)
            st.rerun()

    if st.button("🧹 Clear Conversation", use_container_width=True):
        clear_conversation(current_id)
        st.rerun()

    if st.button("🗑️ Delete Conversation", use_container_width=True):
        delete_conversation(current_id)
        remaining = list_conversations()
        switch_conversation(remaining[0]["id"] if remaining else create_new_chat())
        st.rerun()

    st.divider()
    st.header("🔎 Retrieval")

    retrieval_mode_label = st.selectbox(
        "Search mode",
        ["Global", "Single Provider", "Compare Providers", "Specific Document"],
    )

    selected_provider = None
    selected_providers = []
    selected_document_id = None

    if retrieval_mode_label == "Single Provider":
        selected_provider = st.selectbox("Provider", EXPECTED_PROVIDERS)

    elif retrieval_mode_label == "Compare Providers":
        selected_providers = st.multiselect(
            "Providers",
            EXPECTED_PROVIDERS,
            default=["AWS", "Azure"],
        )
        if len(selected_providers) < 2:
            st.warning("Select at least two providers.")

    elif retrieval_mode_label == "Specific Document":
        document_provider = st.selectbox("Document provider", EXPECTED_PROVIDERS)
        provider_documents = [
            item for item in rag["document_catalog"]
            if item["provider"] == document_provider
        ]
        document_map = {item["document_id"]: item for item in provider_documents}
        selected_document_id = st.selectbox(
            "Document",
            options=list(document_map),
            format_func=lambda document_id: document_label(document_map[document_id]),
        )

    st.divider()
    st.subheader("Backend")
    st.success("V14 backend loaded")
    st.write("**Dense cache:**", "all-MiniLM-L6-v2 · 384 dimensions")
    st.write("**BM25:**", "local 11,502-chunk corpus")
    st.write("**Legacy index reference:**", PINECONE_INDEX_NAME)
    st.write("**Namespace reference:**", PINECONE_NAMESPACE)
    st.write("**Ollama:**", OLLAMA_MODEL)
    st.write("**Documents:**", len(rag["pdf_records"]))
    st.write("**Chunks:**", len(rag["chunks"]))

    st.divider()
    st.subheader("Session")
    st.write("**Conversation ID**")
    st.code(current_id)
    st.write("**Messages:**", count_messages(current_id))
    st.caption(f"SQLite: {MEMORY_DB_PATH}")


current_id = st.session_state.conversation_id
current_conversation = get_conversation(current_id)

st.subheader(current_conversation["title"])
st.caption(f"Conversation ID: {current_id}")
st.success(
    f"V14 ready — {len(rag['pdf_records'])} PDFs, "
    f"{len(rag['chunks']):,} chunks, 5 providers."
)

with st.expander("Example Questions"):
    examples = [
        "What are AWS IAM best practices?",
        "What about least privilege?",
        "Compare AWS and Azure infrastructure as code.",
        "Compare cloud providers for reliability side by side.",
        "What does GCP recommend for sustainability?",
    ]
    for i, example in enumerate(examples):
        if st.button(example, key=f"example_{current_id}_{i}", use_container_width=True):
            st.session_state["example_question"] = example


for message in load_messages(current_id):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            if message.get("standalone_question"):
                with st.expander("Conversation memory / interpreted question"):
                    st.write(message["standalone_question"])
            show_sources(message.get("sources", []))


question = st.chat_input("Ask about AWS, Azure, GCP, Oracle, or IBM...")
if question is None and st.session_state.get("example_question"):
    question = st.session_state.pop("example_question")


if question:
    if retrieval_mode_label == "Compare Providers" and len(selected_providers) < 2:
        st.warning("Select at least two providers before running a comparison.")
        st.stop()

    mode_map = {
        "Global": "global",
        "Single Provider": "provider",
        "Compare Providers": "cross_provider",
        "Specific Document": "document",
    }
    mode = mode_map[retrieval_mode_label]

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Rewriting context, retrieving hybrid evidence, reranking, and validating citations..."):
            try:
                result = ask_v14_with_memory(
                    conversation_id=current_id,
                    question=question,
                    mode=mode,
                    provider=selected_provider,
                    providers=selected_providers,
                    document_id=selected_document_id,
                    show_debug=False,
                )
            except Exception as exc:
                st.error("The question could not be processed.")
                st.exception(exc)
                st.stop()

        quality = result.get("quality", {})
        status = quality.get("status", "unknown")

        if result.get("generation_skipped"):
            st.warning(result["answer"])
        elif result.get("citation_valid") is False:
            st.warning("The answer was generated, but citation validation did not fully pass.")
            st.markdown(result["answer"])
        else:
            st.markdown(result["answer"])

        col1, col2, col3 = st.columns(3)
        with col1:
            st.caption(f"Quality: {status}")
        with col2:
            st.caption(f"Evidence: {result.get('evidence_count', 0)}")
        with col3:
            total_time = result.get("total_time")
            if total_time is not None:
                st.caption(f"Total: {total_time:.2f}s")

        with st.expander("V14 retrieval details"):
            st.write("**Original question:**", question)
            st.write("**Standalone question:**", result.get("standalone_question"))
            st.write("**Scoped question:**", result.get("scoped_question"))
            st.write("**Mode:**", mode)
            st.write("**Required providers:**", result.get("required_providers", []))
            st.write("**Retrieved providers:**", result.get("retrieved_providers", []))
            st.write("**Quality status:**", status)
            st.write("**Quality reason:**", quality.get("reason", ""))
            st.write("**Generation skipped:**", result.get("generation_skipped"))
            st.write("**Citation valid:**", result.get("citation_valid"))
            st.write("**Rerank mode:**", result.get("rerank_mode"))

        show_sources(result.get("evidence", []))

    st.rerun()