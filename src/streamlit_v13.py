
# ============================================================
# MULTI-CLOUD RAG V13
# PRODUCTION STREAMLIT APPLICATION
# ============================================================

from pathlib import Path

import os
import sys

import streamlit as st


# ============================================================
# REPOSITORY PATHS
# ============================================================

SRC_ROOT = (

    Path(__file__)
    .resolve()
    .parent

)

REPO_ROOT = (

    SRC_ROOT.parent

)


if str(SRC_ROOT) not in sys.path:

    sys.path.insert(

        0,

        str(SRC_ROOT),

    )


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(

    page_title=
        "Multi-Cloud RAG V13",

    page_icon=
        "☁️",

    layout=
        "wide",

    initial_sidebar_state=
        "expanded",

)


# ============================================================
# LOAD V13 BACKEND
#
# Streamlit reruns this script frequently.
#
# cache_resource keeps the expensive backend available:
#
# - MiniLM
# - CrossEncoder
# - BM25
# - local chunk cache
# - Pinecone client
# ============================================================

@st.cache_resource(
    show_spinner=False
)
def load_v13_backend():

    import rag_v13

    return rag_v13


# ============================================================
# CHECK REQUIRED SECRET
# ============================================================

if not os.getenv(
    "PINECONE_API_KEY"
):

    st.error(

        "PINECONE_API_KEY is not available "
        "to the Streamlit process."

    )

    st.code(

        '$env:PINECONE_API_KEY="YOUR_KEY_HERE"',

        language=
            "powershell",

    )

    st.caption(

        "Set the environment variable in the same "
        "PowerShell window used to start Streamlit, "
        "then restart the application."

    )

    st.stop()


# ============================================================
# LOAD BACKEND
# ============================================================

try:

    with st.spinner(

        "Loading V13 RAG backend..."

    ):

        rag = (

            load_v13_backend()

        )


except Exception as exc:

    st.error(

        "V13 backend failed to load."

    )

    st.exception(
        exc
    )

    st.stop()


# ============================================================
# MEMORY DATABASE
# ============================================================

STREAMLIT_DB = (

    rag.DEFAULT_MEMORY_DB

)


# ============================================================
# SESSION DEFAULTS
# ============================================================

SESSION_DEFAULTS = {

    "conversation_id":
        None,

    "messages":
        [],

    "mode":
        "global",

    "provider":
        "AWS",

    "providers": [

        "AWS",

        "Azure",

    ],

    "document_id":
        None,

    "last_result":
        None,

    "is_streaming":
        False,

    "last_error":
        None,

}


# ============================================================
# INITIALIZE SESSION
# ============================================================

def initialize_session():

    for key, default_value in (

        SESSION_DEFAULTS.items()

    ):

        if key not in st.session_state:

            if isinstance(
                default_value,
                list,
            ):

                st.session_state[
                    key
                ] = list(
                    default_value
                )

            else:

                st.session_state[
                    key
                ] = default_value


    # --------------------------------------------------------
    # Create initial conversation
    # --------------------------------------------------------

    if (

        st.session_state[
            "conversation_id"
        ]
        is None

    ):

        st.session_state[
            "conversation_id"
        ] = (

            rag.create_new_chat(

                STREAMLIT_DB

            )

        )


    # --------------------------------------------------------
    # Synchronize UI with SQLite
    # --------------------------------------------------------

    refresh_messages()


# ============================================================
# REFRESH CHAT HISTORY
# ============================================================

def refresh_messages():

    conversation_id = (

        st.session_state[
            "conversation_id"
        ]

    )


    if not conversation_id:

        st.session_state[
            "messages"
        ] = []

        return


    st.session_state[
        "messages"
    ] = (

        rag.load_messages(

            conversation_id,

            STREAMLIT_DB,

        )

    )


# ============================================================
# NEW CHAT
# ============================================================

def start_new_chat():

    new_conversation_id = (

        rag.create_new_chat(

            STREAMLIT_DB

        )

    )


    st.session_state[
        "conversation_id"
    ] = (

        new_conversation_id

    )


    st.session_state[
        "messages"
    ] = []


    st.session_state[
        "last_result"
    ] = None


    st.session_state[
        "last_error"
    ] = None


    st.session_state[
        "is_streaming"
    ] = False


# ============================================================
# INITIALIZE APPLICATION STATE
# ============================================================

initialize_session()


# ============================================================
# BACKEND INFORMATION
# ============================================================

backend_info = (

    rag.get_v13_backend_info()

)


# ============================================================
# PROVIDERS
# ============================================================

PROVIDERS = [

    "AWS",

    "Azure",

    "GCP",

    "Oracle",

    "IBM",

]


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title(
        "☁️ Multi-Cloud RAG"
    )


    st.caption(
        "V13 Hybrid Retrieval"
    )


    st.divider()


    # ========================================================
    # NEW CHAT
    # ========================================================

    if st.button(

        "➕ New Chat",

        use_container_width=True,

    ):

        start_new_chat()

        st.rerun()


    # ========================================================
    # RETRIEVAL SETTINGS
    # ========================================================

    st.subheader(
        "Retrieval"
    )


    mode_labels = [

        "Global",

        "Single Provider",

        "Cross Provider",

    ]


    mode_values = {

        "Global":
            "global",

        "Single Provider":
            "provider",

        "Cross Provider":
            "cross_provider",

    }


    reverse_mode_values = {

        value:
            key

        for key, value in mode_values.items()

    }


    current_mode_label = (

        reverse_mode_values.get(

            st.session_state[
                "mode"
            ],

            "Global",

        )

    )


    current_mode_index = (

        mode_labels.index(
            current_mode_label
        )

    )


    selected_mode_label = (

        st.selectbox(

            "Search mode",

            options=
                mode_labels,

            index=
                current_mode_index,

        )

    )


    st.session_state[
        "mode"
    ] = (

        mode_values[
            selected_mode_label
        ]

    )


    # ========================================================
    # SINGLE PROVIDER MODE
    # ========================================================

    if (

        st.session_state[
            "mode"
        ]
        ==
        "provider"

    ):

        current_provider = (

            st.session_state.get(
                "provider"
            )

            or

            "AWS"

        )


        provider_index = (

            PROVIDERS.index(
                current_provider
            )

            if current_provider in PROVIDERS

            else 0

        )


        selected_provider = (

            st.selectbox(

                "Provider",

                options=
                    PROVIDERS,

                index=
                    provider_index,

            )

        )


        st.session_state[
            "provider"
        ] = (

            selected_provider

        )


        st.session_state[
            "providers"
        ] = []


    # ========================================================
    # CROSS-PROVIDER MODE
    # ========================================================

    elif (

        st.session_state[
            "mode"
        ]
        ==
        "cross_provider"

    ):

        stored_providers = [

            provider

            for provider in (

                st.session_state.get(
                    "providers",
                    []
                )

            )

            if provider in PROVIDERS

        ]


        if len(
            stored_providers
        ) < 2:

            stored_providers = [

                "AWS",

                "Azure",

            ]


        selected_providers = (

            st.multiselect(

                "Providers",

                options=
                    PROVIDERS,

                default=
                    stored_providers,

            )

        )


        st.session_state[
            "providers"
        ] = (

            selected_providers

        )


        st.session_state[
            "provider"
        ] = None


    # ========================================================
    # GLOBAL MODE
    # ========================================================

    else:

        st.session_state[
            "provider"
        ] = None


        st.session_state[
            "providers"
        ] = []


    # ========================================================
    # BACKEND STATUS
    # ========================================================

    st.divider()


    st.subheader(
        "Backend"
    )


    st.success(
        "V13 backend loaded"
    )


    st.caption(

        f"Pinecone: "
        f"{backend_info['pinecone_index']}"

    )


    st.caption(

        f"Ollama: "
        f"{backend_info['ollama_model']}"

    )


    st.caption(

        f"Chunks: "
        f"{backend_info['chunks']:,}"

    )


    # ========================================================
    # SESSION INFORMATION
    # ========================================================

    st.divider()


    st.subheader(
        "Session"
    )


    st.caption(
        "Conversation ID"
    )


    st.code(

        st.session_state[
            "conversation_id"
        ],

        language=None,

    )


    message_count_placeholder = (

        st.empty()

    )


    message_count_placeholder.caption(

        f"Messages: "
        f"{len(st.session_state['messages'])}"

    )


# ============================================================
# MAIN PAGE HEADER
# ============================================================

st.title(
    "Multi-Cloud RAG V13"
)


st.caption(

    "Dense Pinecone retrieval + BM25 + "
    "Reciprocal Rank Fusion + CrossEncoder reranking + "
    "Ollama live streaming + persistent SQLite memory."

)


# ============================================================
# SYSTEM METRICS
# ============================================================

metric_columns = (

    st.columns(
        4
    )

)


with metric_columns[0]:

    st.metric(

        "Providers",

        "5",

    )


with metric_columns[1]:

    st.metric(

        "Documents",

        "27",

    )


with metric_columns[2]:

    st.metric(

        "Chunks",

        f"{backend_info['chunks']:,}",

    )


with metric_columns[3]:

    st.metric(

        "Mode",

        st.session_state[
            "mode"
        ],

    )


st.divider()


# ============================================================
# DISPLAY PERSISTED CONVERSATION
# ============================================================

for message in (

    st.session_state[
        "messages"
    ]

):

    role = (

        message.get(
            "role",
            "assistant",
        )

    )


    content = (

        message.get(
            "content",
            "",
        )

    )


    with st.chat_message(
        role
    ):

        st.markdown(
            content
        )


# ============================================================
# CHAT INPUT
# ============================================================

prompt = (

    st.chat_input(

        "Ask about AWS, Azure, GCP, Oracle or IBM...",

        disabled=
            st.session_state[
                "is_streaming"
            ],

    )

)


# ============================================================
# PROCESS USER QUESTION
# ============================================================

if prompt:

    # ========================================================
    # VALIDATE CROSS-PROVIDER SELECTION
    # ========================================================

    if (

        st.session_state[
            "mode"
        ]
        ==
        "cross_provider"

        and

        len(
            st.session_state[
                "providers"
            ]
        )
        <
        2

    ):

        st.warning(

            "Select at least two providers "
            "for Cross Provider mode."

        )

        st.stop()


    # ========================================================
    # DISPLAY USER MESSAGE IMMEDIATELY
    # ========================================================

    with st.chat_message(
        "user"
    ):

        st.markdown(
            prompt
        )


    # ========================================================
    # ASSISTANT MESSAGE
    # ========================================================

    with st.chat_message(
        "assistant"
    ):

        # ----------------------------------------------------
        # Status indicator appears during retrieval/reranking.
        # ----------------------------------------------------

        status = (

            st.status(

                "Retrieving and reranking evidence...",

                expanded=False,

            )

        )


        # ----------------------------------------------------
        # One placeholder is continuously updated by tokens.
        # ----------------------------------------------------

        answer_placeholder = (

            st.empty()

        )


        # ----------------------------------------------------
        # Accumulate streamed text.
        # ----------------------------------------------------

        streamed_parts = []


        first_visible_token = [

            False

        ]


        # ====================================================
        # TOKEN CALLBACK
        #
        # rag_v13 calls this function for every Ollama token.
        # ====================================================

        def on_token(
            token,
        ):

            streamed_parts.append(
                token
            )


            # -----------------------------------------------
            # Once the first token arrives, retrieval is done
            # and Ollama generation has begun.
            # -----------------------------------------------

            if not first_visible_token[0]:

                first_visible_token[0] = True


                status.update(

                    label=
                        "Generating grounded answer...",

                    state=
                        "running",

                )


            partial_answer = (

                "".join(
                    streamed_parts
                )

            )


            # -----------------------------------------------
            # LIVE BROWSER OUTPUT
            #
            # Temporary [SOURCE N] markers may appear here.
            # They are replaced with trusted citations after
            # generation finishes.
            # -----------------------------------------------

            answer_placeholder.markdown(

                partial_answer

            )


        # ====================================================
        # RUN COMPLETE V13 PIPELINE
        # ====================================================

        st.session_state[
            "is_streaming"
        ] = True


        try:

            result = (

                rag
                .ask_v13_with_memory_streaming(

                    conversation_id=
                        st.session_state[
                            "conversation_id"
                        ],

                    question=
                        prompt,

                    mode=
                        st.session_state[
                            "mode"
                        ],

                    provider=
                        st.session_state[
                            "provider"
                        ],

                    providers=
                        st.session_state[
                            "providers"
                        ],

                    document_id=
                        st.session_state[
                            "document_id"
                        ],

                    token_callback=
                        on_token,

                )

            )


            # =================================================
            # FINAL TRUSTED ANSWER
            #
            # Replace temporary streamed [SOURCE N] output with
            # deterministic metadata citations.
            # =================================================

            answer_placeholder.markdown(

                result[
                    "answer"
                ]

            )


            # =================================================
            # COMPLETE STATUS
            # =================================================

            status.update(

                label=
                    "Answer complete",

                state=
                    "complete",

            )


            # =================================================
            # SAVE RESULT IN SESSION
            # =================================================

            st.session_state[
                "last_result"
            ] = result


            st.session_state[
                "last_error"
            ] = None


            # =================================================
            # PERFORMANCE
            # =================================================

            time_to_first_token = (

                result.get(
                    "time_to_first_token"
                )

            )


            total_time = (

                result.get(
                    "total_time"
                )

            )


            performance_columns = (

                st.columns(
                    3
                )

            )


            with performance_columns[0]:

                if isinstance(

                    time_to_first_token,

                    (
                        int,
                        float,
                    ),

                ):

                    st.metric(

                        "First token",

                        f"{time_to_first_token:.2f}s",

                    )

                else:

                    st.metric(

                        "First token",

                        "N/A",

                    )


            with performance_columns[1]:

                if isinstance(

                    total_time,

                    (
                        int,
                        float,
                    ),

                ):

                    st.metric(

                        "Generation",

                        f"{total_time:.2f}s",

                    )

                else:

                    st.metric(

                        "Generation",

                        "N/A",

                    )


            with performance_columns[2]:

                citation_status = (

                    "Valid ✅"

                    if result.get(
                        "citation_valid"
                    )

                    else "Review ⚠️"

                )


                st.metric(

                    "Citations",

                    citation_status,

                )


            # =================================================
            # TRUSTED CITATIONS PANEL
            # =================================================

            citations = (

                result.get(
                    "citations",
                    []
                )

            )


            if citations:

                with st.expander(

                    "Trusted citations",

                    expanded=False,

                ):

                    for citation in citations:

                        st.markdown(

                            f"- {citation}"

                        )


            # =================================================
            # RETRIEVED EVIDENCE PANEL
            # =================================================

            evidence = (

                result.get(
                    "evidence",
                    []
                )

            )


            if evidence:

                with st.expander(

                    "Retrieved evidence",

                    expanded=False,

                ):

                    for number, item in enumerate(

                        evidence,

                        start=1,

                    ):

                        st.markdown(

                            f"**SOURCE {number}**  \n"
                            f"Provider: {item.get('provider', '')}  \n"
                            f"Document: {item.get('document_title', '')}  \n"
                            f"PDF page: {item.get('pdf_page', '')}"

                        )


                        rerank_score = (

                            item.get(
                                "rerank_score"
                            )

                        )


                        if isinstance(

                            rerank_score,

                            (
                                int,
                                float,
                            ),

                        ):

                            st.caption(

                                f"Rerank score: "
                                f"{rerank_score:.4f}"

                            )


                        evidence_text = (

                            item.get(
                                "text",
                                ""
                            )

                        )


                        if evidence_text:

                            st.text(

                                evidence_text[
                                    :1200
                                ]

                            )


                        st.divider()


            # =================================================
            # CONVERSATIONAL REWRITE PANEL
            #
            # Useful while validating memory behavior.
            # =================================================

            standalone_question = (

                result.get(
                    "standalone_question"
                )

            )


            original_question = (

                result.get(
                    "question"
                )

            )


            if (

                standalone_question

                and

                original_question

                and

                standalone_question.strip()
                !=
                original_question.strip()

            ):

                with st.expander(

                    "Conversation memory rewrite",

                    expanded=False,

                ):

                    st.markdown(
                        "**Original**"
                    )

                    st.write(
                        original_question
                    )


                    st.markdown(
                        "**Standalone retrieval question**"
                    )

                    st.write(
                        standalone_question
                    )


            # =================================================
            # REFRESH SQLITE-BACKED MESSAGE HISTORY
            # =================================================

            refresh_messages()


            # =================================================
            # UPDATE SIDEBAR MESSAGE COUNT IMMEDIATELY
            # =================================================

            message_count_placeholder.caption(

                f"Messages: "
                f"{len(st.session_state['messages'])}"

            )


        except Exception as exc:

            st.session_state[
                "last_error"
            ] = str(
                exc
            )


            status.update(

                label=
                    "V13 request failed",

                state=
                    "error",

            )


            answer_placeholder.empty()


            st.error(

                "The V13 RAG request failed."

            )


            st.exception(
                exc
            )


        finally:

            st.session_state[
                "is_streaming"
            ] = False
