from collections import Counter
from pathlib import Path
import os
import re

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import DenseVectorQuery, Pinecone
from sentence_transformers import CrossEncoder, SentenceTransformer


# ============================================================
# RAG V12 - MULTI-CLOUD PRODUCTION BACKEND
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLOUD_DOCUMENT_ROOT = PROJECT_ROOT / "data" / "cloud_documents"

EXPECTED_PROVIDERS = ["AWS", "Azure", "GCP", "Oracle", "IBM"]
EXPECTED_TOTAL_PDFS = 27
EXPECTED_TOTAL_PAGES = 3451
EXPECTED_TOTAL_CHUNKS = 11502

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
NEIGHBOR_RANGE = 1

SEMANTIC_TOP_K = 8
FINAL_TOP_K = 5
CROSS_PROVIDER_SEMANTIC_TOP_K = 12
CROSS_PROVIDER_FINAL_TOP_K = 3
MIN_EVIDENCE_CHARS = 120
MEMORY_MAX_MESSAGES = 12

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
OLLAMA_MODEL = "qwen3:4b-instruct"

PINECONE_INDEX_NAME = "multicloud-rag-v12"
PINECONE_NAMESPACE = "cloud-documents"
PINECONE_VECTOR_FIELD = "embedding"
PINECONE_DIMENSION = 384

REFUSAL_MESSAGE = (
    "I don't have enough information in the provided documents to answer that."
)
MAX_CITATION_RETRIES = 3

PROVIDER_ALIASES = {
    "AWS": ["aws", "amazon web services"],
    "Azure": ["azure", "microsoft azure"],
    "GCP": ["gcp", "google cloud", "google cloud platform"],
    "Oracle": ["oracle", "oci", "oracle cloud"],
    "IBM": ["ibm", "ibm cloud"],
}


# ============================================================
# INGESTION HELPERS
# ============================================================

def clean_cloud_text(text):
    text = text.replace("\t", " ")
    text = re.sub(r"(?m)^\s*\d{1,4}\s*$", "", text)
    text = re.sub(r"[ ]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def slugify(text):
    value = text.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


def title_from_filename(pdf_path):
    title = re.sub(r"[_-]+", " ", pdf_path.stem)
    title = re.sub(r"\s+", " ", title)
    return title.strip()


def get_provider_from_path(pdf_path):
    relative_path = pdf_path.relative_to(CLOUD_DOCUMENT_ROOT)
    provider = relative_path.parts[0]
    if provider not in EXPECTED_PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}")
    return provider


def discover_cloud_pdfs():
    pdf_files = sorted(
        path
        for path in CLOUD_DOCUMENT_ROOT.rglob("*.pdf")
        if path.is_file()
    )

    records = []
    for pdf_path in pdf_files:
        provider = get_provider_from_path(pdf_path)
        document_id = f"{provider.lower()}__{slugify(pdf_path.stem)}"
        records.append(
            {
                "path": pdf_path,
                "provider": provider,
                "document_id": document_id,
                "document_title": title_from_filename(pdf_path),
                "source": pdf_path.name,
            }
        )

    return records


def load_cloud_documents(pdf_records):
    all_pages = []

    for record in pdf_records:
        loader = PyMuPDFLoader(str(record["path"]))
        pages = loader.load()

        for page in pages:
            page.page_content = clean_cloud_text(page.page_content)
            pdf_page = int(page.metadata.get("page", 0)) + 1

            page.metadata["pdf_page"] = pdf_page
            page.metadata["provider"] = record["provider"]
            page.metadata["document_id"] = record["document_id"]
            page.metadata["document_title"] = record["document_title"]
            page.metadata["source"] = record["source"]
            page.metadata["source_path"] = str(record["path"])

        all_pages.extend(pages)

    return all_pages


def create_chunks(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = [
        chunk
        for chunk in splitter.split_documents(pages)
        if chunk.page_content.strip()
    ]

    per_document_chunk_counter = Counter()

    for global_chunk_number, chunk in enumerate(chunks):
        document_id = chunk.metadata["document_id"]
        document_chunk_number = per_document_chunk_counter[document_id]

        chunk.metadata["document_chunk_number"] = document_chunk_number
        chunk.metadata["global_chunk_number"] = global_chunk_number
        chunk.metadata["chunk_id"] = (
            f"{document_id}__chunk_{document_chunk_number:05d}"
        )

        per_document_chunk_counter[document_id] += 1

    return chunks


def build_chunk_lookup(chunks):
    return {
        (
            chunk.metadata["document_id"],
            chunk.metadata["document_chunk_number"],
        ): chunk
        for chunk in chunks
    }


def build_document_catalog(pdf_records):
    return [
        {
            "provider": record["provider"],
            "document_id": record["document_id"],
            "document_title": record["document_title"],
            "source": record["source"],
        }
        for record in pdf_records
    ]


# ============================================================
# INITIALIZATION
# ============================================================

def initialize_rag():
    if not CLOUD_DOCUMENT_ROOT.exists():
        raise FileNotFoundError(
            f"Cloud document folder not found: {CLOUD_DOCUMENT_ROOT}"
        )

    pdf_records = discover_cloud_pdfs()
    pages = load_cloud_documents(pdf_records)
    chunks = create_chunks(pages)
    rag_chunk_lookup = build_chunk_lookup(chunks)

    if len(pdf_records) != EXPECTED_TOTAL_PDFS:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL_PDFS} PDFs, found {len(pdf_records)}."
        )

    if len(pages) != EXPECTED_TOTAL_PAGES:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL_PAGES} pages, found {len(pages)}."
        )

    if len(chunks) != EXPECTED_TOTAL_CHUNKS:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL_CHUNKS} chunks, found {len(chunks)}."
        )

    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    reranker = CrossEncoder(RERANKER_MODEL)
    llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "PINECONE_API_KEY is not set. Set it before starting Streamlit."
        )

    pinecone_client = Pinecone(api_key=api_key)

    if not pinecone_client.indexes.exists(PINECONE_INDEX_NAME):
        raise RuntimeError(
            f"Pinecone index {PINECONE_INDEX_NAME!r} does not exist."
        )

    index = pinecone_client.index(PINECONE_INDEX_NAME)

    return {
        "pdf_records": pdf_records,
        "pages": pages,
        "chunks": chunks,
        "rag_chunk_lookup": rag_chunk_lookup,
        "document_catalog": build_document_catalog(pdf_records),
        "embedding_model": embedding_model,
        "reranker": reranker,
        "llm": llm,
        "pinecone_client": pinecone_client,
        "index": index,
    }


# ============================================================
# RETRIEVAL
# ============================================================

def get_document_neighbors(
    rag_chunk_lookup,
    document_id,
    document_chunk_number,
    neighbor_range=NEIGHBOR_RANGE,
):
    neighbors = []

    for offset in range(-neighbor_range, neighbor_range + 1):
        key = (document_id, document_chunk_number + offset)
        neighbor = rag_chunk_lookup.get(key)
        if neighbor is not None:
            neighbors.append(neighbor)

    return neighbors


def pinecone_search(
    rag,
    question,
    top_k=SEMANTIC_TOP_K,
    provider=None,
    document_id=None,
):
    query_embedding = rag["embedding_model"].encode(
        [question],
        normalize_embeddings=True,
    )[0]

    filters = []

    if provider is not None:
        filters.append({"provider": {"$eq": provider}})

    if document_id is not None:
        filters.append({"document_id": {"$eq": document_id}})

    if len(filters) == 0:
        metadata_filter = None
    elif len(filters) == 1:
        metadata_filter = filters[0]
    else:
        metadata_filter = {"$and": filters}

    response = rag["index"].documents.search(
        namespace=PINECONE_NAMESPACE,
        top_k=top_k,
        score_by=[
            DenseVectorQuery(
                field=PINECONE_VECTOR_FIELD,
                values=query_embedding.tolist(),
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
        filter=metadata_filter,
    )

    results = []

    for match in response.matches:
        results.append(
            {
                "chunk_id": match.get("_id", ""),
                "semantic_score": float(match.score),
                "provider": match.get("provider", ""),
                "document_id": match.get("document_id", ""),
                "document_title": match.get("document_title", ""),
                "source": match.get("source", ""),
                "pdf_page": int(match.get("pdf_page", 0)),
                "document_chunk_number": int(
                    match.get("document_chunk_number", 0)
                ),
                "global_chunk_number": int(
                    match.get("global_chunk_number", 0)
                ),
                "text": match.get("text", ""),
            }
        )

    return results


def expand_with_neighbors(rag, semantic_results):
    expanded = []

    for result in semantic_results:
        neighbors = get_document_neighbors(
            rag["rag_chunk_lookup"],
            result["document_id"],
            result["document_chunk_number"],
        )

        for neighbor in neighbors:
            expanded.append(
                {
                    "chunk_id": neighbor.metadata["chunk_id"],
                    "semantic_score": result["semantic_score"],
                    "provider": neighbor.metadata["provider"],
                    "document_id": neighbor.metadata["document_id"],
                    "document_title": neighbor.metadata["document_title"],
                    "source": neighbor.metadata["source"],
                    "pdf_page": neighbor.metadata["pdf_page"],
                    "document_chunk_number": neighbor.metadata[
                        "document_chunk_number"
                    ],
                    "global_chunk_number": neighbor.metadata[
                        "global_chunk_number"
                    ],
                    "text": neighbor.page_content,
                }
            )

    return expanded


def deduplicate_results(results):
    unique = {}

    for result in results:
        chunk_id = result["chunk_id"]

        if chunk_id not in unique:
            unique[chunk_id] = result
        elif result["semantic_score"] > unique[chunk_id]["semantic_score"]:
            unique[chunk_id] = result

    return list(unique.values())


def filter_informative_results(results):
    informative = [
        result
        for result in results
        if len(result["text"].strip()) >= MIN_EVIDENCE_CHARS
    ]
    return informative if informative else results


def rerank_results(rag, question, results, final_top_k=FINAL_TOP_K):
    if not results:
        return []

    results = filter_informative_results(results)

    pairs = [
        [question, result["text"]]
        for result in results
    ]

    scores = rag["reranker"].predict(pairs)

    for result, score in zip(results, scores):
        result["rerank_score"] = float(score)

    results.sort(
        key=lambda item: item["rerank_score"],
        reverse=True,
    )

    return results[:final_top_k]


def retrieve(
    rag,
    question,
    provider=None,
    document_id=None,
    semantic_top_k=SEMANTIC_TOP_K,
    final_top_k=FINAL_TOP_K,
):
    semantic = pinecone_search(
        rag,
        question,
        top_k=semantic_top_k,
        provider=provider,
        document_id=document_id,
    )

    expanded = expand_with_neighbors(rag, semantic)
    unique = deduplicate_results(expanded)

    return rerank_results(
        rag,
        question,
        unique,
        final_top_k=final_top_k,
    )


def build_provider_retrieval_question(question, provider):
    return (
        f"{provider} guidance and best practices for the user's topic. "
        "Focus on architecture, design principles, implementation guidance, "
        "security, reliability, resilience, operations, and workload practices "
        "that are directly relevant to the question. "
        f"Original question: {question}"
    )


def retrieve_cross_provider(rag, question, providers):
    combined = []

    for provider in providers:
        provider_question = build_provider_retrieval_question(
            question,
            provider,
        )

        provider_results = retrieve(
            rag,
            provider_question,
            provider=provider,
            semantic_top_k=CROSS_PROVIDER_SEMANTIC_TOP_K,
            final_top_k=CROSS_PROVIDER_FINAL_TOP_K,
        )

        combined.extend(provider_results)

    combined.sort(
        key=lambda item: item["rerank_score"],
        reverse=True,
    )

    return combined


# ============================================================
# CITATIONS AND GROUNDED GENERATION
# ============================================================

def create_citation_label(result):
    return (
        f"[{result['provider']} | "
        f"{result['document_title']} | "
        f"PDF page {result['pdf_page']}]"
    )


def normalize_source_markers(text):
    text = re.sub(
        r"\(\s*SOURCE\s+(\d+)\s*\)",
        r"[SOURCE \1]",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"(?<!\[)\bSOURCE\s+(\d+)\b(?!\])",
        r"[SOURCE \1]",
        text,
        flags=re.IGNORECASE,
    )

    return text


SOURCE_PATTERN = re.compile(
    r"\[SOURCE\s+(\d+)\]",
    re.IGNORECASE,
)


def build_context(evidence):
    blocks = []

    for source_number, result in enumerate(evidence, start=1):
        blocks.append(
            (
                f"[SOURCE {source_number}]\n"
                f"Provider: {result['provider']}\n"
                f"Document: {result['document_title']}\n"
                f"Source file: {result['source']}\n"
                f"PDF page: {result['pdf_page']}\n\n"
                f"Evidence:\n{result['text']}"
            )
        )

    return (
        "\n\n--------------------------------------------------\n\n"
    ).join(blocks)


def get_source_numbers(text):
    text = normalize_source_markers(text)
    return [int(number) for number in SOURCE_PATTERN.findall(text)]


def validate_source_numbers(text, evidence):
    source_numbers = get_source_numbers(text)
    valid_numbers = set(range(1, len(evidence) + 1))
    invalid_numbers = [
        number
        for number in source_numbers
        if number not in valid_numbers
    ]
    return source_numbers, invalid_numbers


def get_cited_providers(source_numbers, evidence):
    providers = set()

    for source_number in source_numbers:
        evidence_index = source_number - 1
        if 0 <= evidence_index < len(evidence):
            providers.add(evidence[evidence_index]["provider"])

    return providers


def allowed_source_markers(evidence):
    return " ".join(
        f"[SOURCE {number}]"
        for number in range(1, len(evidence) + 1)
    )


def replace_source_markers(answer, evidence):
    answer = normalize_source_markers(answer)
    citations = []

    def replacement(match):
        source_number = int(match.group(1))
        evidence_index = source_number - 1

        if 0 <= evidence_index < len(evidence):
            citation = create_citation_label(evidence[evidence_index])
            citations.append(citation)
            return citation

        return ""

    final_answer = SOURCE_PATTERN.sub(replacement, answer)
    citations = list(dict.fromkeys(citations))

    return final_answer.strip(), citations


def build_grounded_prompt(
    question,
    evidence,
    required_providers=None,
):
    required_providers = required_providers or []

    provider_requirement = ""
    if required_providers:
        provider_requirement = (
            "\nThis is a provider comparison.\n"
            "You MUST discuss and cite evidence from ALL of these providers:\n\n"
            + ", ".join(required_providers)
            + "\n"
        )

    return f"""
You are a grounded multi-cloud RAG assistant.

Answer ONLY from the supplied evidence.

VALID SOURCE MARKERS:

{allowed_source_markers(evidence)}

{provider_requirement}

RULES:

1. Use only the supplied evidence.
2. Do not use outside knowledge.
3. Do not invent facts.
4. Do not invent citations.
5. Every factual paragraph or bullet must contain at least one valid [SOURCE N] marker.
6. Never use a SOURCE number that does not exist.
7. For provider comparisons, discuss and cite every requested provider.
8. Python will replace SOURCE markers with trusted provider, document, and PDF-page citations.
9. If the evidence truly cannot answer the question, reply exactly:

{REFUSAL_MESSAGE}

QUESTION:

{question}

EVIDENCE:

{build_context(evidence)}

ANSWER:
""".strip()


def build_correction_prompt(
    question,
    evidence,
    previous_answer,
    problem,
    required_providers=None,
):
    required_providers = required_providers or []

    return f"""
Your previous answer failed validation.

PROBLEM:

{problem}

PREVIOUS ANSWER:

{previous_answer}

VALID SOURCE MARKERS:

{allowed_source_markers(evidence)}

REQUIRED PROVIDERS:

{", ".join(required_providers) if required_providers else "None"}

Rewrite the entire answer.

Use only the supplied evidence.
Every factual paragraph or bullet must contain a valid [SOURCE N] marker.
For comparison questions, cite every requested provider.
Do not invent facts or sources.

QUESTION:

{question}

EVIDENCE:

{build_context(evidence)}

CORRECTED ANSWER:
""".strip()


def generate_grounded_answer(
    rag,
    question,
    evidence,
    required_providers=None,
    require_answer=False,
):
    required_providers = required_providers or []

    if not evidence:
        return {
            "answer": REFUSAL_MESSAGE,
            "raw_answer": REFUSAL_MESSAGE,
            "citations": [],
            "source_numbers": [],
            "invalid_sources": [],
            "cited_providers": set(),
            "missing_providers": set(required_providers),
            "refused": True,
            "attempts": 0,
            "valid": not require_answer,
        }

    prompt = build_grounded_prompt(
        question,
        evidence,
        required_providers,
    )

    raw_answer = ""
    attempts = 0

    for attempts in range(1, MAX_CITATION_RETRIES + 1):
        response = rag["llm"].invoke(prompt)
        raw_answer = normalize_source_markers(response.content.strip())

        refused = REFUSAL_MESSAGE.lower() in raw_answer.lower()

        if refused:
            if not require_answer:
                return {
                    "answer": REFUSAL_MESSAGE,
                    "raw_answer": REFUSAL_MESSAGE,
                    "citations": [],
                    "source_numbers": [],
                    "invalid_sources": [],
                    "cited_providers": set(),
                    "missing_providers": set(),
                    "refused": True,
                    "attempts": attempts,
                    "valid": True,
                }

            prompt = build_correction_prompt(
                question,
                evidence,
                raw_answer,
                (
                    "You refused to answer, but retrieved evidence exists "
                    "from all required providers."
                ),
                required_providers,
            )
            continue

        source_numbers, invalid_sources = validate_source_numbers(
            raw_answer,
            evidence,
        )

        cited_providers = get_cited_providers(
            source_numbers,
            evidence,
        )

        missing_providers = set(required_providers) - cited_providers

        if source_numbers and not invalid_sources and not missing_providers:
            break

        problems = []

        if not source_numbers:
            problems.append("No valid SOURCE citations were used.")

        if invalid_sources:
            problems.append(
                "Invalid SOURCE numbers: "
                + ", ".join(str(number) for number in invalid_sources)
            )

        if missing_providers:
            problems.append(
                "Missing provider citations: "
                + ", ".join(sorted(missing_providers))
            )

        prompt = build_correction_prompt(
            question,
            evidence,
            raw_answer,
            "\n".join(problems),
            required_providers,
        )

    source_numbers, invalid_sources = validate_source_numbers(
        raw_answer,
        evidence,
    )

    cited_providers = get_cited_providers(
        source_numbers,
        evidence,
    )

    missing_providers = set(required_providers) - cited_providers

    final_answer, citations = replace_source_markers(
        raw_answer,
        evidence,
    )

    refused = REFUSAL_MESSAGE.lower() in final_answer.lower()

    raw_source_tokens_remain = (
        re.search(
            r"\bSOURCE\s+\d+\b",
            final_answer,
            flags=re.IGNORECASE,
        )
        is not None
    )

    if refused:
        valid = not require_answer
    else:
        valid = (
            len(source_numbers) > 0
            and len(invalid_sources) == 0
            and len(missing_providers) == 0
            and len(citations) > 0
            and not raw_source_tokens_remain
        )

    return {
        "answer": final_answer,
        "raw_answer": raw_answer,
        "citations": citations,
        "source_numbers": source_numbers,
        "invalid_sources": invalid_sources,
        "cited_providers": cited_providers,
        "missing_providers": missing_providers,
        "refused": refused,
        "attempts": attempts,
        "valid": valid,
    }


# ============================================================
# CONVERSATION MEMORY
# ============================================================

def format_conversation_history(conversation_history):
    if not conversation_history:
        return "No previous conversation."

    recent_messages = conversation_history[-MEMORY_MAX_MESSAGES:]
    lines = []

    for message in recent_messages:
        role = message.get("role", "")
        content = message.get("content", "")

        if role == "user":
            label = "USER"
        elif role == "assistant":
            label = "ASSISTANT"
        else:
            continue

        lines.append(f"{label}: {content}")

    return "\n".join(lines)


def contextualize_question(question, conversation_history, llm):
    if not conversation_history:
        return question

    history_text = format_conversation_history(conversation_history)

    system_message = """
You rewrite follow-up questions into standalone search questions.

Use conversation history only to resolve references such as:
it, that, this, they, those, the first one, the second one,
that guidance, that provider, or that recommendation.

Rules:
1. Do not answer the question.
2. Do not add facts.
3. Preserve the user's meaning.
4. Make the question understandable without conversation history.
5. Return ONLY the rewritten standalone question.
""".strip()

    user_message = f"""
Conversation history:

{history_text}

Current question:

{question}

Standalone question:
""".strip()

    response = llm.invoke(
        [
            ("system", system_message),
            ("human", user_message),
        ]
    )

    standalone_question = response.content.strip().strip('"')
    return standalone_question or question


# ============================================================
# ROUTING
# ============================================================

def detect_provider_mentions(question):
    question_lower = question.lower()
    found = []

    for provider in EXPECTED_PROVIDERS:
        aliases = PROVIDER_ALIASES[provider]

        for alias in aliases:
            pattern = rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])"
            if re.search(pattern, question_lower):
                found.append(provider)
                break

    return found


def resolve_retrieval_mode(
    standalone_question,
    mode="Auto",
    provider=None,
    providers=None,
    document_id=None,
):
    providers = providers or []

    if mode == "All Providers":
        return "global", None, [], None

    if mode == "Single Provider":
        if not provider:
            raise ValueError("Select a provider for Single Provider mode.")
        return "provider", provider, [], None

    if mode == "Compare Providers":
        if len(providers) < 2:
            raise ValueError("Select at least two providers for comparison.")
        return "cross_provider", None, providers, None

    if mode == "Specific Document":
        if not document_id:
            raise ValueError("Select a document for Specific Document mode.")
        return "document", None, [], document_id

    detected = detect_provider_mentions(standalone_question)

    if len(detected) >= 2:
        return "cross_provider", None, detected, None

    if len(detected) == 1:
        return "provider", detected[0], [], None

    return "global", None, [], None


# ============================================================
# PUBLIC QUESTION API
# ============================================================

def ask_rag(
    rag,
    question,
    conversation_history=None,
    mode="Auto",
    provider=None,
    providers=None,
    document_id=None,
):
    conversation_history = conversation_history or []

    standalone_question = contextualize_question(
        question,
        conversation_history,
        rag["llm"],
    )

    (
        resolved_mode,
        resolved_provider,
        resolved_providers,
        resolved_document_id,
    ) = resolve_retrieval_mode(
        standalone_question,
        mode=mode,
        provider=provider,
        providers=providers,
        document_id=document_id,
    )

    required_providers = []
    require_answer = False

    if resolved_mode == "provider":
        evidence = retrieve(
            rag,
            standalone_question,
            provider=resolved_provider,
        )
        query_type = f"provider:{resolved_provider}"

    elif resolved_mode == "document":
        evidence = retrieve(
            rag,
            standalone_question,
            document_id=resolved_document_id,
        )
        query_type = f"document:{resolved_document_id}"

    elif resolved_mode == "cross_provider":
        evidence = retrieve_cross_provider(
            rag,
            standalone_question,
            resolved_providers,
        )
        required_providers = resolved_providers
        require_answer = True
        query_type = "cross_provider:" + ",".join(resolved_providers)

    else:
        evidence = retrieve(
            rag,
            standalone_question,
        )
        query_type = "global"

    result = generate_grounded_answer(
        rag,
        standalone_question,
        evidence,
        required_providers=required_providers,
        require_answer=require_answer,
    )

    result["question"] = question
    result["standalone_question"] = standalone_question
    result["query_type"] = query_type
    result["resolved_mode"] = resolved_mode
    result["resolved_provider"] = resolved_provider
    result["resolved_providers"] = resolved_providers
    result["resolved_document_id"] = resolved_document_id
    result["evidence"] = evidence

    return result


def evidence_for_memory(evidence):
    return [
        {
            "provider": result["provider"],
            "document_id": result["document_id"],
            "document_title": result["document_title"],
            "source": result["source"],
            "pdf_page": result["pdf_page"],
            "document_chunk_number": result["document_chunk_number"],
            "global_chunk_number": result["global_chunk_number"],
            "semantic_score": result["semantic_score"],
            "rerank_score": result["rerank_score"],
            "text": result["text"],
        }
        for result in evidence
    ]