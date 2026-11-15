from collections import Counter
from pathlib import Path
import re

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# RAG V12 - MULTI-DOCUMENT INGESTION
#
# Stage 1:
#   - discover many PDFs recursively
#   - identify cloud provider from folder
#   - create document-level metadata
#   - create page-level metadata
#   - chunk every document
#   - create document-safe chunk IDs
#   - create document-safe neighbor lookup
#
# IMPORTANT:
# Neighbor expansion must never cross from one PDF into another.
# ============================================================


CLOUD_DOCUMENT_ROOT = Path(
    "data/cloud_documents"
)

EXPECTED_PROVIDERS = [
    "AWS",
    "Azure",
    "GCP",
    "Oracle",
    "IBM",
]

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


def clean_cloud_text(text):
    text = text.replace("\t", " ")

    text = re.sub(
        r"(?m)^\s*\d{1,4}\s*$",
        "",
        text,
    )

    text = re.sub(
        r"[ ]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


def slugify(text):
    value = text.lower()

    value = re.sub(
        r"[^a-z0-9]+",
        "-",
        value,
    )

    return value.strip("-")


def title_from_filename(pdf_path):
    title = pdf_path.stem

    title = re.sub(
        r"[_-]+",
        " ",
        title,
    )

    title = re.sub(
        r"\s+",
        " ",
        title,
    )

    return title.strip()


def get_provider_from_path(pdf_path):
    try:
        relative_path = pdf_path.relative_to(
            CLOUD_DOCUMENT_ROOT
        )
    except ValueError:
        raise ValueError(
            f"PDF is outside the V12 corpus: {pdf_path}"
        )

    if len(relative_path.parts) < 2:
        raise ValueError(
            "Every V12 PDF must live inside a provider folder. "
            "Example: data/cloud_documents/AWS/file.pdf. "
            f"Problem file: {pdf_path}"
        )

    provider = relative_path.parts[0]

    if provider not in EXPECTED_PROVIDERS:
        raise ValueError(
            f"Unknown provider folder '{provider}' "
            f"for PDF: {pdf_path}"
        )

    return provider


def discover_cloud_pdfs():
    if not CLOUD_DOCUMENT_ROOT.exists():
        raise FileNotFoundError(
            "Cloud document folder not found: "
            f"{CLOUD_DOCUMENT_ROOT}"
        )

    pdf_files = sorted(
        path
        for path in CLOUD_DOCUMENT_ROOT.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower() == ".pdf"
        )
    )

    if not pdf_files:
        raise FileNotFoundError(
            "No PDFs were found under "
            f"{CLOUD_DOCUMENT_ROOT}"
        )

    records = []

    for pdf_path in pdf_files:
        provider = get_provider_from_path(
            pdf_path
        )

        document_id = (
            f"{provider.lower()}__"
            f"{slugify(pdf_path.stem)}"
        )

        records.append(
            {
                "path": pdf_path,
                "provider": provider,
                "document_id": document_id,
                "document_title": title_from_filename(
                    pdf_path
                ),
                "source": pdf_path.name,
            }
        )

    return records


def load_cloud_documents():
    pdf_records = discover_cloud_pdfs()

    all_pages = []

    provider_page_counts = Counter()

    print(
        "PDF documents discovered:",
        len(pdf_records),
    )

    for document_number, record in enumerate(
        pdf_records,
        start=1,
    ):
        print(
            f"[{document_number}/{len(pdf_records)}] "
            f"{record['provider']} - "
            f"{record['source']}"
        )

        loader = PyMuPDFLoader(
            str(record["path"])
        )

        pages = loader.load()

        for page in pages:
            page.page_content = clean_cloud_text(
                page.page_content
            )

            pdf_page = (
                int(
                    page.metadata.get(
                        "page",
                        0,
                    )
                )
                + 1
            )

            page.metadata[
                "pdf_page"
            ] = pdf_page

            page.metadata[
                "provider"
            ] = record["provider"]

            page.metadata[
                "document_id"
            ] = record["document_id"]

            page.metadata[
                "document_title"
            ] = record["document_title"]

            page.metadata[
                "source"
            ] = record["source"]

            page.metadata[
                "source_path"
            ] = str(
                record["path"]
            )

            provider_page_counts[
                record["provider"]
            ] += 1

        all_pages.extend(
            pages
        )

    print(
        "Total pages loaded:",
        len(all_pages),
    )

    for provider in EXPECTED_PROVIDERS:
        print(
            f"{provider} pages:",
            provider_page_counts[
                provider
            ],
        )

    return all_pages


def chunk_cloud_documents(
    pages,
):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = splitter.split_documents(
        pages
    )

    chunks = [
        chunk
        for chunk in chunks
        if chunk.page_content.strip()
    ]

    per_document_counter = Counter()

    for global_number, chunk in enumerate(
        chunks
    ):
        document_id = chunk.metadata[
            "document_id"
        ]

        document_chunk_number = (
            per_document_counter[
                document_id
            ]
        )

        chunk.metadata[
            "document_chunk_number"
        ] = document_chunk_number

        chunk.metadata[
            "global_chunk_number"
        ] = global_number

        chunk.metadata[
            "chunk_characters"
        ] = len(
            chunk.page_content
        )

        chunk.metadata[
            "chunk_id"
        ] = (
            f"{document_id}__chunk_"
            f"{document_chunk_number:05d}"
        )

        per_document_counter[
            document_id
        ] += 1

    print(
        "Final V12 chunks:",
        len(chunks),
    )

    print(
        "Documents represented in chunks:",
        len(
            per_document_counter
        ),
    )

    return chunks


def build_document_chunk_lookup(
    chunks,
):
    lookup = {}

    for chunk in chunks:
        key = (
            chunk.metadata[
                "document_id"
            ],
            chunk.metadata[
                "document_chunk_number"
            ],
        )

        lookup[key] = chunk

    return lookup


def get_document_neighbors(
    chunk,
    chunk_lookup,
    neighbor_range=1,
):
    document_id = chunk.metadata[
        "document_id"
    ]

    current_number = chunk.metadata[
        "document_chunk_number"
    ]

    neighbors = []

    for offset in range(
        -neighbor_range,
        neighbor_range + 1,
    ):
        key = (
            document_id,
            current_number + offset,
        )

        neighbor = chunk_lookup.get(
            key
        )

        if neighbor is not None:
            neighbors.append(
                neighbor
            )

    return neighbors


def prepare_multidocument_corpus():
    pages = load_cloud_documents()

    chunks = chunk_cloud_documents(
        pages
    )

    chunk_lookup = (
        build_document_chunk_lookup(
            chunks
        )
    )

    return {
        "pages": pages,
        "chunks": chunks,
        "chunk_lookup": chunk_lookup,
    }


if __name__ == "__main__":
    corpus = prepare_multidocument_corpus()

    print()
    print(
        "V12 multi-document ingestion ready."
    )
    print(
        "Pages:",
        len(corpus["pages"]),
    )
    print(
        "Chunks:",
        len(corpus["chunks"]),
    )
