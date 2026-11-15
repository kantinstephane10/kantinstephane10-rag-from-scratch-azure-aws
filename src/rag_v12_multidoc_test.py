from collections import Counter

from rag_v12_multidoc import (
    EXPECTED_PROVIDERS,
    build_document_chunk_lookup,
    discover_cloud_pdfs,
    get_document_neighbors,
    load_cloud_documents,
    chunk_cloud_documents,
)


# ============================================================
# RAG V12 - MULTI-DOCUMENT INGESTION TEST
#
# Stage 1 only.
# No embeddings or Pinecone writes yet.
# ============================================================


EXPECTED_COUNTS = {
    "AWS": 8,
    "Azure": 5,
    "GCP": 5,
    "Oracle": 4,
    "IBM": 5,
}

EXPECTED_TOTAL_PDFS = 27


def check(
    condition,
    label,
    results,
):
    if condition:
        print(
            f"{label}: PASS ✅"
        )
        results.append(
            True
        )
    else:
        print(
            f"{label}: FAIL ❌"
        )
        results.append(
            False
        )


def main():
    results = []

    # --------------------------------------------------------
    # DISCOVERY
    # --------------------------------------------------------

    pdf_records = (
        discover_cloud_pdfs()
    )

    provider_counts = Counter(
        record["provider"]
        for record in pdf_records
    )

    check(
        len(pdf_records)
        == EXPECTED_TOTAL_PDFS,
        "27 PDFs discovered",
        results,
    )

    check(
        set(provider_counts)
        == set(EXPECTED_PROVIDERS),
        "All 5 cloud providers discovered",
        results,
    )

    for provider in EXPECTED_PROVIDERS:
        check(
            provider_counts[provider]
            == EXPECTED_COUNTS[provider],
            (
                f"{provider} document count "
                f"= {EXPECTED_COUNTS[provider]}"
            ),
            results,
        )

    document_ids = [
        record["document_id"]
        for record in pdf_records
    ]

    check(
        len(document_ids)
        ==
        len(set(document_ids)),
        "Document IDs are unique",
        results,
    )

    # --------------------------------------------------------
    # PAGE METADATA
    # --------------------------------------------------------

    pages = load_cloud_documents()

    check(
        len(pages) > 0,
        "PDF pages loaded",
        results,
    )

    required_page_metadata = {
        "pdf_page",
        "provider",
        "document_id",
        "document_title",
        "source",
        "source_path",
    }

    check(
        all(
            required_page_metadata.issubset(
                page.metadata
            )
            for page in pages
        ),
        "Every page has V12 document metadata",
        results,
    )

    page_document_ids = {
        page.metadata[
            "document_id"
        ]
        for page in pages
    }

    check(
        len(page_document_ids)
        == EXPECTED_TOTAL_PDFS,
        "All 27 documents produced pages",
        results,
    )

    # --------------------------------------------------------
    # CHUNKS
    # --------------------------------------------------------

    chunks = chunk_cloud_documents(
        pages
    )

    check(
        len(chunks) > len(pages),
        "Documents split into chunks",
        results,
    )

    required_chunk_metadata = (
        required_page_metadata
        |
        {
            "document_chunk_number",
            "global_chunk_number",
            "chunk_characters",
            "chunk_id",
        }
    )

    check(
        all(
            required_chunk_metadata.issubset(
                chunk.metadata
            )
            for chunk in chunks
        ),
        "Every chunk has V12 chunk metadata",
        results,
    )

    chunk_ids = [
        chunk.metadata[
            "chunk_id"
        ]
        for chunk in chunks
    ]

    check(
        len(chunk_ids)
        ==
        len(set(chunk_ids)),
        "Chunk IDs are unique",
        results,
    )

    chunk_document_ids = {
        chunk.metadata[
            "document_id"
        ]
        for chunk in chunks
    }

    check(
        len(chunk_document_ids)
        == EXPECTED_TOTAL_PDFS,
        "All 27 documents produced chunks",
        results,
    )

    # --------------------------------------------------------
    # DOCUMENT-SAFE NEIGHBORS
    # --------------------------------------------------------

    lookup = (
        build_document_chunk_lookup(
            chunks
        )
    )

    check(
        len(lookup)
        == len(chunks),
        "Document chunk lookup is complete",
        results,
    )

    sample_chunk = chunks[
        len(chunks) // 2
    ]

    neighbors = get_document_neighbors(
        sample_chunk,
        lookup,
        neighbor_range=1,
    )

    sample_document_id = (
        sample_chunk.metadata[
            "document_id"
        ]
    )

    check(
        all(
            neighbor.metadata[
                "document_id"
            ]
            == sample_document_id
            for neighbor in neighbors
        ),
        "Neighbor expansion stays inside one PDF",
        results,
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    passed = sum(
        results
    )

    total = len(
        results
    )

    accuracy = (
        passed
        /
        total
        *
        100
    )

    print()
    print(
        "=" * 70
    )
    print(
        "RAG V12 MULTI-DOCUMENT INGESTION TEST"
    )
    print(
        "=" * 70
    )
    print(
        "Documents:",
        len(pdf_records),
    )
    print(
        "Pages:",
        len(pages),
    )
    print(
        "Chunks:",
        len(chunks),
    )
    print(
        f"Tests passed: {passed}/{total}"
    )
    print(
        f"Accuracy: {accuracy:.1f}%"
    )

    if passed == total:
        print(
            "V12 STAGE 1 MULTI-DOCUMENT INGESTION: PASS ✅"
        )
    else:
        print(
            "V12 STAGE 1 MULTI-DOCUMENT INGESTION: FAIL ❌"
        )
        raise SystemExit(
            1
        )


if __name__ == "__main__":
    main()
