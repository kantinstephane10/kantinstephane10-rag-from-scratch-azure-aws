"""Small production smoke test for rag_v14.py.

Run from the directory containing rag_v14.py:
    python test_v14_production.py

Use --full only after Ollama/models/documents are available locally:
    python test_v14_production.py --full
"""

from __future__ import annotations

import sys

import rag_v14 as v14


PARSER_CASES = [
    ("What are AWS IAM best practices?", ["AWS"], ["security"], ["iam"], "provider", 5),
    ("How does Azure achieve high availability?", ["Azure"], ["reliability"], ["high_availability"], "provider", 5),
    ("Compare AWS and Azure infrastructure as code.", ["AWS", "Azure"], ["iac_deployment"], ["infrastructure_as_code"], "cross_provider", 6),
    ("What storage guidance does IBM provide?", ["IBM"], ["storage"], [], "provider", 5),
    ("Explain Oracle network routing best practices.", ["Oracle"], ["networking"], ["routing"], "provider", 5),
    ("What does AWS recommend for sustainability?", ["AWS"], ["sustainability"], [], "provider", 5),
    ("How should cloud workloads be monitored?", [], ["observability"], ["monitoring"], "global", 5),
    ("What does GCP say about encryption and key management?", ["GCP"], ["security"], ["encryption", "key_management"], "provider", 5),
    ("Explain disaster recovery strategies.", [], ["reliability"], ["disaster_recovery"], "global", 5),
    ("Compare cloud providers for reliability side by side.", [], ["reliability"], [], "all_providers", 10),
]


def main() -> None:
    print("=" * 80)
    print("V14 PRODUCTION SMOKE TEST")
    print("=" * 80)

    assert v14.V14_TOPICS == [
        "compute", "storage", "networking", "security", "reliability",
        "iac_deployment", "cost", "governance", "migration",
        "observability", "sustainability",
    ]
    print("11-topic controlled taxonomy: PASS ✅")

    passed = 0
    for question, providers, topics, subtopics, scope, top_k in PARSER_CASES:
        meta = v14.parse_v14_query_metadata(question)
        actual = (
            meta["providers"], meta["topics"], meta["subtopics"],
            meta["retrieval_scope"], v14.determine_v14_final_top_k(meta),
        )
        expected = (providers, topics, subtopics, scope, top_k)
        ok = actual == expected
        passed += int(ok)
        print(("PASS ✅" if ok else "FAIL ❌"), question)
        if not ok:
            print(" expected:", expected)
            print(" actual:  ", actual)

    print(f"Parser tests: {passed}/{len(PARSER_CASES)}")
    assert passed == len(PARSER_CASES)

    if "--full" not in sys.argv:
        print("\nStatic/parser smoke test complete. Use --full for corpus initialization.")
        return

    rag = v14.initialize_v14()
    assert len(rag["pdf_records"]) == 27
    assert len(rag["pages"]) == 3451
    assert len(rag["chunks"]) == 11502
    assert rag["dense_embeddings"].shape == (11502, 384)
    print("27 PDFs: PASS ✅")
    print("3,451 pages: PASS ✅")
    print("11,502 chunks: PASS ✅")
    print("Dense cache 11,502 × 384: PASS ✅")

    result = v14.build_v14_final_evidence("What are AWS IAM best practices?", final_top_k=5)
    assert len(result["evidence"]) == 5
    assert all(item["provider"] == "AWS" for item in result["evidence"])
    print("AWS IAM retrieval Top 5: PASS ✅")

    blocked = v14.ask_v14("What does GCP recommend for sustainability?", show_debug=False)
    assert blocked["generation_skipped"] is True
    assert blocked["quality_status"] == "insufficient_topic_evidence"
    print("Unsupported GCP sustainability blocked: PASS ✅")

    print("\nV14 production smoke test: PASS ✅")


if __name__ == "__main__":
    main()