"""Multi-Cloud RAG V14 production backend.

Frozen from the validated V14 notebook pipeline. V13 is not imported, modified,
or overwritten. This module keeps the V14 contracts that were validated in the
notebook: controlled metadata, document-aware stabilization, deterministic
query parsing, progressive metadata fallback, metadata-aware Dense + BM25,
RRF fusion, provider-balanced comparisons, same-document neighbor expansion,
CrossEncoder reranking, evidence-quality gating, grounded Ollama generation,
deterministic citations, and SQLite conversation memory.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import json
import os
import re
import time
from typing import Any, Iterable

import numpy as np
import requests
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

def _detect_project_root() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in (here, here.parent):
        if (candidate / "data" / "cloud_documents").exists():
            return candidate
    return here.parent


PROJECT_ROOT = _detect_project_root()
CLOUD_DOCUMENT_ROOT = PROJECT_ROOT / "data" / "cloud_documents"

EXPECTED_PROVIDERS = ["AWS", "Azure", "GCP", "Oracle", "IBM"]
EXPECTED_TOTAL_PDFS = 27
EXPECTED_TOTAL_PAGES = 3451
EXPECTED_TOTAL_CHUNKS = 11502

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
NEIGHBOR_RANGE = 1
DENSE_TOP_K = 20
BM25_TOP_K = 20
HYBRID_TOP_K = 10
FINAL_TOP_K = 5
RRF_K = 60
MIN_EVIDENCE_CHARS = 120
MINIMUM_TOTAL_CANDIDATES = 5
MINIMUM_PER_PROVIDER = 2
MEMORY_MAX_MESSAGES = 12

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
OLLAMA_MODEL = "qwen3:4b-instruct"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Retained for UI/operational continuity with V13/V12. V14 production Dense
# retrieval uses the validated local 384-dimensional normalized embedding cache.
PINECONE_INDEX_NAME = "multicloud-rag-v12"
PINECONE_NAMESPACE = "cloud-documents"

REFUSAL_MESSAGE = (
    "Sorry , but I don't have enough information in the provided documents "
    "to answer that question, Thank you !!."
)


# ============================================================
# V14 CONTROLLED METADATA TAXONOMY
# ============================================================

V14_TOPICS = [
    "compute",
    "storage",
    "networking",
    "security",
    "reliability",
    "iac_deployment",
    "cost",
    "governance",
    "migration",
    "observability",
    "sustainability",
]

CONTROLLED_DOCUMENT_TYPES = [
    "well_architected",
    "reference_architecture",
    "best_practices",
    "security_guide",
    "product_guide",
    "whitepaper",
    "blueprint",
    "cloud_adoption_guide",
    "implementation_guide",
    "general_guide",
]

CONTROLLED_CONTENT_TYPES = ["text", "table", "image", "diagram", "mixed"]

CONTROLLED_ARCHITECTURE_DOMAINS = [
    "infrastructure",
    "application",
    "network",
    "security",
    "data",
    "operations",
    "governance",
    "platform",
]

V14_TOPIC_SUBTOPICS = {
    "compute": [
        "virtual_machines", "containers", "kubernetes", "serverless", "autoscaling",
        "compute_scaling", "instance_selection",
    ],
    "storage": [
        "object_storage", "block_storage", "file_storage", "backup", "data_lifecycle",
        "storage_redundancy",
    ],
    "networking": [
        "virtual_network", "routing", "load_balancing", "firewall", "dns", "connectivity",
        "private_connectivity", "network_segmentation",
    ],
    "security": [
        "iam", "least_privilege", "encryption", "key_management", "network_security",
        "logging_auditing", "mfa", "secrets_management", "zero_trust", "threat_detection",
        "identity_federation",
    ],
    "reliability": [
        "high_availability", "resilience", "disaster_recovery", "multi_region",
        "fault_tolerance", "failover", "rto_rpo",
    ],
    "iac_deployment": [
        "infrastructure_as_code", "provisioning", "deployment", "templates", "automation",
        "version_control", "ci_cd",
    ],
    "cost": [
        "cost_optimization", "pricing", "rightsizing", "savings", "cost_allocation",
        "financial_management",
    ],
    "governance": [
        "policy", "organization", "compliance", "landing_zone", "tagging", "guardrails",
        "account_management",
    ],
    "migration": [
        "hybrid_cloud", "modernization", "workload_migration", "migration_planning",
        "application_migration",
    ],
    "observability": [
        "monitoring", "metrics", "logging", "tracing", "alerting", "incident_management",
    ],
    "sustainability": [
        "carbon_reduction", "energy_efficiency", "resource_efficiency",
        "sustainable_architecture", "workload_efficiency",
    ],
}

# Explicit query/classification keywords. Sustainability intentionally uses the
# hardened environmental terms from Cell 103; generic "sustainable" alone is
# not enough to classify arbitrary security/operations text as sustainability.
V14_TOPIC_KEYWORDS = {
    "compute": [
        "compute", "virtual machine", "virtual machines", "instance", "instances",
        "container", "containers", "kubernetes", "serverless", "autoscaling", "auto scaling",
        "lambda", "fargate", "ecs", "eks", "aks", "gke",
    ],
    "storage": [
        "storage", "object storage", "block storage", "file storage", "bucket", "blob",
        "s3", "ebs", "efs", "disk", "backup", "archive",
    ],
    "networking": [
        "network", "networking", "vpc", "vnet", "virtual network", "routing", "route table",
        "load balancing", "load balancer", "firewall", "dns", "connectivity", "vpn",
        "peering", "subnet", "private endpoint",
    ],
    "security": [
        "security", "secure", "iam", "identity", "access management", "least privilege",
        "mfa", "authentication", "authorization", "encryption", "key management", "kms",
        "firewall", "zero trust", "secrets", "audit", "threat",
    ],
    "reliability": [
        "reliability", "reliable", "high availability", "availability", "resilience",
        "resilient", "fault tolerance", "disaster recovery", "failover", "redundancy",
        "multi-region", "multi region", "rto", "rpo",
    ],
    "iac_deployment": [
        "infrastructure as code", "iac", "terraform", "cloudformation", "cloud formation",
        "bicep", "arm template", "arm templates", "deployment manager", "provisioning",
        "deployment", "templates", "cdk",
    ],
    "cost": [
        "cost", "cost optimization", "pricing", "price", "rightsizing", "right-sizing",
        "savings plan", "reserved instance", "finops", "financial management",
    ],
    "governance": [
        "governance", "policy", "policies", "organization", "organizations", "compliance",
        "landing zone", "guardrail", "tagging", "account management",
    ],
    "migration": [
        "migration", "migrate", "migrating", "hybrid cloud", "modernization", "modernize",
        "application migration", "workload migration",
    ],
    "observability": [
        "observability", "monitoring", "monitor", "monitored", "metrics", "logging", "logs", "tracing",
        "alerting", "alerts", "telemetry", "incident",
    ],
    "sustainability": [
        "environmental impact", "environmental sustainability", "energy efficiency",
        "energy efficient", "energy consumption", "energy use", "carbon footprint",
        "carbon emissions", "carbon emission", "carbon reduction", "greenhouse gas",
        "greenhouse gases", "renewable energy", "renewable power", "resource efficiency",
        "resource efficient", "workload efficiency", "workload placement", "hardware efficiency",
        "server utilization", "resource utilization", "sustainable architecture",
    ],
}

V14_SUBTOPIC_KEYWORDS = {
    "virtual_machines": ["virtual machine", "virtual machines", "vm", "instance", "ec2"],
    "containers": ["container", "containers", "ecs", "fargate"],
    "kubernetes": ["kubernetes", "eks", "aks", "gke"],
    "serverless": ["serverless", "lambda", "azure functions", "cloud functions"],
    "autoscaling": ["autoscaling", "auto scaling", "auto-scale", "scale out", "scale in"],
    "compute_scaling": ["scaling", "scale capacity", "elastic compute"],
    "instance_selection": ["instance type", "machine type", "vm size"],
    "object_storage": ["object storage", "s3", "blob storage", "cloud storage"],
    "block_storage": ["block storage", "ebs", "managed disk", "block volume"],
    "file_storage": ["file storage", "efs", "azure files", "filestore"],
    "backup": ["backup", "back up", "snapshot"],
    "data_lifecycle": ["lifecycle", "retention", "tiering"],
    "storage_redundancy": ["storage redundancy", "replication", "redundant storage"],
    "virtual_network": ["vpc", "vnet", "virtual network", "virtual cloud network", "vcn"],
    "routing": ["routing", "route table", "route tables", "router"],
    "load_balancing": ["load balancing", "load balancer", "alb", "nlb"],
    "firewall": ["firewall", "network firewall"],
    "dns": ["dns", "route 53", "cloud dns"],
    "connectivity": ["connectivity", "vpn", "direct connect", "expressroute", "interconnect"],
    "private_connectivity": ["private endpoint", "private link", "privatelink"],
    "network_segmentation": ["network segmentation", "segmentation", "subnet"],
    "iam": ["iam", "identity and access", "identity access management", "access management"],
    "least_privilege": ["least privilege", "minimum permissions", "minimum permission"],
    "encryption": ["encryption", "encrypt", "encrypted"],
    "key_management": ["key management", "kms", "key vault", "cloud kms"],
    "network_security": ["network security", "security group", "nsg", "network acl"],
    "logging_auditing": ["audit", "auditing", "cloudtrail", "activity log", "audit log"],
    "mfa": ["mfa", "multi-factor", "multifactor"],
    "secrets_management": ["secret", "secrets manager", "key vault secret"],
    "zero_trust": ["zero trust", "zero-trust"],
    "threat_detection": ["threat detection", "guardduty", "security command center"],
    "identity_federation": ["federation", "identity provider", "sso", "single sign-on"],
    "high_availability": ["high availability", "highly available", "availability zone"],
    "resilience": ["resilience", "resilient"],
    "disaster_recovery": ["disaster recovery", "recovery strategy", "dr strategy"],
    "multi_region": ["multi-region", "multi region", "multiple regions"],
    "fault_tolerance": ["fault tolerance", "fault-tolerant", "fault tolerant"],
    "failover": ["failover", "fail over"],
    "rto_rpo": ["rto", "rpo", "recovery time objective", "recovery point objective"],
    "infrastructure_as_code": ["infrastructure as code", "iac"],
    "provisioning": ["provisioning", "provision"],
    "deployment": ["deployment", "deploy"],
    "templates": ["template", "templates", "cloudformation", "bicep", "arm template"],
    "automation": ["automation", "automated", "automate"],
    "version_control": ["version control", "git"],
    "ci_cd": ["ci/cd", "cicd", "pipeline", "codepipeline", "github actions"],
    "cost_optimization": ["cost optimization", "cost optimize", "optimize cost"],
    "pricing": ["pricing", "price", "cost model"],
    "rightsizing": ["rightsizing", "right-sizing", "right size"],
    "savings": ["savings plan", "reserved instance", "commitment discount"],
    "cost_allocation": ["cost allocation", "chargeback", "showback"],
    "financial_management": ["financial management", "finops", "cloud financial management"],
    "policy": ["policy", "policies"],
    "organization": ["organization", "organizations", "organizational unit", "ou"],
    "compliance": ["compliance", "regulatory"],
    "landing_zone": ["landing zone", "landing zones"],
    "tagging": ["tagging", "tags"],
    "guardrails": ["guardrail", "guardrails"],
    "account_management": ["account management", "subscription management", "project management"],
    "hybrid_cloud": ["hybrid cloud", "hybrid-cloud"],
    "modernization": ["modernization", "modernize"],
    "workload_migration": ["workload migration", "migrate workload", "migration"],
    "migration_planning": ["migration planning", "migration plan", "migration wave"],
    "application_migration": ["application migration", "application migrate"],
    "monitoring": ["monitoring", "monitor", "monitored", "observability"],
    "metrics": ["metrics", "metric"],
    "logging": ["logging", "logs", "log aggregation"],
    "tracing": ["tracing", "distributed trace", "trace"],
    "alerting": ["alerting", "alerts", "alarm", "alarms"],
    "incident_management": ["incident management", "incident response", "operational event"],
    "carbon_reduction": ["carbon reduction", "carbon emissions", "carbon footprint", "greenhouse gas"],
    "energy_efficiency": ["energy efficiency", "energy efficient", "energy consumption", "energy use"],
    "resource_efficiency": ["resource efficiency", "resource efficient", "resource utilization"],
    "sustainable_architecture": ["sustainable architecture", "environmental sustainability", "environmental impact"],
    "workload_efficiency": ["workload efficiency", "workload placement", "hardware efficiency", "server utilization"],
}

V14_SUBTOPIC_TO_TOPIC = {
    subtopic: topic
    for topic, subtopics in V14_TOPIC_SUBTOPICS.items()
    for subtopic in subtopics
}

PROVIDER_ALIASES = {
    "AWS": ["aws", "amazon web services"],
    "Azure": ["azure", "microsoft azure"],
    "GCP": ["gcp", "google cloud", "google cloud platform"],
    "Oracle": ["oracle", "oci", "oracle cloud"],
    "IBM": ["ibm", "ibm cloud"],
}

V14_DOCUMENT_TOPIC_RULES = [
    ("cost optimization", ["cost"]),
    ("operational excellence", ["observability", "governance"]),
    ("performance efficiency", ["compute", "cost"]),
    ("reliability pillar", ["reliability"]),
    ("security pillar", ["security"]),
    ("security reference architecture", ["security", "networking"]),
    ("sustainability pillar", ["sustainability"]),
    ("well architected framework", ["reliability", "security", "cost", "observability", "governance"]),
    ("achieve reliability", ["reliability"]),
    ("cloud native applications architecture", ["compute", "networking", "reliability"]),
    ("digital transformation cloud adoption", ["governance", "migration"]),
    ("enterprise scale landing zones", ["governance", "networking", "security"]),
    ("hybrid cloud for enterprise architects", ["migration", "networking", "governance"]),
    ("ciso guide", ["security", "governance"]),
    ("cloud trust", ["security", "governance"]),
    ("enterprise foundations blueprint", ["governance", "security", "networking"]),
    ("infrastructure security design overview", ["security", "networking", "compute"]),
    ("nisc security", ["security"]),
    ("cloud object storage", ["storage"]),
    ("deployable architectures powervs", ["compute", "networking"]),
    ("hybrid cloud on power", ["migration", "compute"]),
    ("power cloud security", ["security"]),
    ("iam best practices", ["security"]),
    ("network firewall", ["networking", "security"]),
    ("network routing", ["networking"]),
    ("security architecture", ["security", "networking"]),
]

_RUNTIME: dict[str, Any] | None = None


# ============================================================
# INGESTION / CONTROLLED METADATA
# ============================================================

def clean_cloud_text(text: str) -> str:
    text = text.replace("\t", " ")
    text = re.sub(r"(?m)^\s*\d{1,4}\s*$", "", text)
    text = re.sub(r"[ ]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def title_from_filename(pdf_path: Path) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[_-]+", " ", pdf_path.stem)).strip()


def get_provider_from_path(pdf_path: Path) -> str:
    provider = pdf_path.relative_to(CLOUD_DOCUMENT_ROOT).parts[0]
    if provider not in EXPECTED_PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}")
    return provider


def infer_document_type(title: str) -> str:
    t = title.lower()
    if "well architected" in t or "pillar" in t:
        return "well_architected"
    if "reference architecture" in t or "architecture" in t:
        return "reference_architecture"
    if "best practices" in t:
        return "best_practices"
    if "security" in t and "guide" in t:
        return "security_guide"
    if "product guide" in t:
        return "product_guide"
    if "whitepaper" in t or "white paper" in t:
        return "whitepaper"
    if "blueprint" in t:
        return "blueprint"
    if "cloud adoption" in t:
        return "cloud_adoption_guide"
    if "implementation" in t:
        return "implementation_guide"
    return "general_guide"


def classify_document_topics(document_title: str) -> list[str]:
    t = (document_title or "").lower().strip()
    found: list[str] = []
    for phrase, topics in V14_DOCUMENT_TOPIC_RULES:
        if phrase in t:
            for topic in topics:
                if topic not in found:
                    found.append(topic)
    return found


def infer_architecture_domain(topic: str | None) -> str:
    mapping = {
        "compute": "infrastructure",
        "storage": "data",
        "networking": "network",
        "security": "security",
        "reliability": "operations",
        "iac_deployment": "platform",
        "cost": "governance",
        "governance": "governance",
        "migration": "platform",
        "observability": "operations",
        "sustainability": "infrastructure",
    }
    return mapping.get(topic, "infrastructure")


def _contains_term(text: str, term: str) -> bool:
    """Boundary-aware deterministic phrase match.

    Prevents short controlled terms such as ``ou`` or ``dr`` from matching
    inside unrelated words like ``cloud`` or ``provider``.
    """
    lowered = text.lower()
    escaped = re.escape(term.lower())
    return re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", lowered) is not None


def _term_hits(text: str, terms: list[str]) -> list[str]:
    return [term for term in terms if _contains_term(text, term)]


def classify_v14_chunk(text: str, document_title: str) -> dict[str, Any]:
    lowered_title = (document_title or "").lower()
    scores: dict[str, float] = {}
    hits: dict[str, dict[str, list[str]]] = {}

    for topic in V14_TOPICS:
        text_hits = _term_hits(text, V14_TOPIC_KEYWORDS[topic])
        title_hits = _term_hits(lowered_title, V14_TOPIC_KEYWORDS[topic])
        score = (3 * len(text_hits)) + (4 * len(title_hits))

        # Explicit sustainability documents are trusted as document context,
        # while generic security prose is not allowed to create sustainability.
        if topic == "sustainability" and "sustainability pillar" in lowered_title:
            score += 10
            title_hits = list(dict.fromkeys(title_hits + ["sustainability pillar"]))

        scores[topic] = float(score)
        hits[topic] = {"text_hits": text_hits, "title_hits": title_hits}

    ranked = sorted(scores.items(), key=lambda x: (-x[1], V14_TOPICS.index(x[0])))
    positive = [(topic, score) for topic, score in ranked if score > 0]

    primary_topic = positive[0][0] if positive else None
    best_score = positive[0][1] if positive else 0.0
    related_topics = [topic for topic, _ in positive[:5]]

    if best_score >= 6:
        confidence = "high"
    elif best_score >= 3:
        confidence = "medium"
    elif best_score > 0:
        confidence = "low"
    else:
        confidence = "unclassified"

    detected_subtopics: list[str] = []
    combined = f"{document_title}\n{text}".lower()
    for subtopic, terms in V14_SUBTOPIC_KEYWORDS.items():
        if any(term in combined for term in terms):
            mapped_topic = V14_SUBTOPIC_TO_TOPIC.get(subtopic)
            if mapped_topic in related_topics or mapped_topic == primary_topic:
                detected_subtopics.append(subtopic)

    return {
        "primary_topic": primary_topic,
        "related_topics": related_topics,
        "best_score": best_score,
        "confidence": confidence,
        "subtopics": detected_subtopics,
        "topic_scores": scores,
        "topic_hits": hits,
    }


def choose_effective_primary_topic(
    chunk_primary_topic: str | None,
    confidence: str,
    document_topics: list[str],
) -> str | None:
    if chunk_primary_topic is not None and confidence in {"high", "medium"}:
        return chunk_primary_topic
    if document_topics:
        return document_topics[0]
    return chunk_primary_topic


def build_effective_topics(
    effective_primary_topic: str | None,
    related_topics: list[str],
    document_topics: list[str],
    max_topics: int = 5,
) -> list[str]:
    combined: list[str] = []
    for topic in ([effective_primary_topic] if effective_primary_topic else []) + list(related_topics) + list(document_topics):
        if topic in V14_TOPICS and topic not in combined:
            combined.append(topic)
    return combined[:max_topics]


def discover_cloud_pdfs() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for pdf_path in sorted(CLOUD_DOCUMENT_ROOT.rglob("*.pdf")):
        if not pdf_path.is_file():
            continue
        provider = get_provider_from_path(pdf_path)
        title = title_from_filename(pdf_path)
        records.append({
            "path": pdf_path,
            "provider": provider,
            "document_id": f"{provider.lower()}__{slugify(pdf_path.stem)}",
            "document_title": title,
            "source": pdf_path.name,
            "document_type": infer_document_type(title),
            "document_topics": classify_document_topics(title),
        })
    return records


def load_cloud_documents(pdf_records: list[dict[str, Any]]) -> list[Any]:
    pages: list[Any] = []
    for record in pdf_records:
        loaded = PyMuPDFLoader(str(record["path"])).load()
        for page in loaded:
            page.page_content = clean_cloud_text(page.page_content)
            page.metadata.update({
                "pdf_page": int(page.metadata.get("page", 0)) + 1,
                "provider": record["provider"],
                "document_id": record["document_id"],
                "document_title": record["document_title"],
                "source": record["source"],
                "source_path": str(record["path"]),
                "document_type": record["document_type"],
                "document_topics": list(record["document_topics"]),
                "content_type": "text",
            })
        pages.extend(loaded)
    return pages


def create_chunks(pages: list[Any]) -> list[Any]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = [c for c in splitter.split_documents(pages) if c.page_content.strip()]
    per_document: Counter[str] = Counter()

    for global_number, chunk in enumerate(chunks):
        m = chunk.metadata
        document_id = m["document_id"]
        document_number = per_document[document_id]
        m["document_chunk_number"] = document_number
        m["global_chunk_number"] = global_number
        m["chunk_id"] = f"{document_id}__chunk_{document_number:05d}"

        raw = classify_v14_chunk(chunk.page_content, m["document_title"])
        document_topics = list(m.get("document_topics") or [])
        effective_primary = choose_effective_primary_topic(
            raw["primary_topic"], raw["confidence"], document_topics
        )
        effective_topics = build_effective_topics(
            effective_primary, raw["related_topics"], document_topics
        )

        m.update({
            "topic": effective_primary,
            "topics": effective_topics,
            "subtopics": raw["subtopics"],
            "classification_confidence": raw["confidence"],
            "classification_score": raw["best_score"],
            "document_topics": document_topics,
            "architecture_domain": infer_architecture_domain(effective_primary),
            "metadata_version": "v14",
        })
        per_document[document_id] += 1

    return chunks


def get_chunk_metadata(chunk: Any) -> dict[str, Any]:
    if hasattr(chunk, "metadata"):
        return chunk.metadata
    return chunk.get("metadata", chunk)


def get_chunk_text(chunk: Any) -> str:
    if hasattr(chunk, "page_content"):
        return chunk.page_content
    return chunk.get("text", chunk.get("page_content", ""))


def build_document_catalog(pdf_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ("provider", "document_id", "document_title", "source", "document_type", "document_topics")
    return [{k: record[k] for k in keys} for record in pdf_records]


def _tokenize_bm25(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def initialize_v14(force_reload: bool = False) -> dict[str, Any]:
    global _RUNTIME
    if _RUNTIME is not None and not force_reload:
        return _RUNTIME

    if not CLOUD_DOCUMENT_ROOT.exists():
        raise FileNotFoundError(f"Cloud document folder not found: {CLOUD_DOCUMENT_ROOT}")

    pdf_records = discover_cloud_pdfs()
    pages = load_cloud_documents(pdf_records)
    chunks = create_chunks(pages)

    if len(pdf_records) != EXPECTED_TOTAL_PDFS:
        raise RuntimeError(f"Expected {EXPECTED_TOTAL_PDFS} PDFs, found {len(pdf_records)}")
    if len(pages) != EXPECTED_TOTAL_PAGES:
        raise RuntimeError(f"Expected {EXPECTED_TOTAL_PAGES} pages, found {len(pages)}")
    if len(chunks) != EXPECTED_TOTAL_CHUNKS:
        raise RuntimeError(f"Expected {EXPECTED_TOTAL_CHUNKS} chunks, found {len(chunks)}")

    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    reranker = CrossEncoder(RERANKER_MODEL)

    texts = [get_chunk_text(c) for c in chunks]
    dense_embeddings = embedding_model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    tokenized = [_tokenize_bm25(text) for text in texts]
    bm25 = BM25Okapi(tokenized)

    chunk_lookup = {
        (c.metadata["document_id"], c.metadata["document_chunk_number"]): c
        for c in chunks
    }
    chunk_by_id = {c.metadata["chunk_id"]: c for c in chunks}
    chunk_index_by_id = {c.metadata["chunk_id"]: i for i, c in enumerate(chunks)}

    _RUNTIME = {
        "pdf_records": pdf_records,
        "pages": pages,
        "chunks": chunks,
        "v14_enriched_chunks": chunks,
        "document_catalog": build_document_catalog(pdf_records),
        "embedding_model": embedding_model,
        "dense_embeddings": dense_embeddings,
        "reranker": reranker,
        "bm25": bm25,
        "chunk_lookup": chunk_lookup,
        "chunk_by_id": chunk_by_id,
        "chunk_index_by_id": chunk_index_by_id,
    }
    return _RUNTIME


def _runtime() -> dict[str, Any]:
    return initialize_v14()


# ============================================================
# DETERMINISTIC QUERY METADATA PARSER
# ============================================================

def detect_provider_mentions(question: str) -> list[str]:
    q = question.lower()
    found: list[str] = []
    for provider in EXPECTED_PROVIDERS:
        for alias in PROVIDER_ALIASES[provider]:
            if re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", q):
                found.append(provider)
                break
    return found


def _detect_labels(text: str, mapping: dict[str, list[str]]) -> list[str]:
    return [label for label, terms in mapping.items() if any(_contains_term(text, term) for term in terms)]


def parse_v14_query_metadata(query: str) -> dict[str, Any]:
    normalized = re.sub(r"\s+", " ", query.lower().strip())
    providers = detect_provider_mentions(query)
    topics = _detect_labels(normalized, V14_TOPIC_KEYWORDS)
    subtopics = _detect_labels(normalized, V14_SUBTOPIC_KEYWORDS)

    # Query intent accepts the generic sustainability words. Corpus
    # classification intentionally does not (Cell 103 hardening).
    if any(_contains_term(normalized, term) for term in ["sustainability", "sustainable"]):
        if "sustainability" not in topics:
            topics.append("sustainability")

    # Keep subtopic/topic consistency deterministic.
    for subtopic in subtopics:
        topic = V14_SUBTOPIC_TO_TOPIC.get(subtopic)
        if topic and topic not in topics:
            topics.append(topic)

    comparison_terms = ["compare", "comparison", "versus", " vs ", "side by side", "across providers"]
    is_comparison = len(providers) >= 2 or any(term in f" {normalized} " for term in comparison_terms)
    generic_provider_scope = any(term in normalized for term in ["cloud providers", "providers side by side", "across providers"])

    if is_comparison and providers:
        retrieval_scope = "cross_provider"
    elif is_comparison and not providers:
        retrieval_scope = "all_providers"
    elif len(providers) == 1:
        retrieval_scope = "provider"
    else:
        retrieval_scope = "global"

    return {
        "original_query": query,
        "normalized_query": normalized,
        "providers": providers,
        "topics": topics,
        "subtopics": subtopics,
        "is_comparison": is_comparison,
        "comparison": is_comparison,
        "generic_provider_scope": generic_provider_scope,
        "retrieval_scope": retrieval_scope,
        "parser_version": "v14-rule-v1",
    }


def determine_v14_final_top_k(meta: dict[str, Any]) -> int:
    if meta["retrieval_scope"] == "cross_provider":
        return max(6, 3 * len(meta["providers"]))
    if meta["retrieval_scope"] == "all_providers":
        return 10
    return 5


# ============================================================
# METADATA FILTERING + PROGRESSIVE FALLBACK
# ============================================================

def filter_v14_metadata_pool(
    chunks: Iterable[Any] | None = None,
    providers: list[str] | None = None,
    topics: list[str] | None = None,
    subtopics: list[str] | None = None,
    document_id: str | None = None,
) -> list[Any]:
    chunks = list(chunks if chunks is not None else _runtime()["v14_enriched_chunks"])
    providers = providers or []
    topics = topics or []
    subtopics = subtopics or []
    result: list[Any] = []

    for chunk in chunks:
        m = get_chunk_metadata(chunk)
        if document_id is not None and m.get("document_id") != document_id:
            continue
        if providers and m.get("provider") not in providers:
            continue
        chunk_topics = set(m.get("topics") or [])
        chunk_subtopics = set(m.get("subtopics") or [])
        if topics and not chunk_topics.intersection(topics):
            continue
        if subtopics and not chunk_subtopics.intersection(subtopics):
            continue
        result.append(chunk)
    return result


def v14_candidate_pool_is_sufficient(
    candidates: list[Any],
    query_metadata: dict[str, Any],
    minimum_total: int | None = None,
    minimum_per_provider: int | None = None,
) -> dict[str, Any]:
    minimum_total = MINIMUM_TOTAL_CANDIDATES if minimum_total is None else minimum_total
    minimum_per_provider = MINIMUM_PER_PROVIDER if minimum_per_provider is None else minimum_per_provider
    counts = Counter(get_chunk_metadata(c).get("provider") for c in candidates)

    if query_metadata["retrieval_scope"] == "all_providers":
        required_providers = EXPECTED_PROVIDERS
    else:
        required_providers = query_metadata["providers"]

    provider_sufficient = all(counts.get(p, 0) >= minimum_per_provider for p in required_providers)
    return {
        "candidate_count": len(candidates),
        "provider_counts": dict(counts),
        "provider_sufficient": provider_sufficient,
        "sufficient": len(candidates) >= minimum_total and provider_sufficient,
    }


def _fallback_stages(meta: dict[str, Any]) -> list[dict[str, Any]]:
    providers, topics, subtopics = meta["providers"], meta["topics"], meta["subtopics"]
    stages = [{
        "stage": "strict", "providers": providers, "topics": topics,
        "subtopics": subtopics, "constraints_dropped": [],
    }]
    if subtopics:
        stages.append({
            "stage": "relax_subtopic", "providers": providers, "topics": topics,
            "subtopics": [], "constraints_dropped": ["subtopics"],
        })
    if topics:
        stages.append({
            "stage": "relax_topic", "providers": providers, "topics": [], "subtopics": [],
            "constraints_dropped": ["topics", "subtopics"] if subtopics else ["topics"],
        })
    stages.append({
        "stage": "global", "providers": [], "topics": [], "subtopics": [],
        "constraints_dropped": ["providers", "topics", "subtopics"],
    })

    unique: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for stage in stages:
        signature = (tuple(stage["providers"]), tuple(stage["topics"]), tuple(stage["subtopics"]))
        if signature not in seen:
            seen.add(signature)
            unique.append(stage)
    return unique


def build_v14_candidates_with_fallback(
    query: str,
    chunks: list[Any] | None = None,
    minimum_total: int | None = None,
    minimum_per_provider: int | None = None,
    document_id: str | None = None,
) -> dict[str, Any]:
    chunks = chunks if chunks is not None else _runtime()["v14_enriched_chunks"]
    meta = parse_v14_query_metadata(query)
    trace: list[dict[str, Any]] = []
    selected_stage: dict[str, Any] | None = None
    selected_candidates: list[Any] = []
    selected_sufficiency: dict[str, Any] | None = None

    for stage in _fallback_stages(meta):
        candidates = filter_v14_metadata_pool(
            chunks=chunks,
            providers=stage["providers"],
            topics=stage["topics"],
            subtopics=stage["subtopics"],
            document_id=document_id,
        )
        suff = v14_candidate_pool_is_sufficient(candidates, meta, minimum_total, minimum_per_provider)
        trace.append({**stage, **suff, "document_id": document_id})
        selected_stage, selected_candidates, selected_sufficiency = stage, candidates, suff
        if suff["sufficient"]:
            break

    assert selected_stage is not None and selected_sufficiency is not None
    return {
        "query": query,
        "query_metadata": meta,
        "document_id": document_id,
        "document_constraint_preserved": document_id is not None,
        "selected_stage": selected_stage["stage"],
        "fallback_applied": selected_stage["stage"] != "strict",
        "constraints_dropped": selected_stage["constraints_dropped"],
        "candidates": selected_candidates,
        "candidate_count": len(selected_candidates),
        "provider_counts": selected_sufficiency["provider_counts"],
        "sufficient": selected_sufficiency["sufficient"],
        "stage_trace": trace,
    }


def build_v14_fallback_candidate_tiers(
    query: str,
    chunks: list[Any] | None = None,
    minimum_total: int | None = None,
    minimum_per_provider: int | None = None,
    document_id: str | None = None,
) -> dict[str, Any]:
    chunks = chunks if chunks is not None else _runtime()["v14_enriched_chunks"]
    base = build_v14_candidates_with_fallback(
        query, chunks, minimum_total, minimum_per_provider, document_id
    )
    seen_ids: set[str] = set()
    tiers: list[dict[str, Any]] = []
    tier_map: dict[str, dict[str, Any]] = {}

    for priority, stage in enumerate(base["stage_trace"], start=1):
        stage_candidates = filter_v14_metadata_pool(
            chunks=chunks,
            providers=stage["providers"],
            topics=stage["topics"],
            subtopics=stage["subtopics"],
            document_id=document_id,
        )
        new_candidates: list[Any] = []
        for chunk in stage_candidates:
            cid = get_chunk_metadata(chunk)["chunk_id"]
            if cid in seen_ids:
                continue
            seen_ids.add(cid)
            new_candidates.append(chunk)
            tier_map[cid] = {
                "stage": stage["stage"],
                "priority": priority,
                "constraints_dropped": list(stage["constraints_dropped"]),
            }
        tiers.append({
            "stage": stage["stage"],
            "priority": priority,
            "constraints_dropped": list(stage["constraints_dropped"]),
            "document_id": document_id,
            "all_stage_candidate_count": len(stage_candidates),
            "new_candidate_count": len(new_candidates),
            "candidates": new_candidates,
        })

    combined = [c for tier in tiers for c in tier["candidates"]]
    strict = tiers[0]["candidates"] if tiers else []
    return {
        **base,
        "tiers": tiers,
        "strict_candidates": strict,
        "strict_candidate_count": len(strict),
        "supplemental_candidates": combined[len(strict):],
        "supplemental_candidate_count": max(0, len(combined) - len(strict)),
        "combined_candidates": combined,
        "combined_candidate_count": len(combined),
        "chunk_tier_by_id": tier_map,
        "selected_pool_candidate_count": base["candidate_count"],
    }


# ============================================================
# METADATA-AWARE DENSE + BM25 + RRF
# ============================================================

def _candidate_indices(candidates: list[Any]) -> np.ndarray:
    rag = _runtime()
    return np.asarray([
        rag["chunk_index_by_id"][get_chunk_metadata(c)["chunk_id"]]
        for c in candidates
    ], dtype=np.int64)


def _candidate_ids(candidates: list[Any]) -> set[str]:
    return {get_chunk_metadata(c)["chunk_id"] for c in candidates}


def _dense_search(query: str, candidates: list[Any], top_k: int) -> list[tuple[str, float]]:
    if not candidates:
        return []
    rag = _runtime()
    indices = _candidate_indices(candidates)
    qvec = rag["embedding_model"].encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    )[0].astype(np.float32)
    scores = rag["dense_embeddings"][indices] @ qvec
    order = np.argsort(-scores)[:top_k]
    result: list[tuple[str, float]] = []
    for local_pos in order:
        chunk = candidates[int(local_pos)]
        result.append((get_chunk_metadata(chunk)["chunk_id"], float(scores[int(local_pos)])))
    return result


def _bm25_search(query: str, candidates: list[Any], top_k: int) -> list[tuple[str, float]]:
    if not candidates:
        return []
    rag = _runtime()
    all_scores = rag["bm25"].get_scores(_tokenize_bm25(query))
    indices = _candidate_indices(candidates)
    scores = all_scores[indices]
    order = np.argsort(-scores)[:top_k]
    result: list[tuple[str, float]] = []
    for local_pos in order:
        score = float(scores[int(local_pos)])
        if score <= 0:
            continue        result.append((get_chunk_metadata(candidates[int(local_pos)])["chunk_id"], score))
    return result


def _rrf(dense: list[tuple[str, float]], bm25: list[tuple[str, float]]) -> list[tuple[str, float]]:
    scores: defaultdict[str, float] = defaultdict(float)
    for rank, (cid, _) in enumerate(dense, start=1):
        scores[cid] += 1.0 / (RRF_K + rank)
    for rank, (cid, _) in enumerate(bm25, start=1):
        scores[cid] += 1.0 / (RRF_K + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def rank_v14_fallback_pool(
    query: str,
    candidates: list[Any],
    fallback_stage: str,
    fallback_priority: int,
    constraints_dropped: list[str],
    evidence_tier: str | None = None,
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    rag = _runtime()
    dense = _dense_search(query, candidates, DENSE_TOP_K)
    bm25 = _bm25_search(query, candidates, BM25_TOP_K)
    fused = _rrf(dense, bm25)
    dense_rank = {cid: i for i, (cid, _) in enumerate(dense, start=1)}
    bm25_rank = {cid: i for i, (cid, _) in enumerate(bm25, start=1)}

    out: list[dict[str, Any]] = []
    for cid, rrf_score in fused:
        chunk = rag["chunk_by_id"].get(cid)
        if chunk is None:
            continue
        m = chunk.metadata
        out.append({
            "chunk": chunk,
            "chunk_id": cid,
            "provider": m["provider"],
            "document_id": m["document_id"],
            "document_title": m["document_title"],
            "source": m["source"],
            "pdf_page": m["pdf_page"],
            "document_chunk_number": m["document_chunk_number"],
            "global_chunk_number": m["global_chunk_number"],
            "topic": m.get("topic"),
            "topics": list(m.get("topics") or []),
            "subtopics": list(m.get("subtopics") or []),
            "document_topics": list(m.get("document_topics") or []),
            "classification_confidence": m.get("classification_confidence"),
            "architecture_domain": m.get("architecture_domain"),
            "text": chunk.page_content,
            "rrf_score": float(rrf_score),
            "dense_rank": dense_rank.get(cid),
            "bm25_rank": bm25_rank.get(cid),
            "retrieval_source": (
                "dense+bm25" if cid in dense_rank and cid in bm25_rank
                else "dense" if cid in dense_rank else "bm25"
            ),
            "evidence_tier": evidence_tier,
            "metadata_fallback_stage": fallback_stage,
            "metadata_fallback_priority": fallback_priority,
            "metadata_constraints_dropped": list(constraints_dropped),
        })
    return out


def v14_primary_topic_match(chunk: Any, query_topics: list[str]) -> bool:
    if not query_topics:
        return True
    return get_chunk_metadata(chunk).get("topic") in query_topics


def balanced_v14_provider_merge(
    provider_results: dict[str, list[dict[str, Any]]],
    provider_order: list[str],
    top_k: int,
) -> list[dict[str, Any]]:
    queues = {p: list(provider_results.get(p, [])) for p in provider_order}
    merged: list[dict[str, Any]] = []
    while len(merged) < top_k and any(queues.values()):
        for provider in provider_order:
            if queues[provider] and len(merged) < top_k:
                merged.append(queues[provider].pop(0))
    return merged


def v14_metadata_hybrid_retrieve(
    query: str,
    top_k: int = HYBRID_TOP_K,
    minimum_total: int | None = None,
    minimum_per_provider: int | None = None,
    document_id: str | None = None,
) -> dict[str, Any]:
    fallback = build_v14_fallback_candidate_tiers(
        query,
        minimum_total=minimum_total,
        minimum_per_provider=minimum_per_provider,
        document_id=document_id,
    )
    meta = fallback["query_metadata"]
    comparison = meta["retrieval_scope"] in {"cross_provider", "all_providers"} and document_id is None

    if not comparison:
        ordered: list[dict[str, Any]] = []
        for tier in fallback["tiers"]:
            ordered.extend(rank_v14_fallback_pool(
                query, tier["candidates"], tier["stage"], tier["priority"], tier["constraints_dropped"]
            ))
        results = ordered[:top_k]
        balance_mode = "document_fallback_tiered_rrf" if document_id else "fallback_tiered_rrf"
    else:
        provider_order = meta["providers"] or EXPECTED_PROVIDERS
        provider_results: dict[str, list[dict[str, Any]]] = {p: [] for p in provider_order}
        for tier in fallback["tiers"]:
            for provider in provider_order:
                pool = [c for c in tier["candidates"] if get_chunk_metadata(c).get("provider") == provider]
                primary = [c for c in pool if v14_primary_topic_match(c, meta["topics"])]
                primary_ids = _candidate_ids(primary)
                secondary = [c for c in pool if get_chunk_metadata(c)["chunk_id"] not in primary_ids]
                provider_results[provider].extend(rank_v14_fallback_pool(
                    query, primary, tier["stage"], tier["priority"], tier["constraints_dropped"], "strict_primary"
                ))
                provider_results[provider].extend(rank_v14_fallback_pool(
                    query, secondary, tier["stage"], tier["priority"], tier["constraints_dropped"], "relaxed_secondary"
                ))
        results = balanced_v14_provider_merge(provider_results, provider_order, top_k)
        balance_mode = "provider_balanced_fallback_tiered_rrf"

    # Remove duplicates deterministically.
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in results:
        if item["chunk_id"] in seen:
            continue
        seen.add(item["chunk_id"])
        deduped.append(item)
    results = deduped[:top_k]

    for i, item in enumerate(results, start=1):
        item["rank"] = i

    return {
        "query": query,
        "query_metadata": meta,
        "document_id": document_id,
        "document_constraint_preserved": document_id is not None,
        "results": results,
        "balance_mode": balance_mode,
        "fallback_applied": fallback["fallback_applied"],
        "fallback_selected_stage": fallback["selected_stage"],
        "fallback_stage_trace": fallback["tiers"],
        "fallback_strict_candidate_count": fallback["strict_candidate_count"],
        "fallback_supplemental_candidate_count": fallback["supplemental_candidate_count"],
        "candidate_count": fallback["combined_candidate_count"],
    }


# ============================================================
# SAME-DOCUMENT NEIGHBOR EXPANSION
# ============================================================

def v14_expand_hybrid_neighbors(
    query: str,
    anchor_top_k: int = 10,
    neighbor_range: int = 1,
    document_id: str | None = None,
) -> dict[str, Any]:
    rag = _runtime()
    retrieval = v14_metadata_hybrid_retrieve(query, top_k=anchor_top_k, document_id=document_id)
    anchors = retrieval["results"]
    anchor_lookup = {a["chunk_id"]: a for a in anchors}
    expanded: dict[str, dict[str, Any]] = {}

    for anchor in anchors:
        anchor_doc = anchor["document_id"]
        anchor_number = anchor["document_chunk_number"]
        for offset in range(-neighbor_range, neighbor_range + 1):
            chunk = rag["chunk_lookup"].get((anchor_doc, anchor_number + offset))
            if chunk is None:
                continue
            m = chunk.metadata
            if document_id is not None and m["document_id"] != document_id:
                continue
            cid = m["chunk_id"]
            if cid not in expanded:
                expanded[cid] = {
                    "chunk": chunk,
                    "chunk_id": cid,
                    "provider": m["provider"],
                    "document_id": m["document_id"],
                    "document_title": m["document_title"],
                    "source": m["source"],
                    "pdf_page": m["pdf_page"],
                    "document_chunk_number": m["document_chunk_number"],
                    "global_chunk_number": m["global_chunk_number"],
                    "topic": m.get("topic"),
                    "topics": list(m.get("topics") or []),
                    "subtopics": list(m.get("subtopics") or []),
                    "document_topics": list(m.get("document_topics") or []),
                    "classification_confidence": m.get("classification_confidence"),
                    "architecture_domain": m.get("architecture_domain"),
                    "text": chunk.page_content,
                    "is_anchor": False,
                    "direct_anchor_rank": None,
                    "best_anchor_rank": anchor["rank"],
                    "anchor_links": [],
                    "rrf_score": None,
                    "retrieval_source": None,
                    "evidence_tier": None,
                    "metadata_fallback_stage": None,
                    "metadata_fallback_priority": None,
                    "metadata_constraints_dropped": [],
                    "fallback_provenance_source": None,
                }
            item = expanded[cid]
            item["anchor_links"].append({
                "anchor_chunk_id": anchor["chunk_id"],
                "anchor_rank": anchor["rank"],
                "offset": offset,
            })
            item["best_anchor_rank"] = min(item["best_anchor_rank"], anchor["rank"])
            if offset == 0:
                item["is_anchor"] = True
                item["direct_anchor_rank"] = anchor["rank"]
                item["rrf_score"] = anchor.get("rrf_score")
                item["retrieval_source"] = anchor.get("retrieval_source")
                item["evidence_tier"] = anchor.get("evidence_tier")

    expanded_results: list[dict[str, Any]] = []
    for item in expanded.values():
        linked = []
        for link in item["anchor_links"]:
            anchor = anchor_lookup.get(link["anchor_chunk_id"])
            if anchor is not None:
                linked.append((anchor.get("metadata_fallback_priority") or 1, link["anchor_rank"], anchor))
        if linked:
            priority, _, best_anchor = min(linked, key=lambda x: (x[0], x[1]))
            item["metadata_fallback_stage"] = best_anchor.get("metadata_fallback_stage", "strict")
            item["metadata_fallback_priority"] = priority
            item["metadata_constraints_dropped"] = list(best_anchor.get("metadata_constraints_dropped") or [])
            if item["evidence_tier"] is None:
                item["evidence_tier"] = best_anchor.get("evidence_tier")
            item["fallback_provenance_source"] = "direct_anchor" if item["is_anchor"] else "neighbor_of_anchor"

        item["min_neighbor_distance"] = min(abs(link["offset"]) for link in item["anchor_links"])
        item["context_type"] = "hybrid_anchor" if item["is_anchor"] else "same_document_neighbor"
        expanded_results.append(item)

    expanded_results.sort(key=lambda item: (
        item.get("metadata_fallback_priority") or 999999,
        item["best_anchor_rank"],
        item["min_neighbor_distance"],
        item["document_chunk_number"],
        item["chunk_id"],
    ))

    return {
        **retrieval,
        "anchor_count": len(anchors),
        "anchors": anchors,
        "expanded_count": len(expanded_results),
        "expanded_results": expanded_results,
        "neighbor_range": neighbor_range,
    }


# ============================================================
# CROSSENCODER RERANKING + FINAL EVIDENCE
# ============================================================

def v14_crossencoder_rerank(
    query: str,
    final_top_k: int = FINAL_TOP_K,
    anchor_top_k: int = 10,
    neighbor_range: int = 1,
    document_id: str | None = None,
) -> dict[str, Any]:
    rag = _runtime()
    expanded = v14_expand_hybrid_neighbors(query, anchor_top_k, neighbor_range, document_id)
    candidates = [r for r in expanded["expanded_results"] if len(r["text"].strip()) >= MIN_EVIDENCE_CHARS]
    if not candidates:
        candidates = list(expanded["expanded_results"])
    if not candidates:
        return {
            **expanded,
            "results": [],
            "scored_count": 0,
            "rerank_mode": "document_fallback_aware_crossencoder" if document_id else "fallback_aware_global_crossencoder",
        }

    scores = rag["reranker"].predict([[query, r["text"]] for r in candidates])
    for result, score in zip(candidates, scores):
        result["rerank_score"] = float(score)

    candidates.sort(key=lambda x: (
        x.get("metadata_fallback_priority") or 99,
        -x["rerank_score"],
        x["best_anchor_rank"],
        x["min_neighbor_distance"],
    ))

    meta = expanded["query_metadata"]
    if document_id is None and meta["retrieval_scope"] in {"cross_provider", "all_providers"}:
        provider_order = meta["providers"] or EXPECTED_PROVIDERS
        grouped = {p: [r for r in candidates if r["provider"] == p] for p in provider_order}
        results = balanced_v14_provider_merge(grouped, provider_order, final_top_k)
        rerank_mode = "provider_balanced_fallback_aware_crossencoder"
    else:
        results = candidates[:final_top_k]
        rerank_mode = "document_fallback_aware_crossencoder" if document_id else "fallback_aware_global_crossencoder"

    for i, item in enumerate(results, start=1):
        item["rank"] = i

    return {
        **expanded,
        "results": results,
        "scored_count": len(candidates),
        "rerank_mode": rerank_mode,
    }


def build_v14_final_evidence(
    query: str,
    final_top_k: int = FINAL_TOP_K,
    document_id: str | None = None,
) -> dict[str, Any]:
    result = v14_crossencoder_rerank(
        query,
        final_top_k=final_top_k,
        anchor_top_k=max(10, final_top_k),
        neighbor_range=NEIGHBOR_RANGE,
        document_id=document_id,
    )
    evidence: list[dict[str, Any]] = []
    text_sources: Counter[str] = Counter()

    for source_number, item in enumerate(result["results"], start=1):
        repaired = dict(item)
        repaired["source_number"] = source_number
        if repaired.get("text"):
            repaired["text_contract_source"] = "existing_text"
        elif repaired.get("chunk") is not None:
            repaired["text"] = get_chunk_text(repaired["chunk"])
            repaired["text_contract_source"] = "embedded_chunk"
        else:
            repaired["text"] = ""
            repaired["text_contract_source"] = "missing"
        text_sources[repaired["text_contract_source"]] += 1
        evidence.append(repaired)

    return {
        **result,
        "evidence": evidence,
        "evidence_count": len(evidence),
        "provider_counts": dict(Counter(e["provider"] for e in evidence)),
        "text_contract_sources": dict(text_sources),
        "document_constraint_preserved": document_id is not None,
    }


# ============================================================
# EVIDENCE QUALITY GATE
# ============================================================

def evaluate_v14_evidence_quality(query: str, document_id: str | None = None) -> dict[str, Any]:
    meta = parse_v14_query_metadata(query)
    fallback = build_v14_fallback_candidate_tiers(query, document_id=document_id)
    strict_count = fallback["strict_candidate_count"]
    selected_stage = fallback["selected_stage"]
    constraints_dropped = list(fallback["constraints_dropped"])

    if document_id is not None:
        document_chunks = filter_v14_metadata_pool(document_id=document_id)
        if not document_chunks:
            status, allowed, mode = "insufficient_document_evidence", False, "blocked"
            reason = "The selected document is not present in the corpus."
        else:
            document_provider = get_chunk_metadata(document_chunks[0]).get("provider")
            if meta["providers"] and document_provider not in meta["providers"]:
                status, allowed, mode = "insufficient_provider_evidence", False, "blocked"
                reason = "The selected document does not contain evidence for the explicitly requested provider."
            elif meta["topics"] and "topics" in constraints_dropped and strict_count == 0:
                status, allowed, mode = "insufficient_topic_evidence", False, "blocked"
                reason = "The selected document contains no strict evidence for the requested topic."
            elif strict_count > 0:
                status, allowed, mode = "strong", True, "normal"
                reason = "Strict metadata constraints were satisfied inside the selected document."
            else:
                status, allowed, mode = "insufficient_document_evidence", False, "blocked"
                reason = "The selected document contains no strict evidence for this request."
    else:
        # Global/provider/cross-provider safety behavior validated in Cells 104-105.
        if meta["topics"] and "topics" in constraints_dropped and strict_count == 0:
            status, allowed, mode = "insufficient_topic_evidence", False, "block"
            reason = "No strict evidence matched the requested topic, and retrieval had to remove the topic constraint."
        elif strict_count > 0 or (not meta["topics"] and not meta["subtopics"]):
            status, allowed, mode = "strong", True, "normal"
            reason = "Strict metadata constraints were satisfied."
        else:
            status, allowed, mode = "relaxed", True, "normal_with_warning"
            reason = "Strict evidence was sparse, so controlled fallback was used."

    return {
        "query": query,
        "query_metadata": meta,
        "document_id": document_id,
        "quality_status": status,
        "status": status,
        "generation_allowed": allowed,
        "generation_allowed_by_gate": allowed,
        "generation_mode": mode,
        "strict_candidate_count": strict_count,
        "selected_stage": selected_stage,
        "fallback_applied": fallback["fallback_applied"],
        "constraints_dropped": constraints_dropped,
        "reason": reason,
        "document_constraint_preserved": document_id is not None,
    }


# ============================================================
# GROUNDED GENERATION + DETERMINISTIC CITATIONS
# ============================================================

def create_citation_label(result: dict[str, Any]) -> str:
    return f"[{result['provider']} | {result['document_title']} | PDF page {result['pdf_page']}]"


SOURCE_PATTERN = re.compile(r"\[SOURCE\s+(\d+)\]", re.IGNORECASE)


def normalize_source_markers(text: str) -> str:
    text = re.sub(r"\(\s*SOURCE\s+(\d+)\s*\)", r"[SOURCE \1]", text, flags=re.I)
    return re.sub(r"(?<!\[)\bSOURCE\s+(\d+)\b(?!\])", r"[SOURCE \1]", text, flags=re.I)


def validate_source_numbers(answer: str, evidence: list[dict[str, Any]]) -> tuple[list[int], list[int]]:
    numbers = [int(n) for n in SOURCE_PATTERN.findall(normalize_source_markers(answer))]
    unique_numbers = list(dict.fromkeys(numbers))
    valid = set(range(1, len(evidence) + 1))
    return unique_numbers, [n for n in unique_numbers if n not in valid]


def replace_source_markers(answer: str, evidence: list[dict[str, Any]]) -> tuple[str, list[str]]:
    answer = normalize_source_markers(answer)
    citations: list[str] = []

    def repl(match: re.Match[str]) -> str:
        n = int(match.group(1))
        if 1 <= n <= len(evidence):
            label = create_citation_label(evidence[n - 1])
            citations.append(label)
            return label
        return ""

    final = SOURCE_PATTERN.sub(repl, answer)
    return final.strip(), list(dict.fromkeys(citations))


def deduplicate_adjacent_inline_citations(answer: str) -> str:
    return re.sub(r"(\[[^\]]+\])(?:\1)+", r"\1", answer)


def build_context(evidence: list[dict[str, Any]]) -> str:
    blocks = []
    for source_number, result in enumerate(evidence, start=1):
        blocks.append(
            f"[SOURCE {source_number}]\n"
            f"Provider: {result['provider']}\n"
            f"Document: {result['document_title']}\n"
            f"PDF page: {result['pdf_page']}\n\n"
            f"Evidence:\n{result['text']}"
        )
    return "\n\n---------------------------------------\n\n".join(blocks)


def build_grounded_prompt(
    question: str,
    evidence: list[dict[str, Any]],
    required_providers: list[str] | None = None,
) -> str:
    required_providers = required_providers or []
    comparison = ""
    if required_providers:
        comparison = (
            "\nCROSS-PROVIDER COMPARISON RULES:\n"
            "The requested providers are:\n"
            + ", ".join(required_providers)
            + "\nDiscuss and cite every requested provider when usable evidence exists.\n"
        )

    return f"""
You are a grounded enterprise cloud RAG assistant.
Answer the user's question using ONLY the evidence supplied below.

GROUNDING RULES:
1. Use only information supported by the supplied evidence.
2. Every factual claim derived from the documents must include one or more citations using exactly [SOURCE N].
3. Never invent source numbers, cloud services, recommendations, capabilities, frameworks, or conclusions.
4. If evidence does not establish a point, say so instead of inventing an answer.
5. Do not rank providers merely because one has more retrieved text.
6. Be concise and avoid repeating a conclusion.
7. If there is genuinely not enough evidence, respond exactly:\n{REFUSAL_MESSAGE}

STRICT CITATION FORMAT:
Use [SOURCE 1][SOURCE 2], never [SOURCE 1, SOURCE 2].
Never use negative or unavailable source numbers.
{comparison}
QUESTION:\n{question}\n\nEVIDENCE:\n{build_context(evidence)}\n\nANSWER:
""".strip()


def stream_grounded_ollama_answer(
    question: str,
    evidence: list[dict[str, Any]],
    required_providers: list[str] | None = None,
) -> dict[str, Any]:
    prompt = build_grounded_prompt(question, evidence, required_providers)
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "options": {"temperature": 0.1},
    }
    start = time.perf_counter()
    first_token = None
    parts: list[str] = []
    with requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, stream=True, timeout=300) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            data = json.loads(line.decode("utf-8"))
            text = data.get("message", {}).get("content", "")
            if text:
                if first_token is None:
                    first_token = time.perf_counter() - start
                parts.append(text)
    return {
        "raw_answer": "".join(parts),
        "time_to_first_token": first_token,
        "total_time": time.perf_counter() - start,
    }


def ask_v14(question: str, show_debug: bool = True, document_id: str | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    meta = parse_v14_query_metadata(question)
    quality = evaluate_v14_evidence_quality(question, document_id=document_id)
    final_top_k = determine_v14_final_top_k(meta)

    if meta["retrieval_scope"] in {"cross_provider", "all_providers"}:
        required_providers = meta["providers"] or EXPECTED_PROVIDERS
    else:
        required_providers = meta["providers"][:1]

    if show_debug:
        print("=" * 80)
        print("V14 QUERY UNDERSTANDING")
        print("=" * 80)
        print("Question:", question)
        print("Providers:", meta["providers"])
        print("Topics:", meta["topics"])
        print("Subtopics:", meta["subtopics"])
        print("Scope:", "document" if document_id else meta["retrieval_scope"])
        print("Quality:", quality["quality_status"])
        print("Final evidence target:", final_top_k)

    if not quality["generation_allowed"]:
        return {
            "question": question,
            "answer": REFUSAL_MESSAGE,
            "raw_answer": REFUSAL_MESSAGE,
            "parsed_metadata": meta,
            "query_metadata": meta,
            "document_id": document_id,
            "document_constraint_preserved": document_id is not None,
            "evidence": [],
            "evidence_count": 0,
            "citations": [],
            "citation_labels": [],
            "source_numbers": [],
            "valid_source_numbers": [],
            "invalid_source_numbers": [],
            "citation_valid": None,
            "retrieved_providers": [],
            "required_providers": required_providers,
            "quality": quality,
            "quality_gate": quality,
            "quality_status": quality["quality_status"],
            "generation_mode": quality["generation_mode"],
            "generation_allowed": False,
            "generation_skipped": True,
            "time_to_first_token": None,
            "generation_time": 0.0,
            "total_time": time.perf_counter() - started,
        }

    evidence_result = build_v14_final_evidence(
        question, final_top_k=final_top_k, document_id=document_id
    )
    evidence = evidence_result["evidence"]
    if not evidence:
        quality = {**quality, "quality_status": "insufficient_evidence", "status": "insufficient_evidence"}
        return {
            "question": question,
            "answer": REFUSAL_MESSAGE,
            "raw_answer": REFUSAL_MESSAGE,
            "parsed_metadata": meta,
            "query_metadata": meta,
            "document_id": document_id,
            "document_constraint_preserved": document_id is not None,
            "evidence": [],
            "evidence_count": 0,
            "citations": [],
            "citation_labels": [],
            "source_numbers": [],
            "valid_source_numbers": [],
            "invalid_source_numbers": [],
            "citation_valid": False,
            "retrieved_providers": [],
            "required_providers": required_providers,
            "quality": quality,
            "quality_gate": quality,
            "quality_status": quality["quality_status"],
            "generation_mode": "block",
            "generation_allowed": False,
            "generation_skipped": True,
            "time_to_first_token": None,
            "generation_time": 0.0,
            "total_time": time.perf_counter() - started,
        }

    # Hard document post-condition.
    if document_id is not None and any(e["document_id"] != document_id for e in evidence):
        raise RuntimeError("Document safety violation: evidence escaped the selected document.")

    stream = stream_grounded_ollama_answer(
        question,
        evidence,
        required_providers if len(required_providers) > 1 else None,
    )
    raw_answer = stream["raw_answer"]
    source_numbers, invalid = validate_source_numbers(raw_answer, evidence)
    final_answer, citations = replace_source_markers(raw_answer, evidence)
    final_answer = deduplicate_adjacent_inline_citations(final_answer)
    cited_providers = {evidence[n - 1]["provider"] for n in source_numbers if 1 <= n <= len(evidence)}
    missing = set(required_providers) - cited_providers
    citation_valid = bool(source_numbers) and not invalid and bool(citations) and not missing

    return {
        "question": question,
        "answer": final_answer,
        "raw_answer": raw_answer,
        "parsed_metadata": meta,
        "query_metadata": meta,
        "document_id": document_id,
        "document_constraint_preserved": document_id is not None,
        "evidence": evidence,
        "evidence_count": len(evidence),
        "citations": citations,
        "citation_labels": citations,
        "source_numbers": source_numbers,
        "valid_source_numbers": source_numbers,
        "invalid_source_numbers": invalid,
        "citation_valid": citation_valid,
        "retrieved_providers": sorted({e["provider"] for e in evidence}),
        "required_providers": required_providers,
        "providers_cited": sorted(cited_providers),
        "quality": quality,
        "quality_gate": quality,
        "quality_status": quality["quality_status"],
        "generation_mode": quality["generation_mode"],
        "generation_allowed": True,
        "generation_skipped": False,
        "rerank_mode": evidence_result.get("rerank_mode"),
        "time_to_first_token": stream["time_to_first_token"],
        "generation_time": stream["total_time"],
        "total_time": time.perf_counter() - started,
    }


# ============================================================
# SQLITE CONVERSATION MEMORY
# ============================================================

def _memory_helpers():
    try:
        from memory_db import MEMORY_DB_PATH, auto_title_conversation, load_recent_memory, save_message
    except ImportError:
        try:
            from .memory_db import MEMORY_DB_PATH, auto_title_conversation, load_recent_memory, save_message
        except ImportError as exc:
            raise RuntimeError("memory_db.py must be available beside rag_v14.py") from exc
    return MEMORY_DB_PATH, auto_title_conversation, load_recent_memory, save_message


def _format_history(history: list[dict[str, Any]]) -> str:
    lines = []
    for msg in history[-MEMORY_MAX_MESSAGES:]:
        role = msg.get("role", "").upper()
        if role in {"USER", "ASSISTANT"}:
            lines.append(f"{role}: {msg.get('content', '')}")
    return "\n".join(lines)


def rewrite_followup_question(question: str, history: list[dict[str, Any]]) -> str:
    if not history:
        return question
    prompt = f"""
Rewrite the current follow-up question into one standalone cloud-search question.
Use history only to resolve references. Preserve provider, topic, subtopic, and comparison intent.
Do not answer. Return only the rewritten question.

HISTORY:\n{_format_history(history)}\n\nCURRENT QUESTION:\n{question}\n\nSTANDALONE QUESTION:
""".strip()
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.0},
    }
    response = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=120)
    response.raise_for_status()
    rewritten = response.json().get("message", {}).get("content", "").strip().strip('"')
    return rewritten or question


def _apply_mode_scope(
    question: str,
    mode: str,
    provider: str | None,
    providers: list[str] | None,
) -> str:
    providers = providers or []
    if mode == "provider" and provider and provider.lower() not in question.lower():
        return f"{provider}: {question}"
    if mode == "cross_provider" and providers:
        if not all(p.lower() in question.lower() for p in providers):
            return f"Compare {' and '.join(providers)}. {question}"
    return question


def evidence_for_memory_v14(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "provider", "document_id", "document_title", "source", "pdf_page",
        "document_chunk_number", "text", "rrf_score", "rerank_score",
        "metadata_fallback_stage", "metadata_fallback_priority",
        "metadata_constraints_dropped", "context_type",
    )
    return [{key: item.get(key) for key in fields} for item in evidence]


def ask_v14_with_memory(
    conversation_id: str,
    question: str,
    mode: str = "global",
    provider: str | None = None,
    providers: list[str] | None = None,
    document_id: str | None = None,
    show_debug: bool = True,
) -> dict[str, Any]:
    if mode == "document" and not document_id:
        raise ValueError("document_id is required when mode='document'.")

    db_path, auto_title_conversation, load_recent_memory, save_message = _memory_helpers()
    history = load_recent_memory(conversation_id, db_path)
    standalone = rewrite_followup_question(question, history)
    scoped = _apply_mode_scope(standalone, mode, provider, providers)
    result = ask_v14(scoped, show_debug=show_debug, document_id=document_id)

    save_message(
        conversation_id=conversation_id,
        role="user",
        content=question,
        query_type=mode,
        standalone_question=standalone,
        sources=[],
        db_path=db_path,
    )
    auto_title_conversation(conversation_id, question, db_path)
    save_message(
        conversation_id=conversation_id,
        role="assistant",
        content=result["answer"],
        query_type=mode,
        standalone_question=standalone,
        sources=evidence_for_memory_v14(result["evidence"]),
        db_path=db_path,
    )

    return {
        **result,
        "conversation_id": conversation_id,
        "question": question,
        "standalone_question": standalone,
        "scoped_question": scoped,
        "mode": mode,
        "document_id": document_id,
    }


# Backward-friendly alias for app code.
initialize_rag = initialize_v14