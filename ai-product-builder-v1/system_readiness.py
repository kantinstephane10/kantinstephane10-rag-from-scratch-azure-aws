"""Local prerequisite report; does not claim production certification."""
import importlib.metadata
import shutil
import sys
from pathlib import Path

LIBRARIES = {
    'PyMuPDF': ('PyMuPDF', 'PDF text extraction'),
    'sentence-transformers': ('sentence-transformers', 'Local Hugging Face embeddings'),
    'langchain-text-splitters': ('langchain-text-splitters', 'Document chunking'),
    'rapidocr': ('rapidocr', 'Optional local scanned-PDF OCR'),
    'onnxruntime': ('onnxruntime', 'Optional local OCR inference'),
    'langchain-core': ('langchain-core', 'Optional LangChain agent integration'),
    'langchain-openai': ('langchain-openai', 'Optional LangChain chat adapter'),
    'langgraph': ('langgraph', 'Optional stateful agent workflows'),
    'langchain-chroma': ('langchain-chroma', 'Optional Chroma retrieval'),
    'chromadb': ('chromadb', 'Optional Chroma vector database'),
    'pytest': ('pytest', 'Developer test runner'),
}
PROGRAMS = {
    'git': 'Source control',
    'node': 'Optional JavaScript syntax and front-end tests',
    'tesseract': 'Optional alternative OCR executable',
}


def readiness(directory):
    packages = []
    for dist, (name, purpose) in LIBRARIES.items():
        try:
            version = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            version = None
        packages.append({'name': name, 'purpose': purpose, 'installed': version is not None, 'version': version})
    programs = [{'name': name, 'purpose': purpose, 'available': bool(shutil.which(name))}
                for name, purpose in PROGRAMS.items()]
    try:
        disk_free = shutil.disk_usage(str(Path(directory).resolve())).free
    except OSError:
        disk_free = None
    return {
        'scope': 'local-development',
        'python': sys.version.split()[0],
        'python_supported_target': sys.version_info[:2] >= (3, 11) and sys.version_info[:2] <= (3, 13),
        'libraries': packages,
        'programs': programs,
        'free_disk_bytes': disk_free,
        'website_builder': 'available (no additional Python packages)',
        'windows_python_host_runner': 'automatic, host-privileged, no per-script approval; NOT sandboxed',
        'multiuser_authentication': 'opt-in on localhost; authenticated sessions and project roles',
        'release_staging': 'human-approved atomic local website staging only',
        'metrics': 'in-process HTTP counters and capacity endpoint',
        'load_testing': 'non-destructive localhost health benchmark harness; not a full load certification',
        'required_for_local_rag': ['PyMuPDF', 'sentence-transformers', 'langchain-text-splitters'],
        'production_ready': False,
        'production_gaps': [
            'Public-facing TLS reverse proxy, stronger session controls, audited user lifecycle and authorization/revocation race testing',
            'TLS reverse proxy and internet-facing hardening',
            'Unattended Windows host execution is enabled by default and NOT sandboxed; it exposes the Windows account to prompt-injected or malicious generated code',
            'Distributed durable job queue, retries and crash recovery',
            'End-to-end corpus quality, access-revocation, 1 GiB processing and full-scale benchmarks',
            'Centralized durable metrics, protected backups, secret rotation and CI/CD',
            'Approved local website release staging exists; public deployment and separate security approval are not implemented',
        ],
        'installation': {
            'local_rag': 'python -m pip install -r requirements-rag.txt',
            'local_ocr': 'python -m pip install -r requirements-ocr.txt',
            'optional_agents': 'python -m pip install -r requirements-agent-optional.txt',
        },
    }
