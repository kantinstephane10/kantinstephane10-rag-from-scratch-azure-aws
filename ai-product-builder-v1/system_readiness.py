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
        'required_for_local_rag': ['PyMuPDF', 'sentence-transformers', 'langchain-text-splitters'],
        'production_ready': False,
        'production_gaps': [
            'User authentication, session management and project-level ACLs',
            'TLS reverse proxy and internet-facing hardening',
            'Dedicated authorized agent command runner and tool sandbox',
            'Distributed durable job queue, retries and crash recovery',
            'End-to-end corpus quality, access-revocation and scale benchmarks',
            'Centralized observability, backups, secret lifecycle and CI/CD',
            'Separate security and release approvals',
        ],
        'installation': {
            'local_rag': 'python -m pip install -r requirements-rag.txt',
            'local_ocr': 'python -m pip install -r requirements-ocr.txt',
            'optional_agents': 'python -m pip install -r requirements-agent-optional.txt',
        },
    }