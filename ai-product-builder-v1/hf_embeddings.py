"""Free, local Hugging Face embeddings for document and query vectors.

Only approved public models can be loaded; no remote code execution and no
OpenAI embedding calls. The first use downloads model weights into HF cache.
"""
from functools import lru_cache

from store import StoreError

MODELS = {
    'sentence-transformers/all-MiniLM-L6-v2': 'MiniLM L6 (fast; 384 dimensions)',
    'BAAI/bge-small-en-v1.5': 'BGE Small v1.5 (384 dimensions)',
}
DEFAULT_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
BGE_QUERY_INSTRUCTION = 'Represent this sentence for searching relevant passages: '


def validate_model(model_id):
    if not isinstance(model_id, str) or model_id not in MODELS:
        raise StoreError('Choose an approved local Hugging Face embedding model in Settings.')
    return model_id


@lru_cache(maxsize=2)
def _load(model_id):
    validate_model(model_id)
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise StoreError('Local embeddings require sentence-transformers. Install requirements-rag.txt.') from exc
    try:
        return SentenceTransformer(model_id, trust_remote_code=False, device='cpu')
    except Exception as exc:
        raise StoreError(f'Could not load local Hugging Face model ({type(exc).__name__}). Check cache/network and disk capacity.') from exc


def embed_texts(texts, model_id=DEFAULT_MODEL, *, query=False):
    validate_model(model_id)
    if not isinstance(texts, list) or any(not isinstance(t,str) or not t.strip() for t in texts):
        raise StoreError('Embedding input must be a list of nonempty strings.')
    if not texts:
        return []
    model = _load(model_id)
    try:
        batch = [BGE_QUERY_INSTRUCTION + x if query and model_id.startswith('BAAI/bge-') else x for x in texts]
        vectors = model.encode(batch, batch_size=16, normalize_embeddings=True, show_progress_bar=False)
        vectors = vectors.tolist() if hasattr(vectors, 'tolist') else vectors
        if len(vectors) != len(texts):
            raise StoreError('Local embedding model returned an incorrect vector count.')
        result = [list(map(float, v)) for v in vectors]
        dimensions = {len(v) for v in result}
        if len(dimensions) != 1 or not next(iter(dimensions)):
            raise StoreError('Local embedding dimensions are inconsistent.')
        return result
    except StoreError:
        raise
    except Exception as exc:
        raise StoreError(f'Local embedding inference failed ({type(exc).__name__}).') from exc
