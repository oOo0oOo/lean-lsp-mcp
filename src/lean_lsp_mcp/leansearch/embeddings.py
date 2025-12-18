"""Embedding providers for semantic search.

Supports multiple backends:
- sentence-transformers (default, no API key required)
- OpenAI text-embedding-3-large (best quality)
- Google text-embedding-004 (good quality, free tier)
- Voyage AI voyage-code-2 (excellent for code)
"""

from __future__ import annotations

import logging
import os
from typing import Protocol, Callable
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingFunction(Protocol):
    """Protocol for embedding functions."""

    def __call__(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts into vectors."""
        ...


def get_embedding_function(
    provider: str = "default",
    model: str | None = None,
) -> tuple[Callable[[list[str]], np.ndarray], int]:
    """Get embedding function and its output dimension.

    Args:
        provider: One of 'default', 'openai', 'gemini', 'voyage'
        model: Model name override

    Returns:
        Tuple of (embedding_function, embedding_dimension)
    """
    if provider == "openai":
        return _get_openai_embeddings(model)
    elif provider == "gemini":
        return _get_gemini_embeddings(model)
    elif provider == "voyage":
        return _get_voyage_embeddings(model)
    else:
        return _get_local_embeddings(model)


def _get_local_embeddings(
    model: str | None = None,
) -> tuple[Callable[[list[str]], np.ndarray], int]:
    """Get sentence-transformers embeddings (local, no API key)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise RuntimeError(
            "sentence-transformers is required for local embeddings. "
            "Install with: pip install sentence-transformers"
        )

    model_name = model or "all-MiniLM-L6-v2"
    logger.info(f"Using local sentence-transformers: {model_name}")

    st_model = SentenceTransformer(model_name)
    dim = st_model.get_sentence_embedding_dimension()

    def embed(texts: list[str]) -> np.ndarray:
        return st_model.encode(texts, show_progress_bar=False)

    return embed, dim


def _get_openai_embeddings(
    model: str | None = None,
) -> tuple[Callable[[list[str]], np.ndarray], int]:
    """Get OpenAI embeddings."""
    try:
        import openai
    except ImportError:
        raise RuntimeError(
            "openai is required for OpenAI embeddings. "
            "Install with: pip install openai"
        )

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set for OpenAI embeddings")

    model_name = model or "text-embedding-3-large"
    logger.info(f"Using OpenAI embeddings: {model_name}")

    # Dimensions for known models
    dims = {
        "text-embedding-3-large": 3072,
        "text-embedding-3-small": 1536,
        "text-embedding-ada-002": 1536,
    }
    dim = dims.get(model_name, 3072)

    client = openai.OpenAI(api_key=api_key)

    def embed(texts: list[str]) -> np.ndarray:
        response = client.embeddings.create(input=texts, model=model_name)
        return np.array([e.embedding for e in response.data])

    return embed, dim


def _get_gemini_embeddings(
    model: str | None = None,
) -> tuple[Callable[[list[str]], np.ndarray], int]:
    """Get Google Gemini embeddings."""
    try:
        import google.generativeai as genai
    except ImportError:
        raise RuntimeError(
            "google-generativeai is required for Gemini embeddings. "
            "Install with: pip install google-generativeai"
        )

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY or GEMINI_API_KEY not set for Gemini embeddings"
        )

    genai.configure(api_key=api_key)
    model_name = model or "models/text-embedding-004"
    logger.info(f"Using Gemini embeddings: {model_name}")

    dim = 768  # text-embedding-004 dimension

    def embed(texts: list[str]) -> np.ndarray:
        results = []
        for text in texts:
            response = genai.embed_content(model=model_name, content=text)
            results.append(response["embedding"])
        return np.array(results)

    return embed, dim


def _get_voyage_embeddings(
    model: str | None = None,
) -> tuple[Callable[[list[str]], np.ndarray], int]:
    """Get Voyage AI embeddings (excellent for code)."""
    try:
        import voyageai
    except ImportError:
        raise RuntimeError(
            "voyageai is required for Voyage embeddings. "
            "Install with: pip install voyageai"
        )

    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        raise RuntimeError("VOYAGE_API_KEY not set for Voyage embeddings")

    model_name = model or "voyage-code-2"
    logger.info(f"Using Voyage embeddings: {model_name}")

    # voyage-code-2 has 1024 dimensions
    dim = 1024

    client = voyageai.Client(api_key=api_key)

    def embed(texts: list[str]) -> np.ndarray:
        result = client.embed(texts, model=model_name)
        return np.array(result.embeddings)

    return embed, dim


def check_embeddings_available() -> tuple[bool, str]:
    """Check if embedding dependencies are available."""
    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
        return True, ""
    except ImportError:
        return False, (
            "sentence-transformers is required for local semantic search. "
            "Install with: pip install sentence-transformers\n"
            "For better embeddings, also set one of:\n"
            "  - OPENAI_API_KEY (uses text-embedding-3-large)\n"
            "  - VOYAGE_API_KEY (uses voyage-code-2, excellent for code)"
        )
