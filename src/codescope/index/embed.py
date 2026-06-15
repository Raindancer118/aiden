"""Embedding backends for semantic code search.

Local-first by design:

  * ``FastEmbedEmbedder`` (default when available) - code-aware ONNX model,
    no API key, runs offline. Requires the ``embeddings`` extra.
  * ``GeminiEmbedder`` - optional cloud backend (``google-genai``), highest
    quality, uses the ``CODE_RETRIEVAL_QUERY`` task type. Requires an API key.
  * ``HashingEmbedder`` - dependency-free deterministic fallback used when no
    real backend is available (and in tests). Lexical, not semantic, but keeps
    the vector path functional everywhere.

All embedders L2-normalize their output so cosine similarity == dot product and
sqlite-vec's L2 distance is monotonic with cosine similarity.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
from abc import ABC, abstractmethod

log = logging.getLogger(__name__)

DEFAULT_FASTEMBED_MODEL = "jinaai/jina-embeddings-v2-base-code"
DEFAULT_GEMINI_MODEL = "gemini-embedding-001"
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]


class Embedder(ABC):
    """Abstract embedding backend."""

    id: str
    dim: int

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents (indexed code chunks)."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query. Backends may specialize this."""
        return self.embed_documents([text])[0]


class HashingEmbedder(Embedder):
    """Deterministic, dependency-free hashing embedder (lexical fallback)."""

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.id = f"hashing-{dim}"

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in _TOKEN_RE.findall(text.lower()):
            h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "little")
            idx = h % self.dim
            sign = 1.0 if (h >> 63) & 1 else -1.0
            vec[idx] += sign
        return _l2_normalize(vec)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]


class FastEmbedEmbedder(Embedder):
    """Local ONNX code-aware embeddings via fastembed."""

    def __init__(self, model_name: str = DEFAULT_FASTEMBED_MODEL):
        from fastembed import TextEmbedding  # lazy: only when selected

        self._model = TextEmbedding(model_name=model_name)
        self.id = f"fastembed:{model_name}"
        # Probe dimensionality once.
        probe = next(iter(self._model.embed(["dimension probe"])))
        self.dim = len(probe)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [_l2_normalize([float(x) for x in v]) for v in self._model.embed(texts)]


class GeminiEmbedder(Embedder):
    """Cloud embeddings via Google Gemini (optional, requires API key)."""

    def __init__(self, model_name: str = DEFAULT_GEMINI_MODEL, dim: int = 768, api_key: str | None = None):
        from google import genai  # lazy: only when selected

        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("GeminiEmbedder requires GEMINI_API_KEY (or GOOGLE_API_KEY).")
        self._genai = genai
        self._client = genai.Client(api_key=key)
        self._model_name = model_name
        self.dim = dim
        self.id = f"gemini:{model_name}:{dim}"

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        from google.genai import types

        out: list[list[float]] = []
        for text in texts:  # Gemini embeds one input per request
            resp = self._client.models.embed_content(
                model=self._model_name,
                contents=text,
                config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=self.dim),
            )
            out.append(_l2_normalize(list(resp.embeddings[0].values)))
        return out

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "RETRIEVAL_DOCUMENT") if texts else []

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "CODE_RETRIEVAL_QUERY")[0]


def get_embedder(name: str = "auto", **kwargs: object) -> Embedder:
    """Resolve an embedder by name.

    ``auto`` prefers fastembed (local, code-aware) and falls back to the
    hashing embedder if fastembed is not installed or fails to initialize.
    """
    name = (name or "auto").lower()
    if name in ("fastembed", "auto"):
        try:
            return FastEmbedEmbedder(**kwargs)  # type: ignore[arg-type]
        except Exception as e:
            if name == "fastembed":
                raise
            log.info("fastembed unavailable (%s); falling back to hashing embedder. Install the 'embeddings' extra for semantic search.", e)
            return HashingEmbedder()
    if name == "gemini":
        return GeminiEmbedder(**kwargs)  # type: ignore[arg-type]
    if name == "hashing":
        return HashingEmbedder(**kwargs)  # type: ignore[arg-type]
    raise ValueError(f"Unknown embedder: {name!r}")


def embedder_from_id(embedder_id: str) -> Embedder:
    """Reconstruct the embedder that produced an index, from its stored id.

    Query embeddings must use the same backend/space as the indexed vectors.
    """
    if embedder_id.startswith("hashing-"):
        return HashingEmbedder(dim=int(embedder_id.split("-", 1)[1]))
    if embedder_id.startswith("fastembed:"):
        return FastEmbedEmbedder(model_name=embedder_id.split(":", 1)[1])
    if embedder_id.startswith("gemini:"):
        _, model_name, dim = embedder_id.split(":", 2)
        return GeminiEmbedder(model_name=model_name, dim=int(dim))
    raise ValueError(f"Unknown embedder id: {embedder_id!r}")
