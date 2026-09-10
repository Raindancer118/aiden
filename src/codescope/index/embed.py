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

Two properties matter for a local-first index and are enforced here:

*Bounded memory.* :meth:`Embedder.embed_batched` yields vectors in small,
length-homogeneous batches. ONNX pads every batch to its longest member and
attention cost grows quadratically with that length, so one long body in a
256-item batch used to inflate peak RSS into the tens of gigabytes. Sorting by
length before batching removes almost all of that padding waste, and yielding
per batch lets callers persist as they go instead of buffering every vector.

*Reused models.* Loading a code embedding model costs seconds and hundreds of
megabytes, so :func:`get_embedder` / :func:`embedder_from_id` memoize instances
per configuration. Without this, every single search reloaded the model.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import threading
from abc import ABC, abstractmethod
from collections.abc import Iterator

log = logging.getLogger(__name__)

DEFAULT_FASTEMBED_MODEL = "jinaai/jina-embeddings-v2-base-code"
DEFAULT_GEMINI_MODEL = "gemini-embedding-001"

#: Upper bound on documents per forward pass. The *effective* batch is chosen
#: per batch from the character budget below, so short symbols still go out in
#: large batches while long ones do not blow up memory.
DEFAULT_BATCH_SIZE = 128

#: Budget for ``count * longest_length ** 2`` per forward pass, in char^2.
#:
#: Transformer attention allocates a ``length x length`` matrix per item, and
#: ONNX pads every item in a batch to the longest one, so a batch's peak
#: memory tracks ``count * longest^2`` -- not the item count, and not
#: ``count * longest``. Budgeting the actual quantity is what keeps peak
#: memory flat across a run whose batches get progressively longer, while
#: still letting hundreds of short symbols share one pass.
#:
#: At the default body budget (~1900 chars) this allows ~5 of the longest
#: symbols per pass, or ~128 short ones.
DEFAULT_BATCH_COST = 20_000_000

#: Hard cap on the characters handed to the model. Symbol bodies are already
#: truncated at parse time (``parser._MAX_BODY_CHARS``); this is the backstop
#: for callers that pass raw snippets (``find_similar_code``, diff blocks).
DEFAULT_MAX_CHARS = 4000

_TOKEN_RE = re.compile(r"[^\W\d]\w*|_\w*")

#: Characters of the body that go into the embedding text. Bodies are stored
#: in full for BM25/trigram; the model only needs the head of the body, where
#: a symbol's intent lives, plus the context header built around it.
DEFAULT_EMBED_BODY_CHARS = 1800

_ENV_MODEL = "CODESCOPE_EMBED_MODEL"
_ENV_BATCH = "CODESCOPE_EMBED_BATCH"
_ENV_BATCH_COST = "CODESCOPE_EMBED_BATCH_COST"
_ENV_THREADS = "CODESCOPE_EMBED_THREADS"

_CACHE: dict[str, "Embedder"] = {}
_CACHE_LOCK = threading.Lock()


def _env_int(name: str, default: int | None) -> int | None:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        log.warning("Ignoring invalid %s=%r (expected an integer).", name, raw)
        return default
    return value if value > 0 else default


def build_embed_text(name: str, kind: str, signature: str, body: str, body_chars: int = DEFAULT_EMBED_BODY_CHARS) -> str:
    """Build the text that represents a symbol in vector space.

    Embedding a bare body throws away what a symbol *is*: a query like
    "validate an auth token" should match a body that never spells those words
    but is called ``validate_token``. Prefixing the kind, name and signature
    puts those tokens into the same vector as the implementation.

    The file path is deliberately *not* part of this text. Path is already a
    weighted BM25 field, so location is covered lexically, whereas putting it
    in the vector makes two identical functions in different files look
    different -- which is precisely the case clone detection must catch.
    """
    head = f"{kind} {name}\n{signature}".strip()
    body = body.strip()
    if body_chars > 0:
        body = body[:body_chars]
    return f"{head}\n\n{body}" if body else head


def build_snippet_embed_text(snippet: str, body_chars: int = DEFAULT_EMBED_BODY_CHARS) -> str:
    """Shape a raw code snippet like an indexed symbol before embedding it.

    Indexed symbols carry a ``kind name`` / signature header, so embedding a
    bare snippet compares two differently-shaped texts and systematically
    understates the similarity of genuine duplicates. A snippet has no known
    name or kind, but its first line is its signature, so reusing it as the
    header restores the shape.
    """
    first_line = next((ln.strip() for ln in snippet.splitlines() if ln.strip()), "")
    return build_embed_text(name="", kind="", signature=first_line, body=snippet, body_chars=body_chars)


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]


class Embedder(ABC):
    """Abstract embedding backend."""

    id: str
    dim: int
    #: Hard cap on documents per forward pass.
    batch_size: int = DEFAULT_BATCH_SIZE
    #: Cap on ``count * longest_length ** 2`` per forward pass.
    batch_cost: int = DEFAULT_BATCH_COST
    #: Characters per document handed to the backend.
    max_chars: int = DEFAULT_MAX_CHARS

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents (indexed code chunks)."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query. Backends may specialize this."""
        return self.embed_documents([text])[0]

    def _truncate(self, texts: list[str]) -> list[str]:
        limit = self.max_chars
        if limit <= 0:
            return texts
        return [t if len(t) <= limit else t[:limit] for t in texts]

    def embed_batched(self, texts: list[str]) -> Iterator[list[tuple[int, list[float]]]]:
        """Embed ``texts`` in bounded, length-homogeneous batches, longest first.

        Yields ``[(original_index, vector), ...]`` per batch so callers can
        persist incrementally: an interrupted run only loses the batch in
        flight. Grouping texts of similar length removes the padding waste
        that dominates ONNX inference cost.

        The batches run from longest to shortest, which is what actually keeps
        peak memory flat. ONNX Runtime's allocator grows its arena to fit each
        new tensor shape and never returns it, so feeding it steadily longer
        batches makes memory climb for the entire run -- measured at 7 GB and
        still rising on this repository. Starting with the largest batch
        allocates the high-water mark once; every later, smaller batch reuses
        it. Measured on the same workload: 2.9 GB on the first batch, then
        flat to the end.
        """
        if not texts:
            return
        order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
        max_items = max(1, self.batch_size)
        budget = max(1, self.batch_cost)

        current: list[int] = []
        longest = 0
        for index in order:
            length = max(1, len(texts[index]))
            # Cost of adding this item: the batch is padded to its longest
            # member and attention is quadratic in that length, so the
            # projected cost is count * longest^2.
            projected = max(longest, length) ** 2 * (len(current) + 1)
            if current and (len(current) >= max_items or projected > budget):
                yield self._embed_indices(texts, current)
                current, longest = [], 0
            current.append(index)
            longest = max(longest, length)
        if current:
            yield self._embed_indices(texts, current)

    def _embed_indices(self, texts: list[str], indices: list[int]) -> list[tuple[int, list[float]]]:
        vectors = self.embed_documents([texts[i] for i in indices])
        return list(zip(indices, vectors, strict=True))


class HashingEmbedder(Embedder):
    """Deterministic, dependency-free hashing embedder (lexical fallback)."""

    def __init__(self, dim: int = 256):
        if dim <= 0:
            raise ValueError("Embedding dimension must be positive.")
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
    """Local ONNX code-aware embeddings via fastembed.

    The model is loaded once per process (see :func:`get_embedder`). Batch size
    and thread count are configurable because they, not the model choice, set
    the memory ceiling of a full reindex.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_FASTEMBED_MODEL,
        batch_size: int | None = None,
        threads: int | None = None,
        max_chars: int | None = None,
    ):
        from fastembed import TextEmbedding  # lazy: only when selected

        self.batch_size = batch_size or _env_int(_ENV_BATCH, DEFAULT_BATCH_SIZE) or DEFAULT_BATCH_SIZE
        self.batch_cost = _env_int(_ENV_BATCH_COST, DEFAULT_BATCH_COST) or DEFAULT_BATCH_COST
        self.max_chars = max_chars if max_chars is not None else DEFAULT_MAX_CHARS
        threads = threads if threads is not None else _env_int(_ENV_THREADS, None)
        self._model = TextEmbedding(model_name=model_name, threads=threads)
        self.id = f"fastembed:{model_name}"
        # Probe dimensionality once.
        probe = next(iter(self._model.embed(["dimension probe"])))
        self.dim = len(probe)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.embed(self._truncate(texts), batch_size=self.batch_size)
        return [_l2_normalize([float(x) for x in v]) for v in vectors]


class GeminiEmbedder(Embedder):
    """Cloud embeddings via Google Gemini (optional, requires API key)."""

    def __init__(self, model_name: str = DEFAULT_GEMINI_MODEL, dim: int = 768, api_key: str | None = None):
        if dim <= 0:
            raise ValueError("Embedding dimension must be positive.")
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
        for text in self._truncate(texts):  # Gemini embeds one input per request
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


def _build_embedder(name: str, **kwargs: object) -> Embedder:
    if name in ("fastembed", "auto"):
        try:
            kwargs.setdefault("model_name", os.environ.get(_ENV_MODEL) or DEFAULT_FASTEMBED_MODEL)
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


def _cached(cache_key: str, factory) -> Embedder:  # type: ignore[no-untyped-def]
    """Return a memoized embedder, building it outside the lock's critical path.

    Model construction is slow (seconds) and must not be serialized behind a
    lock held by another caller building a *different* model, so we build
    optimistically and keep whichever instance landed in the cache first.
    """
    with _CACHE_LOCK:
        hit = _CACHE.get(cache_key)
    if hit is not None:
        return hit
    built = factory()
    with _CACHE_LOCK:
        return _CACHE.setdefault(cache_key, built)


def get_embedder(name: str = "auto", **kwargs: object) -> Embedder:
    """Resolve an embedder by name, reusing an already-loaded instance.

    ``auto`` prefers fastembed (local, code-aware) and falls back to the
    hashing embedder if fastembed is not installed or fails to initialize.
    The model name can be overridden with ``CODESCOPE_EMBED_MODEL``.
    """
    name = (name or "auto").lower()
    key = f"{name}|" + "|".join(f"{k}={v!r}" for k, v in sorted(kwargs.items()))
    return _cached(key, lambda: _build_embedder(name, **kwargs))


def embedder_from_id(embedder_id: str) -> Embedder:
    """Reconstruct the embedder that produced an index, from its stored id.

    Query embeddings must use the same backend/space as the indexed vectors.
    Instances are cached per id: a search must never pay a model load.
    """

    def build() -> Embedder:
        if embedder_id.startswith("hashing-"):
            return HashingEmbedder(dim=int(embedder_id.split("-", 1)[1]))
        if embedder_id.startswith("fastembed:"):
            return FastEmbedEmbedder(model_name=embedder_id.split(":", 1)[1])
        if embedder_id.startswith("gemini:"):
            _, model_name, dim = embedder_id.split(":", 2)
            return GeminiEmbedder(model_name=model_name, dim=int(dim))
        raise ValueError(f"Unknown embedder id: {embedder_id!r}")

    return _cached(f"id|{embedder_id}", build)


def clear_embedder_cache() -> None:
    """Drop all cached embedder instances (frees the loaded models)."""
    with _CACHE_LOCK:
        _CACHE.clear()
