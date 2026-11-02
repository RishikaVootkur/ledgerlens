"""Embeddings from `nomic-embed-text` running in Ollama.

nomic-embed-text expects task prefixes: `search_document:` for passages and
`search_query:` for queries. Query embeddings are cached on disk so repeated eval
runs do not re-embed the same questions.
"""

from __future__ import annotations

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from ledgerlens.config import get_settings
from ledgerlens.llm.cache import ResponseCache, cache_key

EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768
MAX_CHARS = 6000  # ~1.5k tokens; longer passages are truncated to protect the context window


def _transient(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code >= 500


@retry(
    retry=retry_if_exception(_transient),
    wait=wait_exponential_jitter(multiplier=1, max=30),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _embed_raw(texts: list[str]) -> list[list[float]]:
    base = get_settings().ollama_base_url
    resp = httpx.post(
        f"{base}/api/embed",
        json={"model": EMBED_MODEL, "input": texts, "truncate": True},
        timeout=600,
    )
    resp.raise_for_status()
    vectors: list[list[float]] = resp.json()["embeddings"]
    return vectors


def embed_documents(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = [f"search_document: {t[:MAX_CHARS]}" for t in texts[i : i + batch_size]]
        out.extend(_embed_raw(batch))
    return out


def embed_query(text: str) -> list[float]:
    s = get_settings()
    cache = ResponseCache(s.cache_dir, replay_dirs=[s.eval_dir / "cache"])
    key = cache_key("embed", {"model": EMBED_MODEL, "query": text})
    if (hit := cache.get("embed", key)) is not None:
        vector: list[float] = hit["vector"]
        return vector
    vector = _embed_raw([f"search_query: {text}"])[0]
    cache.put("embed", key, {"model": EMBED_MODEL, "query": text, "vector": vector})
    return vector
