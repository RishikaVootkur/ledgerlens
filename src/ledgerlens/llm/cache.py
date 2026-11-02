"""Content-addressed response cache stored as one JSON file per request.

JSON files (not a database) so the slice CI replays can be committed and reviewed in
diffs. Keys hash the full request, so any change to model, prompt, tools or sampling
parameters is a miss. Reads check the writable cache first, then any read-only replay
directories (for example the committed `eval/cache`).
"""

from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

CACHE_VERSION = 1


class CacheMissError(LookupError):
    """Raised in replay mode when a request was never recorded."""


def cache_key(namespace: str, payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        {"v": CACHE_VERSION, "ns": namespace, "payload": payload},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


class ResponseCache:
    def __init__(self, root: Path, replay_dirs: list[Path] | None = None) -> None:
        self.root = root
        self.replay_dirs = replay_dirs or []

    def _path(self, base: Path, namespace: str, key: str) -> Path:
        return base / namespace / key[:2] / f"{key}.json"

    def get(self, namespace: str, key: str) -> dict[str, Any] | None:
        for base in (self.root, *self.replay_dirs):
            path = self._path(base, namespace, key)
            if path.exists():
                data: dict[str, Any] = json.loads(path.read_text())
                return data
        return None

    def put(self, namespace: str, key: str, record: dict[str, Any]) -> Path:
        path = self._path(self.root, namespace, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Unique temp name per write: concurrent writers (threads or processes) never collide.
        tmp = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        tmp.write_text(
            json.dumps(record, indent=1, sort_keys=True, ensure_ascii=False, default=str)
        )
        tmp.replace(path)  # atomic, so a crash never leaves a half-written entry
        return path

    def path_for(self, namespace: str, key: str) -> Path | None:
        for base in (self.root, *self.replay_dirs):
            path = self._path(base, namespace, key)
            if path.exists():
                return path
        return None
