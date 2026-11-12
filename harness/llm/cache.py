"""Record/replay cache for model calls.

Every response is stored under a hash of (model, full request). That gives us:
  - no call is ever paid for twice, which matters on the free tier
  - resumed runs pick up exactly where they stopped
  - CI replays recorded runs with zero API calls ("replay" mode)

The seed is part of the request, so different seeds never share an entry.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Literal

from .errors import CacheMiss

Mode = Literal["off", "read_write", "record", "replay"]


def request_key(model: str, request: dict) -> str:
    canonical = json.dumps({"model": model, "request": request}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


class ResponseCache:
    def __init__(self, root: str | Path, mode: Mode = "read_write"):
        self.root = Path(root)
        self.mode = mode

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, model: str, request: dict) -> dict | None:
        if self.mode in ("off", "record"):
            return None
        path = self._path(request_key(model, request))
        if path.exists():
            return json.loads(path.read_text())["response"]
        if self.mode == "replay":
            raise CacheMiss(f"no recorded response for this request in {self.root}")
        return None

    def put(self, model: str, request: dict, response: dict) -> None:
        if self.mode in ("off", "replay"):
            return
        key = request_key(model, request)
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"model": model, "saved_at": time.time(), "response": response}
        # write then rename, so parallel workers never read a half-written file
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(entry))
        os.replace(tmp, path)
