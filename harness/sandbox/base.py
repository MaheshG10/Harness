from __future__ import annotations

import shutil
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

MAX_OUTPUT_BYTES = 256_000  # hard cap on what we keep from one command


@dataclass
class ExecResult:
    exit_code: int
    output: str  # stdout and stderr, interleaved
    timed_out: bool = False
    duration: float = 0.0
    truncated: bool = False


class PathEscape(ValueError):
    pass


class Sandbox(ABC):
    """A workspace on the host plus somewhere to run commands against it.

    Files live in a host temp dir so file tools are fast and simple.
    Commands run wherever the backend decides (a container, or the host).
    """

    #: how the workspace root looks from inside the sandbox
    workdir: str

    def __init__(self, source: str | Path | None = None):
        self.root = Path(tempfile.mkdtemp(prefix="harness-ws-")).resolve()
        if source is not None:
            shutil.copytree(source, self.root, dirs_exist_ok=True)

    # ---- lifecycle ----

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def exec(self, command: str, timeout: float = 60.0) -> ExecResult: ...

    def stop(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def __enter__(self) -> "Sandbox":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # ---- paths ----

    def resolve(self, path: str) -> Path:
        """Map a model-supplied path to a host path inside the workspace.

        Accepts relative paths and absolute paths under `workdir`.
        Anything that escapes the workspace (.., symlinks out) is refused.
        """
        p = path.strip() or "."
        if p.startswith(self.workdir.rstrip("/") + "/") or p == self.workdir:
            p = str(PurePosixPath(p).relative_to(self.workdir)) or "."
        elif p.startswith("/"):
            raise PathEscape(f"{path} is outside the workspace ({self.workdir})")
        full = (self.root / p).resolve()
        if full != self.root and self.root not in full.parents:
            raise PathEscape(f"{path} is outside the workspace")
        return full

    def rel(self, full: Path) -> str:
        return full.relative_to(self.root).as_posix() or "."


def clip(data: bytes, limit: int = MAX_OUTPUT_BYTES) -> tuple[str, bool]:
    """Keep the head and tail of oversized output; errors usually live at the end."""
    if len(data) <= limit:
        return data.decode("utf-8", "replace"), False
    half = limit // 2
    dropped = len(data) - limit
    text = (
        data[:half].decode("utf-8", "replace")
        + f"\n... [{dropped} bytes cut] ...\n"
        + data[-half:].decode("utf-8", "replace")
    )
    return text, True
