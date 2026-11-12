from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_dotenv(path: str | Path = ".env") -> None:
    """Tiny .env reader. Real environment variables always win."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip('"').strip("'")
        if value:
            os.environ.setdefault(key.strip(), value)


def _int(name: str) -> int | None:
    v = os.environ.get(name, "").strip()
    return int(v) if v else None


@dataclass
class Settings:
    api_key: str | None
    model: str
    rpm: int | None
    tpm: int | None
    rpd: int | None
    home: Path

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            api_key=os.environ.get("GEMINI_API_KEY") or None,
            model=os.environ.get("HARNESS_MODEL", "gemini-flash-latest"),
            rpm=_int("HARNESS_RPM"),
            tpm=_int("HARNESS_TPM"),
            rpd=_int("HARNESS_RPD"),
            home=Path(os.environ.get("HARNESS_HOME", ".harness")),
        )

    def client(self, model: str | None = None, cache_mode: str = "read_write"):
        from .llm import GeminiClient, RateLimiter, ResponseCache

        model = model or self.model
        limiter = RateLimiter(
            rpm=self.rpm,
            tpm=self.tpm,
            rpd=self.rpd,
            state_path=self.home / "quota" / f"{model}.json",
        )
        cache = ResponseCache(self.home / "cache", mode=cache_mode)  # type: ignore[arg-type]
        return GeminiClient(self.api_key, model, limiter=limiter, cache=cache)
