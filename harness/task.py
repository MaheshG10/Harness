from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Task:
    id: str
    prompt: str
    repo: Path
    test_command: str = "python -m pytest -q"
    test_timeout: int = 120


def load_task(path: str | Path) -> Task:
    """A task is a folder with task.yaml and a repo/ folder."""
    d = Path(path)
    spec = yaml.safe_load((d / "task.yaml").read_text(encoding="utf-8"))
    repo = d / "repo"
    if not repo.is_dir():
        raise FileNotFoundError(f"{repo} is missing")
    return Task(
        id=spec.get("id", d.name),
        prompt=spec["prompt"].strip(),
        repo=repo,
        test_command=spec.get("test_command", "python -m pytest -q"),
        test_timeout=int(spec.get("test_timeout", 120)),
    )
