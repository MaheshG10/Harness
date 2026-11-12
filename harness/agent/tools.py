"""The coding tools the agent gets.

Kept deliberately small. Each tool returns plain text written for the model:
short, with line numbers where they help, and with a hint when something
goes wrong so the next step can fix it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..sandbox import Sandbox
from ..tools import ToolRegistry

SHELL_OUTPUT_LIMIT = 30_000  # chars; a safety cap, context strategies may cut further
READ_LIMIT_LINES = 400
SEARCH_LIMIT = 100
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".venv", "node_modules"}


@dataclass
class ToolContext:
    sandbox: Sandbox
    finished: bool = False
    summary: str | None = None
    files_changed: set[str] = field(default_factory=set)


def _numbered(lines: list[str], start: int) -> str:
    width = len(str(start + len(lines)))
    return "\n".join(f"{i:>{width}}| {line}" for i, line in enumerate(lines, start))


def _cap(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + f"\n... [{len(text) - limit} chars cut] ...\n" + text[-half:]


def build_registry() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.tool
    def list_files(ctx: ToolContext, path: str = ".", depth: int = 2) -> str:
        """List files and folders in the workspace.

        Args:
            path: Folder to list, relative to the workspace root.
            depth: How many folder levels to show (1-5).
        """
        base = ctx.sandbox.resolve(path)
        if not base.is_dir():
            raise NotADirectoryError(f"{path} is not a folder")
        depth = max(1, min(depth, 5))
        out: list[str] = []

        def walk(d, level):
            for entry in sorted(d.iterdir(), key=lambda p: (p.is_file(), p.name)):
                if entry.name in SKIP_DIRS:
                    continue
                out.append("  " * level + entry.name + ("/" if entry.is_dir() else ""))
                if len(out) >= 500:
                    return
                if entry.is_dir() and level + 1 < depth:
                    walk(entry, level + 1)

        walk(base, 0)
        if not out:
            return "(empty)"
        return "\n".join(out) + ("\n... (listing cut at 500 entries)" if len(out) >= 500 else "")

    @reg.tool
    def read_file(ctx: ToolContext, path: str, start: int = 1, end: int | None = None) -> str:
        """Read a text file. Lines come back numbered so you can refer to them.

        Args:
            path: File path relative to the workspace root.
            start: First line to show, 1-based.
            end: Last line to show. Defaults to start + 399.
        """
        f = ctx.sandbox.resolve(path)
        if not f.is_file():
            raise FileNotFoundError(f"{path} does not exist. Use list_files to see what is there.")
        lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(1, start)
        capped = end is None
        end = min(len(lines), end or start + READ_LIMIT_LINES - 1)
        if start > len(lines):
            return f"(file has only {len(lines)} lines)"
        body = _numbered(lines[start - 1 : end], start)
        if capped and end < len(lines):
            body += f"\n... ({len(lines) - end} more lines; read again with start={end + 1})"
        return body

    @reg.tool
    def write_file(ctx: ToolContext, path: str, content: str) -> str:
        """Create a file or overwrite it completely. For small changes prefer edit_file.

        Args:
            path: File path relative to the workspace root.
            content: The full new file content.
        """
        f = ctx.sandbox.resolve(path)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(content, encoding="utf-8")
        ctx.files_changed.add(ctx.sandbox.rel(f))
        return f"wrote {ctx.sandbox.rel(f)} ({len(content.splitlines())} lines)"

    @reg.tool
    def edit_file(ctx: ToolContext, path: str, old: str, new: str, replace_all: bool = False) -> str:
        """Replace an exact piece of text in a file.

        `old` must match the file exactly, including spaces and indentation, and
        must be unique unless replace_all is true. Include a few surrounding
        lines to make it unique.

        Args:
            path: File path relative to the workspace root.
            old: The exact text to replace.
            new: The replacement text.
            replace_all: Replace every occurrence instead of exactly one.
        """
        f = ctx.sandbox.resolve(path)
        if not f.is_file():
            raise FileNotFoundError(f"{path} does not exist")
        if old == "":
            raise ValueError("old is empty; use write_file to create a file")
        text = f.read_text(encoding="utf-8")
        count = text.count(old)
        if count == 0:
            hint = ""
            first = old.strip().splitlines()[0].strip() if old.strip() else ""
            if first and first in text:
                line_no = text[: text.index(first)].count("\n") + 1
                hint = f" The first line appears near line {line_no}; check whitespace and the lines around it."
            raise ValueError(f"old text not found in {path}.{hint} Read the file again before editing.")
        if count > 1 and not replace_all:
            raise ValueError(f"old text appears {count} times in {path}. Add surrounding lines to make it unique.")
        text = text.replace(old, new) if replace_all else text.replace(old, new, 1)
        f.write_text(text, encoding="utf-8")
        ctx.files_changed.add(ctx.sandbox.rel(f))
        # show the edited region so the model can confirm it
        pos = text.find(new) if new else -1
        if pos < 0:
            return f"edited {path} ({count if replace_all else 1} replacement)"
        line = text[:pos].count("\n") + 1
        lines = text.splitlines()
        lo, hi = max(1, line - 3), min(len(lines), line + new.count("\n") + 3)
        return f"edited {path}:\n" + _numbered(lines[lo - 1 : hi], lo)

    @reg.tool
    def run_shell(ctx: ToolContext, command: str, timeout: int = 60) -> str:
        """Run a shell command in the workspace (no network). Use it to run tests or scripts.

        Args:
            command: The command, e.g. "python -m pytest -q".
            timeout: Seconds before the command is killed (max 300).
        """
        r = ctx.sandbox.exec(command, timeout=max(1, min(timeout, 300)))
        status = "timed out" if r.timed_out else f"exit code {r.exit_code}"
        return f"[{status}, {r.duration:.1f}s]\n" + _cap(r.output.rstrip(), SHELL_OUTPUT_LIMIT)

    @reg.tool
    def search_code(ctx: ToolContext, pattern: str, path: str = ".", regex: bool = False) -> str:
        """Search file contents. Returns file:line: text for each match.

        Args:
            pattern: Text to find.
            path: File or folder to search, relative to the workspace root.
            regex: Treat the pattern as a Python regular expression.
        """
        base = ctx.sandbox.resolve(path)
        rx = re.compile(pattern if regex else re.escape(pattern))
        files = [base] if base.is_file() else sorted(
            p for p in base.rglob("*") if p.is_file() and not SKIP_DIRS & set(p.relative_to(base).parts)
        )
        hits: list[str] = []
        for f in files:
            try:
                text = f.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for n, line in enumerate(text.splitlines(), 1):
                if rx.search(line):
                    hits.append(f"{ctx.sandbox.rel(f)}:{n}: {line.strip()[:200]}")
                    if len(hits) >= SEARCH_LIMIT:
                        return "\n".join(hits) + f"\n... (stopped at {SEARCH_LIMIT} matches)"
        return "\n".join(hits) if hits else "no matches"

    @reg.tool
    def finish(ctx: ToolContext, summary: str) -> str:
        """Call this when the task is done (or cannot be done). Ends the run.

        Args:
            summary: One or two sentences on what you changed and why.
        """
        ctx.finished = True
        ctx.summary = summary
        return "done"

    return reg
