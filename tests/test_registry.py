from __future__ import annotations

from typing import Literal, Optional

import pytest

from harness.llm.types import ToolCall
from harness.tools import SchemaError, ToolRegistry


def make_registry():
    reg = ToolRegistry()

    @reg.tool
    def read_file(ctx, path: str, start: int = 1, end: Optional[int] = None) -> str:
        """Read a file from the workspace.

        Args:
            path: File path relative to the repo root.
            start: First line to read, 1-based.
            end: Last line to read. Reads to the end if omitted.
        """
        lines = ctx[path].splitlines()
        return "\n".join(lines[start - 1 : end])

    @reg.tool
    def grep(pattern: str, mode: Literal["regex", "fixed"] = "fixed", paths: list[str] | None = None):
        """Search files for a pattern.

        Args:
            pattern: What to look for.
            mode: How to read the pattern.
            paths: Limit the search to these paths.
        """
        return {"pattern": pattern, "mode": mode, "paths": paths}

    @reg.tool
    def explode() -> str:
        """Always fails."""
        raise RuntimeError("disk on fire")

    return reg


def spec(reg, name):
    return next(s for s in reg.specs() if s.name == name)


def test_schema_from_hints_and_docstring():
    s = spec(make_registry(), "read_file")
    assert s.description == "Read a file from the workspace."
    props = s.parameters["properties"]
    assert set(props) == {"path", "start", "end"}  # ctx is hidden from the model
    assert props["path"] == {"type": "string", "description": "File path relative to the repo root."}
    assert props["end"]["type"] == "integer"
    assert props["end"]["description"].startswith("Last line to read. Reads to the end")
    assert s.parameters["required"] == ["path"]


def test_literal_and_list_schema():
    props = spec(make_registry(), "grep").parameters["properties"]
    assert props["mode"] == {"type": "string", "enum": ["regex", "fixed"], "description": "How to read the pattern."}
    assert props["paths"]["type"] == "array" and props["paths"]["items"] == {"type": "string"}


def test_call_injects_ctx_and_coerces_int():
    reg = make_registry()
    out = reg.call(ToolCall("read_file", {"path": "a.py", "start": 2.0}), ctx={"a.py": "x\ny\nz"})
    assert not out.is_error and out.content == "y\nz"


def test_dict_output_is_json():
    out = make_registry().call(ToolCall("grep", {"pattern": "foo"}))
    assert out.content == '{"pattern": "foo", "mode": "fixed", "paths": null}'


@pytest.mark.parametrize(
    "call,needle",
    [
        (ToolCall("nope", {}), "unknown tool 'nope'"),
        (ToolCall("read_file", {}), "missing required argument(s) ['path']"),
        (ToolCall("read_file", {"path": "a", "colour": 1}), "unknown argument(s) ['colour']"),
        (ToolCall("read_file", {"path": 3}), "bad argument 'path'"),
        (ToolCall("grep", {"pattern": "x", "mode": "fuzzy"}), "must be one of"),
        (ToolCall("explode", {}), "RuntimeError: disk on fire"),
    ],
)
def test_failures_become_error_results(call, needle):
    out = make_registry().call(call, ctx={})
    assert out.is_error and needle in out.content


def test_call_id_is_kept():
    out = make_registry().call(ToolCall("grep", {"pattern": "x"}, id="call-1"))
    assert out.id == "call-1"


def test_rejects_tools_without_docstring_or_hints():
    reg = ToolRegistry()
    with pytest.raises(SchemaError, match="docstring"):

        @reg.tool
        def nodoc(x: str):
            return x

    with pytest.raises(SchemaError, match="type hint"):

        @reg.tool
        def nohint(x):
            """Has a doc."""


def test_duplicate_names_rejected():
    reg = ToolRegistry()

    @reg.tool
    def a() -> str:
        """A."""

    with pytest.raises(SchemaError, match="duplicate"):
        reg.tool(a)
