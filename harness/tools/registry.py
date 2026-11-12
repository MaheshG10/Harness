"""Tool registry.

Tools are plain Python functions. The JSON schema the model sees is built
from the type hints and the docstring, so the schema can never drift from
the code. Calls are validated before they run, and every failure comes back
to the model as an error message instead of crashing the agent loop. Models
recover from a clear error far more often than from a dead run.
"""

from __future__ import annotations

import inspect
import json
import re
import types
import typing
from dataclasses import dataclass
from typing import Any, Callable, Literal, get_args, get_origin

from ..llm.types import ToolCall, ToolResult, ToolSpec

CTX_PARAM = "ctx"

_PRIMITIVES = {str: "string", int: "integer", float: "number", bool: "boolean"}


class SchemaError(TypeError):
    pass


def _is_optional(tp: Any) -> tuple[bool, Any]:
    origin = get_origin(tp)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) == 1 and len(get_args(tp)) == 2:
            return True, args[0]
    return False, tp


def type_schema(tp: Any) -> dict:
    if tp in _PRIMITIVES:
        return {"type": _PRIMITIVES[tp]}
    origin = get_origin(tp)
    if origin is Literal:
        values = get_args(tp)
        if not all(isinstance(v, str) for v in values):
            raise SchemaError("only string Literals are supported")
        return {"type": "string", "enum": list(values)}
    if origin is list:
        (item,) = get_args(tp) or (str,)
        return {"type": "array", "items": type_schema(item)}
    raise SchemaError(f"unsupported parameter type: {tp!r}")


def parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """Google-style docstring -> (summary, {arg: description})."""
    if not doc:
        return "", {}
    doc = inspect.cleandoc(doc)
    parts = re.split(r"^\s*Args:\s*$", doc, maxsplit=1, flags=re.M)
    summary = " ".join(parts[0].split())
    args: dict[str, str] = {}
    if len(parts) == 2:
        current = None
        for line in parts[1].splitlines():
            m = re.match(r"^\s*(\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$", line)
            if m:
                current = m.group(1)
                args[current] = m.group(2).strip()
            elif current and line.strip():
                args[current] += " " + line.strip()
    return summary, args


@dataclass
class Tool:
    name: str
    fn: Callable[..., Any]
    spec: ToolSpec
    wants_ctx: bool
    params: dict[str, tuple[Any, bool]]  # name -> (type, required)


def build_tool(fn: Callable[..., Any], name: str | None = None) -> Tool:
    sig = inspect.signature(fn)
    hints = typing.get_type_hints(fn)
    summary, arg_docs = parse_docstring(fn.__doc__)
    if not summary:
        raise SchemaError(f"tool {fn.__name__} needs a docstring; the model reads it")

    properties: dict[str, dict] = {}
    required: list[str] = []
    params: dict[str, tuple[Any, bool]] = {}
    wants_ctx = False
    for pname, p in sig.parameters.items():
        if pname == CTX_PARAM:
            wants_ctx = True
            continue
        if pname not in hints:
            raise SchemaError(f"{fn.__name__}.{pname} needs a type hint")
        optional, tp = _is_optional(hints[pname])
        schema = type_schema(tp)
        if pname in arg_docs:
            schema["description"] = arg_docs[pname]
        properties[pname] = schema
        is_required = p.default is inspect.Parameter.empty and not optional
        if is_required:
            required.append(pname)
        params[pname] = (tp, is_required)

    parameters: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        parameters["required"] = required
    tname = name or fn.__name__
    return Tool(tname, fn, ToolSpec(tname, summary, parameters), wants_ctx, params)


def _coerce(value: Any, tp: Any) -> Any:
    """Light coercion for what models actually send (e.g. 5.0 for an int)."""
    if tp is int and isinstance(value, float) and value.is_integer():
        return int(value)
    if tp is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if tp in _PRIMITIVES:
        if tp is not bool and isinstance(value, bool):
            raise TypeError(f"expected {_PRIMITIVES[tp]}, got boolean")
        if not isinstance(value, tp):
            raise TypeError(f"expected {_PRIMITIVES[tp]}, got {type(value).__name__}")
        return value
    origin = get_origin(tp)
    if origin is Literal:
        if value not in get_args(tp):
            raise TypeError(f"must be one of {list(get_args(tp))}")
        return value
    if origin is list:
        if not isinstance(value, list):
            raise TypeError("expected an array")
        (item,) = get_args(tp) or (str,)
        return [_coerce(v, item) for v in value]
    return value


def _stringify(out: Any) -> str:
    if out is None:
        return "ok"
    if isinstance(out, str):
        return out
    return json.dumps(out, default=str)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def tool(self, fn: Callable[..., Any] | None = None, *, name: str | None = None):
        def register(f: Callable[..., Any]) -> Callable[..., Any]:
            t = build_tool(f, name)
            if t.name in self._tools:
                raise SchemaError(f"duplicate tool name: {t.name}")
            self._tools[t.name] = t
            return f

        return register(fn) if fn else register

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self) -> list[ToolSpec]:
        return [t.spec for t in self._tools.values()]

    def call(self, call: ToolCall, ctx: Any = None) -> ToolResult:
        def fail(msg: str) -> ToolResult:
            return ToolResult(call.name, msg, is_error=True, id=call.id)

        tool = self._tools.get(call.name)
        if tool is None:
            return fail(f"unknown tool '{call.name}'. available: {', '.join(self.names)}")

        args = dict(call.args or {})
        unknown = set(args) - set(tool.params)
        if unknown:
            return fail(f"unknown argument(s) {sorted(unknown)} for {tool.name}")
        missing = [p for p, (_, req) in tool.params.items() if req and p not in args]
        if missing:
            return fail(f"missing required argument(s) {missing} for {tool.name}")
        try:
            for pname, value in list(args.items()):
                if value is None and not tool.params[pname][1]:
                    del args[pname]
                    continue
                args[pname] = _coerce(value, tool.params[pname][0])
        except TypeError as e:
            return fail(f"bad argument '{pname}': {e}")

        if tool.wants_ctx:
            args[CTX_PARAM] = ctx
        try:
            out = tool.fn(**args)
        except Exception as e:  # tools fail; the model should hear about it
            return fail(f"{type(e).__name__}: {e}")
        return ToolResult(call.name, _stringify(out), id=call.id)
