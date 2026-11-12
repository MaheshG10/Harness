"""Gemini client over the plain REST API.

No SDK on purpose: the wire format is small, and owning it means the retry,
rate limit and cache behaviour is explicit and testable.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Callable

import requests

from .cache import ResponseCache
from .errors import QuotaExhausted, RequestError, TransientError
from .ratelimit import RateLimiter
from .retry import RetryPolicy, call_with_retry
from .types import (
    GenerateConfig,
    LLMResponse,
    Message,
    ToolCall,
    ToolSpec,
    Usage,
)
from .usage import UsageMeter

API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
RETRYABLE = {408, 429, 500, 502, 503, 504}


@dataclass
class HttpResponse:
    status: int
    headers: dict[str, str]
    body: Any


Transport = Callable[[str, dict, dict, float], HttpResponse]


def requests_transport(url: str, headers: dict, payload: dict, timeout: float) -> HttpResponse:
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    except (requests.ConnectionError, requests.Timeout) as e:
        raise TransientError(0, f"network: {e}") from e
    try:
        body = r.json()
    except ValueError:
        body = r.text
    return HttpResponse(r.status_code, dict(r.headers), body)


# ---------- wire format ----------


def to_contents(messages: list[Message]) -> list[dict]:
    contents = []
    for m in messages:
        parts: list[dict] = []
        if m.text:
            p: dict[str, Any] = {"text": m.text}
            if m.text_signature:
                p["thoughtSignature"] = m.text_signature
            parts.append(p)
        for c in m.tool_calls:
            fc: dict[str, Any] = {"name": c.name, "args": c.args}
            if c.id:
                fc["id"] = c.id
            p = {"functionCall": fc}
            if c.signature:
                p["thoughtSignature"] = c.signature
            parts.append(p)
        for r in m.tool_results:
            key = "error" if r.is_error else "result"
            fr: dict[str, Any] = {"name": r.name, "response": {key: r.content}}
            if r.id:
                fr["id"] = r.id
            parts.append({"functionResponse": fr})
        if parts:
            contents.append({"role": m.role, "parts": parts})
    return contents


def build_request(
    messages: list[Message],
    tools: list[ToolSpec] | None,
    system: str | None,
    config: GenerateConfig,
) -> dict:
    gen: dict[str, Any] = {
        "temperature": config.temperature,
        "maxOutputTokens": config.max_output_tokens,
    }
    if config.seed is not None:
        gen["seed"] = config.seed
    req: dict[str, Any] = {"contents": to_contents(messages), "generationConfig": gen}
    if system:
        req["systemInstruction"] = {"parts": [{"text": system}]}
    if tools:
        req["tools"] = [
            {
                "functionDeclarations": [
                    {"name": t.name, "description": t.description, "parameters": t.parameters}
                    for t in tools
                ]
            }
        ]
        req["toolConfig"] = {"functionCallingConfig": {"mode": "AUTO"}}
    return req


def parse_response(body: dict) -> LLMResponse:
    meta = body.get("usageMetadata", {})
    usage = Usage(
        input_tokens=meta.get("promptTokenCount", 0),
        output_tokens=meta.get("candidatesTokenCount", 0),
        thinking_tokens=meta.get("thoughtsTokenCount", 0),
        cached_tokens=meta.get("cachedContentTokenCount", 0),
    )
    cands = body.get("candidates") or []
    if not cands:
        reason = body.get("promptFeedback", {}).get("blockReason", "NO_CANDIDATES")
        return LLMResponse(Message(role="model"), usage, reason, body.get("modelVersion"))

    cand = cands[0]
    msg = Message(role="model")
    texts: list[str] = []
    for part in cand.get("content", {}).get("parts", []):
        if part.get("thought"):
            continue  # thinking summaries are not part of the answer
        if "functionCall" in part:
            fc = part["functionCall"]
            msg.tool_calls.append(
                ToolCall(
                    name=fc["name"],
                    args=fc.get("args") or {},
                    id=fc.get("id"),
                    signature=part.get("thoughtSignature"),
                )
            )
        elif "text" in part:
            texts.append(part["text"])
            if part.get("thoughtSignature") and not msg.text_signature:
                msg.text_signature = part["thoughtSignature"]
    msg.text = "".join(texts) or None
    return LLMResponse(msg, usage, cand.get("finishReason"), body.get("modelVersion"))


def _parse_delay(s: str) -> float | None:
    m = re.fullmatch(r"([\d.]+)s", s.strip())
    return float(m.group(1)) if m else None


def classify_error(status: int, headers: dict, body: Any) -> Exception:
    err = body.get("error", {}) if isinstance(body, dict) else {}
    message = err.get("message") or (body if isinstance(body, str) else "") or "unknown error"
    details = err.get("details", [])

    if status == 429:
        for d in details:
            if d.get("@type", "").endswith("QuotaFailure"):
                for v in d.get("violations", []):
                    if "PerDay" in v.get("quotaId", ""):
                        return QuotaExhausted(f"daily quota hit: {v.get('quotaId')}")

    if status in RETRYABLE:
        hint = None
        for d in details:
            if d.get("@type", "").endswith("RetryInfo") and d.get("retryDelay"):
                hint = _parse_delay(d["retryDelay"])
        if hint is None:
            ra = {k.lower(): v for k, v in headers.items()}.get("retry-after")
            if ra and ra.replace(".", "", 1).isdigit():
                hint = float(ra)
        return TransientError(status, message, hint)

    return RequestError(status, message)


def estimate_tokens(request: dict) -> int:
    # ~4 characters per token is close enough for pacing
    return len(json.dumps(request)) // 4


# ---------- client ----------


class GeminiClient:
    def __init__(
        self,
        api_key: str | None,
        model: str,
        *,
        limiter: RateLimiter | None = None,
        cache: ResponseCache | None = None,
        retry: RetryPolicy | None = None,
        meter: UsageMeter | None = None,
        transport: Transport = requests_transport,
        timeout: float = 120.0,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.api_key = api_key
        self.model = model
        self.limiter = limiter
        self.cache = cache
        self.retry = retry or RetryPolicy()
        self.meter = meter or UsageMeter()
        self.transport = transport
        self.timeout = timeout
        self.sleep = sleep

    def generate(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        system: str | None = None,
        config: GenerateConfig | None = None,
    ) -> LLMResponse:
        config = config or GenerateConfig()
        request = build_request(messages, tools, system, config)

        if self.cache:
            body = self.cache.get(self.model, request)
            if body is not None:
                resp = parse_response(body)
                resp.from_cache = True
                self.meter.add_call(resp.usage, from_cache=True)
                return resp

        body = call_with_retry(
            lambda: self._post(request),
            self.retry,
            sleep=self.sleep,
            on_retry=lambda *_: self.meter.add_retry(),
        )
        if self.cache:
            self.cache.put(self.model, request, body)
        resp = parse_response(body)
        self.meter.add_call(resp.usage, from_cache=False)
        return resp

    def _post(self, request: dict) -> dict:
        if not self.api_key:
            raise RequestError(0, "GEMINI_API_KEY is not set")
        ticket = self.limiter.acquire(estimate_tokens(request)) if self.limiter else None
        url = f"{API_ROOT}/models/{self.model}:generateContent"
        headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}
        r = self.transport(url, headers, request, self.timeout)
        if r.status != 200:
            raise classify_error(r.status, r.headers, r.body)
        if ticket is not None:
            total = r.body.get("usageMetadata", {}).get("totalTokenCount")
            if total is not None:
                self.limiter.settle(ticket, total)
        return r.body
