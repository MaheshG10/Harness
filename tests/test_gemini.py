import pytest

from harness.llm import (
    GeminiClient,
    GenerateConfig,
    Message,
    QuotaExhausted,
    RequestError,
    ResponseCache,
    ToolCall,
    ToolResult,
    ToolSpec,
)
from harness.llm.gemini import HttpResponse, build_request, parse_response


def ok_body(parts, usage=(10, 5)):
    return {
        "candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": "STOP"}],
        "usageMetadata": {
            "promptTokenCount": usage[0],
            "candidatesTokenCount": usage[1],
            "totalTokenCount": sum(usage),
        },
        "modelVersion": "gemini-test-001",
    }


class FakeTransport:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, url, headers, payload, timeout):
        self.requests.append((url, headers, payload))
        return self.responses.pop(0)


def client(transport, **kw):
    return GeminiClient("test-key", "gemini-test", transport=transport, sleep=lambda s: None, **kw)


def test_tool_call_round_trip_keeps_signature():
    body = ok_body(
        [{"functionCall": {"name": "read_file", "args": {"path": "a.py"}}, "thoughtSignature": "sig123"}]
    )
    resp = parse_response(body)
    call = resp.message.tool_calls[0]
    assert (call.name, call.args, call.signature) == ("read_file", {"path": "a.py"}, "sig123")

    history = [
        Message.user("fix it"),
        resp.message,
        Message.results([ToolResult("read_file", "print(1)")]),
    ]
    req = build_request(history, None, None, GenerateConfig())
    model_turn = req["contents"][1]
    assert model_turn["role"] == "model"
    assert model_turn["parts"][0]["thoughtSignature"] == "sig123"
    fr = req["contents"][2]["parts"][0]["functionResponse"]
    assert fr == {"name": "read_file", "response": {"result": "print(1)"}}


def test_tool_errors_are_sent_as_error_key():
    req = build_request(
        [Message.results([ToolResult("run", "boom", is_error=True)])], None, None, GenerateConfig()
    )
    assert req["contents"][0]["parts"][0]["functionResponse"]["response"] == {"error": "boom"}


def test_request_has_tools_system_and_seed():
    spec = ToolSpec("finish", "Stop.", {"type": "object", "properties": {}})
    req = build_request([Message.user("hi")], [spec], "be brief", GenerateConfig(seed=7))
    assert req["systemInstruction"]["parts"][0]["text"] == "be brief"
    assert req["tools"][0]["functionDeclarations"][0]["name"] == "finish"
    assert req["generationConfig"]["seed"] == 7


def test_thought_parts_are_dropped():
    resp = parse_response(ok_body([{"text": "thinking...", "thought": True}, {"text": "answer"}]))
    assert resp.message.text == "answer"


def test_blocked_prompt_returns_empty_message():
    resp = parse_response({"promptFeedback": {"blockReason": "SAFETY"}})
    assert resp.finish_reason == "SAFETY" and resp.message.text is None


def test_retries_429_with_retry_info_then_succeeds():
    rate_limited = HttpResponse(
        429,
        {},
        {
            "error": {
                "message": "Resource exhausted",
                "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "3s"}],
            }
        },
    )
    t = FakeTransport(rate_limited, HttpResponse(200, {}, ok_body([{"text": "hi"}])))
    c = client(t)
    resp = c.generate([Message.user("hello")])
    assert resp.message.text == "hi"
    assert c.meter.snapshot()["retries"] == 1
    assert t.requests[0][1]["x-goog-api-key"] == "test-key"


def test_daily_quota_429_is_not_retried():
    daily = HttpResponse(
        429,
        {},
        {
            "error": {
                "message": "quota",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                        "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}],
                    }
                ],
            }
        },
    )
    t = FakeTransport(daily)
    with pytest.raises(QuotaExhausted):
        client(t).generate([Message.user("hello")])
    assert len(t.requests) == 1


def test_bad_request_fails_fast():
    t = FakeTransport(HttpResponse(400, {}, {"error": {"message": "bad schema"}}))
    with pytest.raises(RequestError):
        client(t).generate([Message.user("hello")])
    assert len(t.requests) == 1


def test_cache_hit_skips_network_and_counts_usage(tmp_path):
    cache = ResponseCache(tmp_path)
    t = FakeTransport(HttpResponse(200, {}, ok_body([{"text": "hi"}], usage=(100, 20))))
    c = client(t, cache=cache)
    first = c.generate([Message.user("hello")], config=GenerateConfig(seed=1))
    second = c.generate([Message.user("hello")], config=GenerateConfig(seed=1))
    assert len(t.requests) == 1
    assert not first.from_cache and second.from_cache
    snap = c.meter.snapshot()
    assert snap["calls"] == 2 and snap["cache_hits"] == 1 and snap["total_tokens"] == 240


def test_missing_key_is_a_clear_error():
    c = GeminiClient(None, "m", transport=FakeTransport())
    with pytest.raises(RequestError, match="GEMINI_API_KEY"):
        c.generate([Message.user("hi")])


def test_model_message_with_text_and_calls():
    msg = Message(role="model", text="let me look", tool_calls=[ToolCall("ls", {})])
    req = build_request([msg], None, None, GenerateConfig())
    kinds = [list(p)[0] for p in req["contents"][0]["parts"]]
    assert kinds == ["text", "functionCall"]
