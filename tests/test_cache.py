import pytest

from harness.llm.cache import ResponseCache, request_key
from harness.llm.errors import CacheMiss

REQ = {"contents": [{"role": "user", "parts": [{"text": "hi"}]}], "generationConfig": {"seed": 1}}
RESP = {"candidates": [{"content": {"parts": [{"text": "hello"}]}}]}


def test_key_is_stable_across_dict_order():
    a = {"x": 1, "y": {"b": 2, "a": 1}}
    b = {"y": {"a": 1, "b": 2}, "x": 1}
    assert request_key("m", a) == request_key("m", b)


def test_key_changes_with_model_and_seed():
    other_seed = {**REQ, "generationConfig": {"seed": 2}}
    assert request_key("m", REQ) != request_key("n", REQ)
    assert request_key("m", REQ) != request_key("m", other_seed)


def test_read_write_round_trip(tmp_path):
    c = ResponseCache(tmp_path)
    assert c.get("m", REQ) is None
    c.put("m", REQ, RESP)
    assert c.get("m", REQ) == RESP


def test_replay_raises_on_miss(tmp_path):
    c = ResponseCache(tmp_path, mode="replay")
    with pytest.raises(CacheMiss):
        c.get("m", REQ)


def test_replay_never_writes(tmp_path):
    c = ResponseCache(tmp_path, mode="replay")
    c.put("m", REQ, RESP)
    assert not any(tmp_path.rglob("*.json"))


def test_record_mode_ignores_existing(tmp_path):
    ResponseCache(tmp_path).put("m", REQ, RESP)
    rec = ResponseCache(tmp_path, mode="record")
    assert rec.get("m", REQ) is None
