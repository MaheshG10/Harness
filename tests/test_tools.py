import pytest

from harness.agent.tools import ToolContext, build_registry
from harness.llm.types import ToolCall
from harness.sandbox import LocalSandbox

SRC = "def mean(xs):\n    return sum(xs) / (len(xs) - 1)\n\n\ndef total(xs):\n    return sum(xs)\n"


@pytest.fixture
def env(tmp_path):
    (tmp_path / "stats.py").write_text(SRC)
    (tmp_path / "test_stats.py").write_text(
        "from stats import mean\n\ndef test_mean():\n    assert mean([1, 2, 3]) == 2\n"
    )
    (tmp_path / "__pycache__").mkdir()
    sb = LocalSandbox(tmp_path)
    ctx = ToolContext(sb)
    reg = build_registry()

    def call(name, **args):
        return reg.call(ToolCall(name, args), ctx)

    yield call, ctx
    sb.stop()


def test_list_files_skips_noise(env):
    call, _ = env
    out = call("list_files").content
    assert "stats.py" in out and "__pycache__" not in out


def test_read_file_numbers_lines(env):
    call, _ = env
    out = call("read_file", path="stats.py", start=2, end=2).content
    assert out == "2|     return sum(xs) / (len(xs) - 1)"


def test_read_missing_file_has_hint(env):
    call, _ = env
    r = call("read_file", path="nope.py")
    assert r.is_error and "list_files" in r.content


def test_edit_then_tests_pass(env):
    call, ctx = env
    assert "1 failed" in call("run_shell", command="python -m pytest -q").content
    r = call("edit_file", path="stats.py", old="(len(xs) - 1)", new="len(xs)")
    assert not r.is_error and "edited stats.py" in r.content
    out = call("run_shell", command="python -m pytest -q").content
    assert out.startswith("[exit code 0") and "1 passed" in out
    assert ctx.files_changed == {"stats.py"}


def test_edit_not_found_points_to_line(env):
    call, _ = env
    r = call("edit_file", path="stats.py", old="def total(xs):\n  return sum(xs)", new="x")
    assert r.is_error and "near line 5" in r.content


def test_edit_ambiguous_needs_replace_all(env):
    call, _ = env
    r = call("edit_file", path="stats.py", old="sum(xs)", new="sum(list(xs))")
    assert r.is_error and "appears 2 times" in r.content
    r = call("edit_file", path="stats.py", old="sum(xs)", new="sum(list(xs))", replace_all=True)
    assert not r.is_error


def test_write_file_creates_dirs(env):
    call, ctx = env
    call("write_file", path="pkg/new.py", content="x = 1\n")
    assert (ctx.sandbox.root / "pkg" / "new.py").read_text() == "x = 1\n"


def test_search_code(env):
    call, _ = env
    assert call("search_code", pattern="sum(xs)").content.count("stats.py:") == 2
    assert "stats.py:1:" in call("search_code", pattern=r"def \w+\(xs\)", regex=True).content
    assert call("search_code", pattern="zzz").content == "no matches"


def test_path_escape_is_an_error_not_a_crash(env):
    call, _ = env
    r = call("read_file", path="../../etc/passwd")
    assert r.is_error and "outside the workspace" in r.content


def test_finish_marks_context(env):
    call, ctx = env
    call("finish", summary="fixed off-by-one in mean")
    assert ctx.finished and ctx.summary == "fixed off-by-one in mean"


def test_shell_timeout_reported(env):
    call, _ = env
    assert call("run_shell", command="sleep 5", timeout=1).content.startswith("[timed out")
