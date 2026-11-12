"""End-to-end: scripted model, real tools, real sandbox, real tests."""

from pathlib import Path

import pytest

from harness.agent import Agent, AgentConfig, ToolContext, build_registry
from harness.llm.scripted import ScriptedLLM, call, say
from harness.llm.types import Message, ToolCall
from harness.sandbox import DockerSandbox, LocalSandbox
from harness.task import load_task

from .test_sandbox import TEST_IMAGE, docker_ready

TASKS = Path(__file__).parent.parent / "examples" / "tasks"

SLUG_IMPL = '''import re


def slugify(text: str, max_length: int = 50) -> str:
    """Turn a title into a URL slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_length].rstrip("-")
'''

SCRIPTS = {
    "mean_off_by_one": [
        call("list_files"),
        call("run_shell", command="python -m pytest -q"),
        call("read_file", path="stats.py"),
        call("edit_file", path="stats.py", old="sum(xs) / (len(xs) - 1)", new="sum(xs) / len(xs)"),
        call("run_shell", command="python -m pytest -q"),
        call("finish", summary="mean divided by n-1; now divides by n"),
    ],
    "slugify": [
        call("read_file", path="text_utils.py"),
        call("read_file", path="test_text_utils.py"),
        call("write_file", path="text_utils.py", content=SLUG_IMPL),
        call("run_shell", command="python -m pytest -q"),
        call("finish", summary="implemented slugify with a regex"),
    ],
    "config_parser": [
        call("search_code", pattern="def split_pair"),
        call("read_file", path="confkit/util.py"),
        call("edit_file", path="confkit/util.py", old='line.split("=")', new='line.split("=", 1)'),
        call("run_shell", command="python -m pytest -q"),
        call("finish", summary="split only on the first '='"),
    ],
}


def sandboxes():
    return [
        pytest.param(LocalSandbox, id="local"),
        pytest.param(
            lambda src: DockerSandbox(src, image=TEST_IMAGE),
            id="docker",
            marks=pytest.mark.skipif(not docker_ready(), reason="docker image not available"),
        ),
    ]


@pytest.mark.parametrize("make", sandboxes())
@pytest.mark.parametrize("task_id", sorted(SCRIPTS))
def test_demo_tasks_solved_end_to_end(make, task_id):
    task = load_task(TASKS / task_id)
    llm = ScriptedLLM(SCRIPTS[task_id])
    with make(task.repo) as sb:
        before = sb.exec(task.test_command, timeout=60)
        assert before.exit_code != 0  # the task really starts broken
        result = Agent(llm, build_registry()).run(task.prompt, ToolContext(sb))
        assert result.status == "finished", result.error
        assert result.tool_errors == 0, [r.content for s in result.steps for r in s.results if r.is_error]
        after = sb.exec(task.test_command, timeout=60)
        assert after.exit_code == 0, after.output
    assert result.usage.total == 120 * len(result.steps)


def run_local(turns, **cfg):
    task = load_task(TASKS / "mean_off_by_one")
    llm = ScriptedLLM(turns)
    with LocalSandbox(task.repo) as sb:
        ctx = ToolContext(sb)
        return Agent(llm, build_registry(), AgentConfig(**cfg)).run(task.prompt, ctx), llm


def test_stops_at_max_steps():
    result, _ = run_local([call("list_files")] * 10, max_steps=3)
    assert result.status == "max_steps" and len(result.steps) == 3


def test_stops_on_token_budget():
    result, _ = run_local([call("list_files")] * 10, max_total_tokens=250)
    assert result.status == "token_budget" and len(result.steps) == 3


def test_idle_model_is_nudged_then_stopped():
    result, llm = run_local([say("I think the bug is in mean."), say("Yes, definitely.")])
    assert result.status == "no_action"
    second_view = llm.seen[1]
    assert second_view[-1].role == "user" and "without using a tool" in second_view[-1].text


def test_nudge_after_tool_results_keeps_turns_alternating():
    result, llm = run_local([call("list_files"), Message(role="model"), call("finish", summary="x")])
    roles = [m.role for m in llm.seen[2]]
    assert all(a != b for a, b in zip(roles, roles[1:])), roles
    assert result.status == "finished"


def test_tool_errors_flow_back_to_model():
    result, llm = run_local([call("read_file", path="missing.py"), call("finish", summary="gave up")])
    assert result.tool_errors == 1
    fed_back = llm.seen[1][-1].tool_results[0]
    assert fed_back.is_error and "does not exist" in fed_back.content


def test_calls_after_finish_get_skipped_responses():
    both = Message(role="model", tool_calls=[ToolCall("finish", {"summary": "s"}), ToolCall("list_files", {})])
    result, _ = run_local([both])
    assert result.status == "finished"
    assert [r.content for r in result.steps[0].results] == ["done", "skipped: run already finished"]


def test_llm_error_ends_run_cleanly():
    from harness.llm.errors import RequestError

    def boom(_):
        raise RequestError(400, "bad request")

    result, _ = run_local([boom])
    assert result.status == "error" and "bad request" in result.error
