import os
import shutil
import subprocess

import pytest

from harness.sandbox import DockerSandbox, LocalSandbox, PathEscape
from harness.sandbox.base import clip

TEST_IMAGE = os.environ.get("HARNESS_TEST_IMAGE", "harness-sandbox:py312")


def docker_ready() -> bool:
    if not shutil.which("docker"):
        return False
    r = subprocess.run(["docker", "image", "inspect", TEST_IMAGE], capture_output=True)
    return r.returncode == 0


def backends():
    return [
        pytest.param(lambda src: LocalSandbox(src), id="local"),
        pytest.param(
            lambda src: DockerSandbox(src, image=TEST_IMAGE),
            id="docker",
            marks=pytest.mark.skipif(not docker_ready(), reason=f"docker image {TEST_IMAGE} not available"),
        ),
    ]


@pytest.fixture
def project(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text("def add(a, b):\n    return a + b\n")
    (tmp_path / "test_mod.py").write_text(
        "from pkg.mod import add\n\ndef test_add():\n    assert add(2, 2) == 4\n"
    )
    return tmp_path


@pytest.mark.parametrize("make", backends())
def test_runs_tests_in_workspace_copy(make, project):
    with make(project) as sb:
        r = sb.exec("python -m pytest -q", timeout=60)
        assert r.exit_code == 0, r.output
        assert "1 passed" in r.output
        (sb.root / "pkg" / "mod.py").write_text("def add(a, b):\n    return a - b\n")
        r = sb.exec("python -m pytest -q", timeout=60)
        assert r.exit_code != 0 and "1 failed" in r.output
    # the original project is untouched
    assert "a + b" in (project / "pkg" / "mod.py").read_text()


@pytest.mark.parametrize("make", backends())
def test_timeout_kills_command(make, project):
    with make(project) as sb:
        r = sb.exec("sleep 30", timeout=1)
        assert r.timed_out and r.duration < 10


@pytest.mark.parametrize("make", backends())
def test_workspace_is_removed_on_stop(make, project):
    sb = make(project)
    with sb:
        root = sb.root
    assert not root.exists()


@pytest.mark.skipif(not docker_ready(), reason="docker not available")
def test_docker_has_no_network_and_readonly_root(project):
    with DockerSandbox(project, image=TEST_IMAGE) as sb:
        r = sb.exec(
            "python -c \"import socket; socket.create_connection(('1.1.1.1', 53), timeout=2)\"",
            timeout=15,
        )
        assert r.exit_code != 0
        assert sb.exec("touch /etc/x", timeout=10).exit_code != 0
        assert sb.exec("touch /work/ok && touch /tmp/ok", timeout=10).exit_code == 0


def test_resolve_blocks_escapes(project):
    sb = LocalSandbox(project)
    try:
        assert sb.resolve("pkg/mod.py") == sb.root / "pkg" / "mod.py"
        assert sb.resolve(sb.workdir + "/pkg") == sb.root / "pkg"
        for bad in ["../x", "/etc/passwd", "pkg/../../x"]:
            with pytest.raises(PathEscape):
                sb.resolve(bad)
        os.symlink("/etc", sb.root / "link")
        with pytest.raises(PathEscape):
            sb.resolve("link/passwd")
    finally:
        sb.stop()


def test_docker_paths_accept_work_prefix(project):
    sb = DockerSandbox(project)  # not started; resolve is pure
    try:
        assert sb.resolve("/work/pkg/mod.py") == sb.root / "pkg" / "mod.py"
        assert sb.resolve("/work") == sb.root
        with pytest.raises(PathEscape):
            sb.resolve("/workshop/x")
    finally:
        sb.stop()


def test_clip_keeps_head_and_tail():
    data = b"HEAD" + b"x" * 1000 + b"TAIL"
    text, cut = clip(data, limit=100)
    assert cut and text.startswith("HEAD") and text.endswith("TAIL")
    assert "bytes cut" in text
