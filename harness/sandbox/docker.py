"""Docker sandbox.

One long-lived container per task, with the workspace bind-mounted at /work.
Locked down: no network, capped CPU / memory / process count, read-only root
filesystem, all Linux capabilities dropped, no privilege escalation.

Timeouts are enforced inside the container with `timeout -k`, because killing
the `docker exec` client on the host leaves the process running in the
container. A host-side timeout is only a backstop.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import time
import uuid

from .base import ExecResult, Sandbox, clip

DEFAULT_IMAGE = "harness-sandbox:py312"

DOCKERFILE = """\
FROM python:3.12-slim
RUN pip install --no-cache-dir pytest==8.3.5
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOME=/tmp
WORKDIR /work
"""


class DockerError(RuntimeError):
    pass


def _docker(*args: str, timeout: float = 120, input: bytes | None = None) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            ["docker", *args], capture_output=True, timeout=timeout, input=input
        )
    except FileNotFoundError as e:
        raise DockerError("docker CLI not found. Install Docker or use --sandbox local") from e


def image_exists(image: str) -> bool:
    return _docker("image", "inspect", image).returncode == 0


def build_image(image: str = DEFAULT_IMAGE) -> None:
    r = _docker("build", "-t", image, "-", input=DOCKERFILE.encode(), timeout=900)
    if r.returncode != 0:
        raise DockerError(f"image build failed:\n{r.stderr.decode(errors='replace')[-2000:]}")


class DockerSandbox(Sandbox):
    workdir = "/work"

    def __init__(
        self,
        source=None,
        image: str = DEFAULT_IMAGE,
        cpus: float = 1.0,
        mem_mb: int = 1024,
        pids: int = 256,
    ):
        super().__init__(source)
        self.image = image
        self.cpus, self.mem_mb, self.pids = cpus, mem_mb, pids
        self.name = f"harness-{uuid.uuid4().hex[:12]}"
        self._started = False

    def start(self) -> None:
        if not image_exists(self.image):
            if self.image != DEFAULT_IMAGE:
                raise DockerError(f"image {self.image} not found")
            build_image(self.image)
        user = []
        if hasattr(os, "getuid"):  # keep files in the bind mount owned by you
            user = ["--user", f"{os.getuid()}:{os.getgid()}"]
        r = _docker(
            "run", "-d", "--rm",
            "--name", self.name,
            "--network", "none",
            "--cpus", str(self.cpus),
            "--memory", f"{self.mem_mb}m",
            "--memory-swap", f"{self.mem_mb}m",
            "--pids-limit", str(self.pids),
            "--read-only",
            "--tmpfs", "/tmp:rw,size=256m",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            *user,
            "-v", f"{self.root}:/work",
            "-w", "/work",
            "-e", "HOME=/tmp",
            self.image,
            "sleep", "infinity",
        )  # fmt: skip
        if r.returncode != 0:
            raise DockerError(f"container start failed: {r.stderr.decode(errors='replace').strip()}")
        self._started = True

    def exec(self, command: str, timeout: float = 60.0) -> ExecResult:
        if not self._started:
            raise DockerError("sandbox not started")
        t = max(1, int(timeout))
        wrapped = f"timeout -k 2 {t} sh -c {shlex.quote(command)}"
        t0 = time.perf_counter()
        try:
            r = _docker("exec", self.name, "sh", "-c", f"{wrapped} 2>&1", timeout=t + 15)
        except subprocess.TimeoutExpired:
            # backstop: the container is in a bad state, restart it
            self._restart()
            return ExecResult(-9, "command hung and the sandbox was restarted", True,
                              time.perf_counter() - t0)
        text, cut = clip(r.stdout + r.stderr)
        timed_out = r.returncode in (124, 137)
        if timed_out:
            text += f"\n[timed out after {t}s]"
        return ExecResult(r.returncode, text, timed_out, time.perf_counter() - t0, cut)

    def _restart(self) -> None:
        _docker("rm", "-f", self.name, timeout=30)
        self._started = False
        self.name = f"harness-{uuid.uuid4().hex[:12]}"
        self.start()

    def stop(self) -> None:
        if self._started:
            _docker("rm", "-f", self.name, timeout=30)
            self._started = False
        super().stop()
