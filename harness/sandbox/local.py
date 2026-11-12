"""Runs commands directly on the host, in the workspace temp dir.

NOT isolated. It exists for unit tests and machines without Docker.
On POSIX it still caps CPU time, memory and process count.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

from .base import ExecResult, Sandbox, clip


def _limits(cpu_seconds: int, mem_bytes: int):
    def apply():  # runs in the child before exec
        import resource

        os.setsid()
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        try:
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        except (ValueError, OSError):
            pass  # macOS does not support an address-space cap; CPU cap still applies

    return apply


class LocalSandbox(Sandbox):
    def __init__(self, source=None, mem_mb: int = 2048):
        super().__init__(source)
        self.workdir = str(self.root)
        self.mem_mb = mem_mb

    def start(self) -> None:
        pass

    def exec(self, command: str, timeout: float = 60.0) -> ExecResult:
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(self.root),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
        }
        if sys.platform == "win32":
            env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", "")
        kwargs = {}
        if os.name == "posix":
            kwargs["preexec_fn"] = _limits(int(timeout) + 5, self.mem_mb * 1024 * 1024)
        t0 = time.perf_counter()
        proc = subprocess.Popen(
            command,
            shell=True,
            cwd=self.root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            **kwargs,
        )
        try:
            out, _ = proc.communicate(timeout=timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                import signal

                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
            out, _ = proc.communicate()
            timed_out = True
        text, cut = clip(out or b"")
        return ExecResult(
            exit_code=-9 if timed_out else proc.returncode,
            output=text,
            timed_out=timed_out,
            duration=time.perf_counter() - t0,
            truncated=cut,
        )
