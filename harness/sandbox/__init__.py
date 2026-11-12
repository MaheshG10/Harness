from .base import ExecResult, PathEscape, Sandbox
from .docker import DEFAULT_IMAGE, DockerError, DockerSandbox, build_image
from .local import LocalSandbox


def make_sandbox(kind: str, source=None, image: str = DEFAULT_IMAGE) -> Sandbox:
    if kind == "docker":
        return DockerSandbox(source, image=image)
    if kind == "local":
        return LocalSandbox(source)
    raise ValueError(f"unknown sandbox kind: {kind}")


__all__ = [
    "DEFAULT_IMAGE",
    "DockerError",
    "DockerSandbox",
    "ExecResult",
    "LocalSandbox",
    "PathEscape",
    "Sandbox",
    "build_image",
    "make_sandbox",
]
