"""Run-time support for task generation: environment checks and the live progress line.

``check_github`` makes one read-only GraphQL call (``rateLimit``), which GitHub only answers for a
valid token, so it proves the token works and shows the quota left. ``check_docker`` asks the daemon
for its version. Both return a result with a plain-language ``detail`` instead of raising, so the
CLI can report every problem at once.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TextIO

from cascade.errors import CascadeError
from cascade.taskgen.build_env import DockerRun, default_docker_run
from cascade.taskgen.github import TOKEN_ENV, GitHubClient

_RATE_QUERY = "query { rateLimit { limit remaining resetAt } }"


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


def check_github(client_factory: Callable[[], GitHubClient] = GitHubClient) -> CheckResult:
    if not os.environ.get(TOKEN_ENV):
        return CheckResult(
            "github", False,
            f"{TOKEN_ENV} is not set. Create a read-only token (public repos is enough) and "
            f"`export {TOKEN_ENV}=...` in the shell that runs cascade.",
        )  # fmt: skip
    try:
        limit = client_factory().query(_RATE_QUERY, {})["rateLimit"]
    except CascadeError as e:
        return CheckResult("github", False, f"{TOKEN_ENV} is set but GitHub rejected it: {e}")
    return CheckResult(
        "github", True,
        f"token accepted; {limit['remaining']}/{limit['limit']} GraphQL points left "
        f"(resets {limit['resetAt']})",
    )  # fmt: skip


def check_docker(run_fn: DockerRun = default_docker_run) -> CheckResult:
    try:
        proc = run_fn(["docker", "info", "--format", "{{.ServerVersion}}"], 30.0)
    except FileNotFoundError:
        return CheckResult("docker", False, "the `docker` command is not installed or not on PATH")
    except Exception as e:
        return CheckResult("docker", False, f"could not run `docker info`: {e}")
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or ["no output"]
        return CheckResult(
            "docker", False, f"the Docker daemon is not reachable (start Docker): {tail[0]}"
        )
    return CheckResult("docker", True, f"daemon running, server version {proc.stdout.strip()}")


class ProgressPrinter:
    """One live line: ``[repo] stage done/total PRs | validated so far N``.

    On a terminal it rewrites the line in place; otherwise (a log file, CI) it prints a new line at
    most every ``every_s`` seconds and always at the end of a stage. Safe to call from threads.
    """

    def __init__(
        self,
        stream: TextIO | None = None,
        *,
        every_s: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._stream = stream or sys.stderr
        self._tty = self._stream.isatty()
        self._every = every_s
        self._clock = clock
        self._last = float("-inf")
        self._open = False  # an in-place line is on screen and needs a newline before other output
        self._lock = threading.Lock()

    def __call__(self, repo: str, stage: str, done: int, total: int, validated: int) -> None:
        unit = "month slices" if stage == "mine" else "PRs"
        line = f"[{repo}] {stage:8s} {done}/{total} {unit} | validated so far: {validated}"
        final = done >= total
        with self._lock:
            now = self._clock()
            if self._tty:
                self._stream.write("\r\x1b[K" + line + ("\n" if final else ""))
                self._open = not final
            elif final or now - self._last >= self._every:
                self._stream.write(line + "\n")
            else:
                return
            self._last = now
            self._stream.flush()

    def note(self, message: str) -> None:
        """Print a message without leaving a half-drawn progress line behind."""
        with self._lock:
            if self._open:
                self._stream.write("\n")
                self._open = False
            self._stream.write(message + "\n")
            self._stream.flush()
