"""Per-repo Docker image and the executor that runs tests in it with the network off.

One image per repo: a pinned Python (installed by uv), the repo's test dependencies, and the repo
itself installed *editable* so that `import <pkg>` and `importlib.metadata` both work. The install
runs at build time (network on) with the source at /workspace; the source is then deleted from the
image, leaving a `.pth` that points at /workspace. At test time a task's tree is bind-mounted at
/workspace, so editable imports resolve to it, and the container needs no network.

Everything that shells out to docker takes an injectable `run_fn`, so the logic is unit-tested
without a daemon. The real build and real container runs are not exercised by CI.
"""

from __future__ import annotations

import hashlib
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

from cascade.config import ResolvedRepo
from cascade.errors import CascadeError
from cascade.harness.grading import GradingRunner
from cascade.harness.sandbox import RunFn, SandboxSpec, prepare_workspace
from cascade.taskgen.results import RunResult, parse_report

# Infrastructure pins (not model names or prices). Verified to exist when written: the uv image
# tag on ghcr.io and the Debian tag on Docker Hub. Changing either changes the image tag.
UV_VERSION = "0.11.2"
BASE_IMAGE = "debian:bookworm-slim"

PLUGIN_NAME = "cascade_pytest_report.py"
REPORT_FILE = ".cascade_results.jsonl"
CONTAINER_PLUGIN_DIR = "/opt/cascade"

DockerRun = RunFn


class DockerUnavailableError(CascadeError):
    """The docker CLI or daemon is not usable."""


class ImageBuildError(CascadeError):
    """`docker build` failed."""


def default_docker_run(
    argv: Sequence[str], timeout: float | None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603  (argv list, no shell)
        list(argv), capture_output=True, text=True, timeout=timeout, check=False
    )


def plugin_path() -> Path:
    path = Path(__file__).resolve().parents[2] / "docker" / PLUGIN_NAME
    if not path.is_file():
        raise CascadeError(f"pytest report plugin not found at {path}")
    return path


def render_dockerfile(repo: ResolvedRepo) -> str:
    py = repo.python_version
    return f"""\
FROM {BASE_IMAGE}
COPY --from=ghcr.io/astral-sh/uv:{UV_VERSION} /uv /uvx /bin/
ENV UV_PYTHON_INSTALL_DIR=/opt/python \\
    UV_LINK_MODE=copy \\
    UV_COMPILE_BYTECODE=1 \\
    UV_NO_CACHE=1 \\
    VIRTUAL_ENV=/opt/venv \\
    PATH=/opt/venv/bin:$PATH
RUN uv python install {py} && uv venv --python {py} /opt/venv
COPY repo/ /workspace/
WORKDIR /workspace
RUN {repo.install_cmd}
RUN rm -rf /workspace/* /workspace/.[!.]*
COPY {PLUGIN_NAME} {CONTAINER_PLUGIN_DIR}/{PLUGIN_NAME}
"""


def image_tag(repo: ResolvedRepo, dockerfile: str, ref_commit: str, plugin_text: str) -> str:
    """Deterministic tag: identical inputs always give the same tag, so builds are reusable."""
    digest = hashlib.sha256(
        "\n".join([dockerfile, ref_commit, plugin_text, repo.test_cmd]).encode()
    ).hexdigest()[:12]
    return f"cascade-env/{repo.slug.lower()}:{digest}"


def docker_available(run_fn: DockerRun = default_docker_run) -> bool:
    try:
        proc = run_fn(["docker", "info", "--format", "{{.ServerVersion}}"], 30.0)
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def image_exists(tag: str, run_fn: DockerRun = default_docker_run) -> bool:
    try:
        return run_fn(["docker", "image", "inspect", tag], 30.0).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def build_image(
    repo: ResolvedRepo,
    clone: Path,
    ref_commit: str,
    *,
    run_fn: DockerRun = default_docker_run,
    build_timeout_s: float = 1800.0,
) -> str:
    """Build (or reuse) the repo image at ``ref_commit``. Returns its tag."""
    dockerfile = render_dockerfile(repo)
    plugin_text = plugin_path().read_text(encoding="utf-8")
    tag = image_tag(repo, dockerfile, ref_commit, plugin_text)
    if image_exists(tag, run_fn):
        return tag
    with tempfile.TemporaryDirectory(prefix="cascade-build-") as tmp:
        ctx = Path(tmp)
        prepare_workspace(clone, ref_commit, ctx / "repo")
        (ctx / "Dockerfile").write_text(dockerfile, encoding="utf-8")
        (ctx / PLUGIN_NAME).write_text(plugin_text, encoding="utf-8")
        proc = run_fn(["docker", "build", "--tag", tag, str(ctx)], build_timeout_s)
    if proc.returncode != 0:
        tail = (proc.stdout + proc.stderr)[-1500:]
        raise ImageBuildError(f"docker build failed for {tag}:\n{tail}")
    return tag


class DockerTestExecutor:
    """Runs pytest for one workspace inside the task image and returns per-test outcomes."""

    def __init__(self, image: str, runner: GradingRunner, test_cmd: str) -> None:
        self._image = image
        self._runner = runner
        self._test_cmd = shlex.split(test_cmd)

    def run(self, *, workspace: Path, pytest_args: Sequence[str], timeout_s: float) -> RunResult:
        report = workspace / REPORT_FILE
        report.unlink(missing_ok=True)
        argv = [
            *self._test_cmd,
            "-p", "cascade_pytest_report",
            "--continue-on-collection-errors",
            "-p", "no:cacheprovider",
            "-q", "--tb=no",
            *pytest_args,
        ]  # fmt: skip
        spec = SandboxSpec(
            image=self._image,
            workspace=workspace,
            timeout_s=timeout_s,
            env={
                "PYTHONPATH": CONTAINER_PLUGIN_DIR,
                "CASCADE_REPORT_PATH": f"/workspace/{REPORT_FILE}",
            },
        )
        result = self._runner.run(spec, argv)
        tail = (result.stdout + result.stderr)[-1500:]
        outcomes: dict[str, str] = {}
        if not result.timed_out and report.is_file():
            outcomes = parse_report(report.read_text(encoding="utf-8"))
        report.unlink(missing_ok=True)
        return RunResult(
            exit_code=result.exit_code,
            duration_s=result.duration_s,
            timed_out=result.timed_out,
            outcomes=outcomes,
            stdout_tail=tail,
        )


def remove_tree(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)
