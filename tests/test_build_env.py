from __future__ import annotations

import datetime as dt
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from cascade.config import ResolvedRepo, SelectionLimits
from cascade.harness.grading import GradingRunner
from cascade.taskgen import build_env as be
from cascade.taskgen.build_env import (
    DockerTestExecutor,
    ImageBuildError,
    build_image,
    docker_available,
    image_tag,
    render_dockerfile,
)
from cascade.taskgen.results import ERROR, FAILED, PASSED
from tests.taskgen_helpers import Upstream, build_upstream


def repo(**over: str) -> ResolvedRepo:
    base = {
        "name": "o/Repo",
        "license": "MIT",
        "python_version": "3.12.13",
        "install_cmd": "uv pip install --group tests -e .",
        "test_cmd": "python -m pytest",
    }
    return ResolvedRepo(
        **{**base, **over},
        window_start=dt.date(2025, 1, 1),
        window_end=dt.date(2025, 1, 31),
        limits=SelectionLimits(),
    )


def done(
    argv: Sequence[str], rc: int = 0, out: str = "", err: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(list(argv), rc, stdout=out, stderr=err)


# --- Dockerfile and tag ----------------------------------------------------------------------


def test_dockerfile_pins_python_via_uv_and_installs_editable_then_clears_the_source() -> None:
    text = render_dockerfile(repo())
    assert text.startswith(f"FROM {be.BASE_IMAGE}\n")
    assert f"COPY --from=ghcr.io/astral-sh/uv:{be.UV_VERSION} /uv /uvx /bin/" in text
    assert "uv python install 3.12.13 && uv venv --python 3.12.13 /opt/venv" in text
    assert "UV_PYTHON_INSTALL_DIR=/opt/python" in text and "VIRTUAL_ENV=/opt/venv" in text
    lines = text.splitlines()
    install = lines.index("RUN uv pip install --group tests -e .")
    wipe = next(i for i, ln in enumerate(lines) if ln.startswith("RUN rm -rf /workspace"))
    copy_src = lines.index("COPY repo/ /workspace/")
    assert copy_src < install < wipe  # install with the source present, then remove it
    assert (
        f"COPY cascade_pytest_report.py {be.CONTAINER_PLUGIN_DIR}/cascade_pytest_report.py" in text
    )


def test_dockerfile_never_disables_nor_needs_network_flags() -> None:
    # Network is controlled at `docker run` time (tests) by the sandbox, not baked into the image.
    assert "--network" not in render_dockerfile(repo())


def test_image_tag_is_deterministic_and_sensitive_to_every_input() -> None:
    r = repo()
    df = render_dockerfile(r)
    base = image_tag(r, df, "a" * 40, "plugin v1")
    assert base == image_tag(r, df, "a" * 40, "plugin v1")
    assert base.startswith("cascade-env/o__repo:") and len(base.split(":")[1]) == 12
    assert base != image_tag(r, df, "b" * 40, "plugin v1")
    assert base != image_tag(r, df, "a" * 40, "plugin v2")
    r2 = repo(python_version="3.13.1")
    assert base != image_tag(r2, render_dockerfile(r2), "a" * 40, "plugin v1")
    r3 = repo(test_cmd="python -m pytest -x")
    assert base != image_tag(r3, render_dockerfile(r3), "a" * 40, "plugin v1")


# --- docker probing and building -------------------------------------------------------------


def test_docker_available() -> None:
    assert docker_available(lambda argv, t: done(argv, 0)) is True
    assert docker_available(lambda argv, t: done(argv, 1, err="Cannot connect")) is False

    def missing(argv: Sequence[str], timeout: float | None) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("docker")

    assert docker_available(missing) is False


@pytest.fixture
def up(tmp_path: Path) -> Upstream:
    return build_upstream(tmp_path)


def test_build_image_assembles_context_and_builds_once(up: Upstream) -> None:
    seen: dict[str, object] = {}
    calls: list[list[str]] = []

    def fake(argv: Sequence[str], timeout: float | None) -> subprocess.CompletedProcess[str]:
        calls.append(list(argv))
        if argv[1] == "image":
            return done(argv, 1)  # not built yet
        ctx = Path(argv[-1])  # `docker build --tag <tag> <context>`
        seen["files"] = sorted(p.name for p in ctx.iterdir())
        seen["has_src"] = (ctx / "repo/src/tinylib/clamp.py").exists()
        seen["has_git"] = (ctx / "repo/.git").exists()
        seen["dockerfile"] = (ctx / "Dockerfile").read_text()
        return done(argv)

    tag = build_image(repo(), up.path, up.base_sha, run_fn=fake)
    assert seen["files"] == ["Dockerfile", "cascade_pytest_report.py", "repo"]
    assert seen["has_src"] is True and seen["has_git"] is False  # exported tree, no history
    assert "uv python install 3.12.13" in str(seen["dockerfile"])
    build_call = next(c for c in calls if c[1] == "build")
    assert build_call[:4] == ["docker", "build", "--tag", tag]


def test_build_image_reuses_an_existing_image(up: Upstream) -> None:
    calls: list[list[str]] = []

    def fake(argv: Sequence[str], timeout: float | None) -> subprocess.CompletedProcess[str]:
        calls.append(list(argv))
        return done(argv, 0)  # `image inspect` succeeds

    tag = build_image(repo(), up.path, up.base_sha, run_fn=fake)
    assert tag.startswith("cascade-env/") and [c[1] for c in calls] == ["image"]


def test_build_failure_raises_with_the_log_tail(up: Upstream) -> None:
    def fake(argv: Sequence[str], timeout: float | None) -> subprocess.CompletedProcess[str]:
        if argv[1] == "image":
            return done(argv, 1)
        return done(argv, 1, out="step 5/9", err="ERROR: no matching distribution")

    with pytest.raises(ImageBuildError, match="no matching distribution"):
        build_image(repo(), up.path, up.base_sha, run_fn=fake)


# --- executor --------------------------------------------------------------------------------


def fake_container(report_lines: list[str], rc: int = 1, seen: list[list[str]] | None = None):
    """Stands in for `docker run`: writes the plugin report into the mounted workspace."""

    def run(argv: Sequence[str], timeout: float | None) -> subprocess.CompletedProcess[str]:
        argv = list(argv)
        if seen is not None:
            seen.append(argv)
        if argv[1] == "run":
            mount = argv[argv.index("--mount") + 1]
            source = dict(kv.split("=", 1) for kv in mount.split(","))["source"]
            (Path(source) / be.REPORT_FILE).write_text("\n".join(report_lines) + "\n")
        return done(argv, rc, out="1 failed, 1 passed")

    return run


def test_executor_builds_the_pytest_command_and_parses_the_report(tmp_path: Path) -> None:
    work = tmp_path / "work"
    ws = work / "ws"
    ws.mkdir(parents=True)
    seen: list[list[str]] = []
    lines = [
        '{"nodeid": "tests/t.py::a", "when": "call", "outcome": "passed"}',
        '{"nodeid": "tests/t.py::b", "when": "call", "outcome": "failed"}',
        '{"nodeid": "tests/broken.py", "when": "collect", "outcome": "failed"}',
    ]
    runner = GradingRunner(sandbox_root=work, run_fn=fake_container(lines, seen=seen))
    result = DockerTestExecutor("img:1", runner, "python -m pytest").run(
        workspace=ws, pytest_args=["tests/t.py"], timeout_s=42
    )
    assert result.outcomes == {
        "tests/t.py::a": PASSED,
        "tests/t.py::b": FAILED,
        "tests/broken.py": ERROR,
    }
    assert result.exit_code == 1 and not result.timed_out
    assert not (ws / be.REPORT_FILE).exists()  # report removed after being read

    argv = seen[0]
    assert argv[argv.index("--network") + 1] == "none"
    assert "--env" in argv and "PYTHONPATH=/opt/cascade" in argv
    assert f"CASCADE_REPORT_PATH=/workspace/{be.REPORT_FILE}" in argv
    tail = argv[argv.index("img:1") + 1 :]
    assert tail[:3] == ["python", "-m", "pytest"]
    assert tail[3:5] == ["-p", "cascade_pytest_report"]
    assert "--continue-on-collection-errors" in tail and tail[-1] == "tests/t.py"


def test_executor_reports_timeouts_and_kills_the_container(tmp_path: Path) -> None:
    work = tmp_path / "work"
    ws = work / "ws"
    ws.mkdir(parents=True)
    calls: list[list[str]] = []

    def fake(argv: Sequence[str], timeout: float | None) -> subprocess.CompletedProcess[str]:
        calls.append(list(argv))
        if argv[1] == "run":
            raise subprocess.TimeoutExpired(list(argv), timeout or 0)
        return done(argv)

    runner = GradingRunner(sandbox_root=work, run_fn=fake)
    result = DockerTestExecutor("img", runner, "pytest").run(
        workspace=ws, pytest_args=[], timeout_s=5
    )
    assert result.timed_out and result.exit_code is None and result.outcomes == {}
    assert calls[1][:3] == ["docker", "rm", "-f"]


def test_executor_without_a_report_yields_no_outcomes(tmp_path: Path) -> None:
    work = tmp_path / "work"
    ws = work / "ws"
    ws.mkdir(parents=True)
    runner = GradingRunner(sandbox_root=work, run_fn=lambda a, t: done(a, 2, err="usage error"))
    result = DockerTestExecutor("img", runner, "pytest").run(
        workspace=ws, pytest_args=[], timeout_s=5
    )
    assert result.exit_code == 2 and result.outcomes == {}
