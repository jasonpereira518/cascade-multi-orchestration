"""Grading runner: executes TRUSTED commands (task validation, hidden-test grading).

Unlike SandboxRunner this one does not run the leak guard, because grading workspaces legitimately
contain the test patch and the gold patch. It must never be handed to an agent. It shares every
isolation flag with the agent-side runner through ContainerRunner (network off, capabilities
dropped, read-only root, resource limits, no host environment), so the only difference between the
two is what they are willing to mount.
"""

from __future__ import annotations

from cascade.harness.sandbox import ContainerRunner


class GradingRunner(ContainerRunner):
    """Mount policy: the workspace must live inside ``sandbox_root``; nothing else is checked."""
