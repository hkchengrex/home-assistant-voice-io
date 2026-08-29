"""Optional listener lifecycle integrations."""

from __future__ import annotations

from dataclasses import dataclass
import re
import shutil
import subprocess
from typing import Protocol


@dataclass(frozen=True)
class ListenerUpdate:
    updated: bool = False
    paused: bool = False
    waiting_for_templates: bool = False
    error: str | None = None


class ListenerManager(Protocol):
    def update(self, *, has_templates: bool) -> ListenerUpdate: ...
    def pause(self) -> ListenerUpdate: ...


class NoopListenerManager:
    def update(self, *, has_templates: bool) -> ListenerUpdate:
        return ListenerUpdate(waiting_for_templates=not has_templates)

    def pause(self) -> ListenerUpdate:
        return ListenerUpdate()


class SystemdUserServiceManager:
    """Linux systemd user-service adapter; never selected implicitly."""

    def __init__(self, service: str) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.@-]+", service):
            raise ValueError("Invalid managed service name")
        if shutil.which("systemctl") is None:
            raise RuntimeError("systemctl is not available on this platform")
        self.service = service

    def _run(self, action: str) -> str | None:
        try:
            subprocess.run(
                ["systemctl", "--user", action, self.service],
                check=True,
                capture_output=True,
                text=True,
                timeout=20,
            )
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            return (getattr(exc, "stderr", None) or str(exc)).strip()
        return None

    def update(self, *, has_templates: bool) -> ListenerUpdate:
        error = self._run("restart" if has_templates else "stop")
        return ListenerUpdate(
            updated=has_templates and error is None,
            waiting_for_templates=not has_templates,
            error=error,
        )

    def pause(self) -> ListenerUpdate:
        error = self._run("stop")
        return ListenerUpdate(paused=error is None, error=error)
