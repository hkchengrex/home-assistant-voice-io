"""Automation-neutral command action contracts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal


ActionStatus = Literal["sent", "unassigned", "failed"]


@dataclass(frozen=True)
class ActionResult:
    status: ActionStatus
    error: str | None = None
    suppress_feedback: bool = False


CommandHandler = Callable[[str], ActionResult]


def no_action(command_name: str) -> ActionResult:
    del command_name
    return ActionResult(status="unassigned")
