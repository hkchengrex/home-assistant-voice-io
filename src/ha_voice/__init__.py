"""Offline, template-based voice command recognition and training."""

from importlib.metadata import PackageNotFoundError, version

from .actions import ActionResult, CommandHandler
from .config import AppConfig, CommandConfig, RecognizerConfig, load_config
from .matcher import MatchResult, Template, classify, load_templates

try:
    __version__ = version("local-voice-pipeline")
except PackageNotFoundError:  # Source checkout without an installed distribution.
    __version__ = "0.1.0"

__all__ = [
    "ActionResult",
    "AppConfig",
    "CommandConfig",
    "CommandHandler",
    "MatchResult",
    "RecognizerConfig",
    "Template",
    "classify",
    "load_config",
    "load_templates",
]
