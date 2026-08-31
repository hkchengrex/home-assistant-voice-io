"""Home Assistant Voice IO: voice recognition, training, and response generation."""

from importlib.metadata import PackageNotFoundError, version

from .actions import ActionResult, CommandHandler
from .config import AppConfig, CommandConfig, RecognizerConfig, load_config
from .matcher import MatchResult, Template, classify, load_templates
from .voice_generation import (
    BatchCloneResult,
    CloneResult,
    CloneSettings,
    GeneratedBatch,
    HuggingFaceSpaceVoiceCloner,
    VoiceGenerationError,
)

try:
    __version__ = version("voice-io")
except PackageNotFoundError:  # Source checkout without an installed distribution.
    __version__ = "0.1.0"

__all__ = [
    "ActionResult",
    "BatchCloneResult",
    "AppConfig",
    "CommandConfig",
    "CommandHandler",
    "CloneResult",
    "CloneSettings",
    "GeneratedBatch",
    "HuggingFaceSpaceVoiceCloner",
    "MatchResult",
    "RecognizerConfig",
    "Template",
    "VoiceGenerationError",
    "classify",
    "load_config",
    "load_templates",
]
