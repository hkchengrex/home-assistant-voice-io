"""Safe persistent editing for the self-service voice-command library."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import unicodedata

from .config import CommandConfig, load_config


_COMMAND_SECTION = "[commands.{name}]"


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as output:
            output.write(text)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _command_name(utterance: str, existing: set[str]) -> str:
    ascii_phrase = unicodedata.normalize("NFKD", utterance).encode(
        "ascii", "ignore"
    ).decode("ascii")
    base = re.sub(r"[^a-z0-9]+", "_", ascii_phrase.casefold()).strip("_")
    if not base:
        base = "command"
    if base[0].isdigit():
        base = f"command_{base}"
    base = base[:48].rstrip("_")
    candidate = base
    suffix = 2
    while candidate in existing:
        tail = f"_{suffix}"
        candidate = f"{base[: 48 - len(tail)]}{tail}"
        suffix += 1
    return candidate


def add_command(
    config_path: Path,
    *,
    utterance: str,
    description: str = "",
    target_samples: int = 10,
) -> CommandConfig:
    """Append a validated command section and return the loaded command."""
    utterance = " ".join(utterance.split())
    description = " ".join(description.split())
    if not utterance or len(utterance) > 100:
        raise ValueError("Command phrase must be between 1 and 100 characters")
    if len(description) > 240:
        raise ValueError("Description must be 240 characters or fewer")
    if not 1 <= target_samples <= 100:
        raise ValueError("Target recordings must be between 1 and 100")

    config = load_config(config_path)
    name = _command_name(utterance, set(config.commands))
    block = "\n".join(
        (
            _COMMAND_SECTION.format(name=name),
            f"description = {json.dumps(description or 'Custom voice command', ensure_ascii=False)}",
            f"utterance = {json.dumps(utterance, ensure_ascii=False)}",
            f"target_samples = {target_samples}",
            f"intent_group = {json.dumps(name)}",
            'response = ""',
            "",
        )
    )
    current = config_path.read_text(encoding="utf-8")
    updated = current.rstrip() + "\n\n" + block
    try:
        _write_atomic(config_path, updated)
        return load_config(config_path).commands[name]
    except BaseException:
        _write_atomic(config_path, current)
        raise


def _command_block(text: str, name: str) -> re.Match[str]:
    pattern = re.compile(
        rf"(?ms)^\[commands\.{re.escape(name)}\][ \t]*\r?\n.*?(?=^\[|\Z)"
    )
    match = pattern.search(text)
    if match is None:
        raise ValueError("Command configuration was not found")
    return match


def archive_recording(
    recordings_dir: Path,
    *,
    command_name: str,
    recording: Path,
) -> Path:
    """Move one take into the recoverable underscore-prefixed archive."""
    if recording.parent != recordings_dir / command_name or not recording.is_file():
        raise ValueError("Recording was not found")
    target_dir = recordings_dir / "_trash" / _stamp() / command_name
    target_dir.mkdir(parents=True, exist_ok=False)
    target = target_dir / recording.name
    recording.replace(target)
    return target


def archive_command(
    config_path: Path,
    recordings_dir: Path,
    *,
    command_name: str,
) -> Path:
    """Remove a command while archiving its configuration and recordings."""
    config = load_config(config_path)
    if command_name not in config.commands:
        raise ValueError("Unknown command")
    if len(config.commands) <= 1:
        raise ValueError("Keep at least one command in the library")
    text = config_path.read_text(encoding="utf-8")
    match = _command_block(text, command_name)
    block = match.group(0).rstrip() + "\n"
    updated = (text[: match.start()] + text[match.end() :]).rstrip() + "\n"

    archive_dir = recordings_dir / "_trash" / _stamp() / command_name
    archive_dir.mkdir(parents=True, exist_ok=False)
    source = recordings_dir / command_name
    moved = archive_dir / "recordings"
    if source.exists():
        shutil.move(str(source), str(moved))
    try:
        _write_atomic(config_path, updated)
        (archive_dir / "command.toml").write_text(block, encoding="utf-8")
        load_config(config_path)
    except BaseException:
        _write_atomic(config_path, text)
        if moved.exists() and not source.exists():
            shutil.move(str(moved), str(source))
        raise
    return archive_dir
