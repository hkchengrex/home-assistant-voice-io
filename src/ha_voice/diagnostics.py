"""Bounded diagnostic captures for reviewing false voice triggers."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import threading
from typing import Any

import numpy as np

from .audio import Audio, save_wav


DEFAULT_TRIGGER_CAPTURE_LIMIT = 20
_EVENT_ID = re.compile(r"\d{8}T\d{12}Z")
_RESULT_FIELDS = (
    "accepted",
    "command",
    "utterance",
    "best_command",
    "best_utterance",
    "runner_command",
    "runner_score",
    "score",
    "margin",
    "scores",
    "rejection_reason",
    "audio_seconds",
    "match_seconds",
    "action_status",
)


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _result_metadata(result: dict[str, Any], filename: str) -> dict[str, Any]:
    metadata = {
        field: _json_value(result[field])
        for field in _RESULT_FIELDS
        if field in result
    }
    metadata["audio_file"] = filename
    return metadata


class TriggerCaptureQueue:
    """Save accepted Start events and their next command in a bounded FIFO."""

    def __init__(
        self,
        recordings_dir: Path,
        *,
        sample_rate: int,
        max_events: int = DEFAULT_TRIGGER_CAPTURE_LIMIT,
    ) -> None:
        if max_events < 1:
            raise ValueError("max_events must be at least 1")
        self.recordings_dir = recordings_dir
        self.root = recordings_dir / "_diagnostics" / "trigger_candidates"
        self.sample_rate = sample_rate
        self.max_events = max_events
        self._lock = threading.Lock()
        self._pending_event_id: str | None = None
        self.root.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._prune_locked()

    def _event_dir(self, event_id: str) -> Path:
        if not _EVENT_ID.fullmatch(event_id):
            raise ValueError("Invalid diagnostic event")
        return self.root / event_id

    def _event_dirs_locked(self) -> list[Path]:
        return sorted(
            path
            for path in self.root.iterdir()
            if path.is_dir() and _EVENT_ID.fullmatch(path.name)
        )

    def _prune_locked(self) -> None:
        events = self._event_dirs_locked()
        for stale in events[: max(0, len(events) - self.max_events)]:
            shutil.rmtree(stale)

    @staticmethod
    def _read_metadata(event_dir: Path) -> dict[str, Any]:
        try:
            payload = json.loads((event_dir / "metadata.json").read_text("utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("Diagnostic metadata is unavailable") from exc
        if not isinstance(payload, dict):
            raise ValueError("Diagnostic metadata is invalid")
        return payload

    @staticmethod
    def _write_json(path: Path, metadata: dict[str, Any]) -> None:
        temporary = path.with_suffix(".tmp.json")
        temporary.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    @classmethod
    def _write_metadata(cls, event_dir: Path, metadata: dict[str, Any]) -> None:
        cls._write_json(event_dir / "metadata.json", metadata)

    def _new_event_id_locked(self) -> str:
        now = datetime.now(timezone.utc)
        while True:
            event_id = now.strftime("%Y%m%dT%H%M%S%fZ")
            if not (self.root / event_id).exists():
                return event_id
            now = datetime.now(timezone.utc)

    def capture(self, samples: np.ndarray, result: dict[str, Any]) -> str | None:
        """Capture an accepted Start or attach the following command."""
        kind = result.get("kind")
        if kind == "start_phrase" and result.get("accepted"):
            with self._lock:
                event_id = self._new_event_id_locked()
                event_dir = self.root / event_id
                event_dir.mkdir(parents=True)
                save_wav(
                    event_dir / "start.wav",
                    Audio(np.asarray(samples, dtype=np.float32), self.sample_rate),
                )
                metadata = {
                    "id": event_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "start": _result_metadata(result, "start.wav"),
                    "command": None,
                }
                self._write_metadata(event_dir, metadata)
                self._pending_event_id = event_id
                self._prune_locked()
                return event_id

        if kind == "command":
            with self._lock:
                event_id = self._pending_event_id
                self._pending_event_id = None
                if event_id is None:
                    return None
                event_dir = self._event_dir(event_id)
                if not event_dir.is_dir():
                    return None
                save_wav(
                    event_dir / "command.wav",
                    Audio(np.asarray(samples, dtype=np.float32), self.sample_rate),
                )
                metadata = self._read_metadata(event_dir)
                metadata["command"] = _result_metadata(result, "command.wav")
                self._write_metadata(event_dir, metadata)
                return event_id
        return None

    def list_events(self) -> list[dict[str, Any]]:
        with self._lock:
            events: list[dict[str, Any]] = []
            for event_dir in reversed(self._event_dirs_locked()):
                try:
                    metadata = self._read_metadata(event_dir)
                except ValueError:
                    continue
                event_id = event_dir.name
                start = metadata.get("start")
                command = metadata.get("command")
                if isinstance(start, dict) and (event_dir / "start.wav").is_file():
                    start["audio_url"] = f"/api/diagnostics/{event_id}/start.wav"
                if isinstance(command, dict) and (event_dir / "command.wav").is_file():
                    command["audio_url"] = (
                        f"/api/diagnostics/{event_id}/command.wav"
                    )
                events.append(metadata)
            return events

    def audio_path(self, event_id: str, filename: str) -> Path:
        if filename not in {"start.wav", "command.wav"}:
            raise ValueError("Invalid diagnostic audio")
        path = self._event_dir(event_id) / filename
        if not path.is_file():
            raise FileNotFoundError("Diagnostic audio not found")
        return path

    def promote_false_trigger(self, event_id: str) -> dict[str, Any]:
        """Move a reviewed event into active negative-template folders."""
        with self._lock:
            event_dir = self._event_dir(event_id)
            if not event_dir.is_dir():
                raise FileNotFoundError("Diagnostic event not found")
            metadata = self._read_metadata(event_dir)
            promoted: list[str] = []
            destinations = (
                ("start.wav", "_not_start_phrase", f"{event_id}_start.wav"),
                ("command.wav", "_not_command", f"{event_id}_command.wav"),
            )
            for source_name, folder, destination_name in destinations:
                source = event_dir / source_name
                if not source.is_file():
                    continue
                destination = self.recordings_dir / folder / destination_name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                promoted.append(folder)
            metadata_dir = self.recordings_dir / "_negative_metadata"
            metadata_dir.mkdir(parents=True, exist_ok=True)
            self._write_json(
                metadata_dir / f"{event_id}.json",
                {**metadata, "promoted": promoted},
            )
            shutil.rmtree(event_dir)
            if self._pending_event_id == event_id:
                self._pending_event_id = None
            return {"event_id": event_id, "promoted": promoted}

    def dismiss(self, event_id: str) -> None:
        with self._lock:
            event_dir = self._event_dir(event_id)
            if not event_dir.is_dir():
                raise FileNotFoundError("Diagnostic event not found")
            shutil.rmtree(event_dir)
            if self._pending_event_id == event_id:
                self._pending_event_id = None
