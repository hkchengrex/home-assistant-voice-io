"""Bounded diagnostic captures for reviewing false voice triggers."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import hashlib
import math
from pathlib import Path
import re
import shutil
import threading
import time
from typing import Any

import numpy as np

from .audio import Audio, encode_wav, load_wav, save_wav, trim_silence, trim_spoken_phrase
from .config import AppConfig


DEFAULT_TRIGGER_CAPTURE_LIMIT = 20
_EVENT_ID = re.compile(r"\d{8}T\d{12}Z")
_RESULT_FIELDS = (
    "kind",
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
    "min_audio_seconds",
    "max_audio_seconds",
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


def _result_metadata(result: dict[str, Any], filename: str, config: AppConfig | None = None) -> dict[str, Any]:
    metadata = {
        field: _json_value(result[field])
        for field in _RESULT_FIELDS
        if field in result
    }
    metadata["audio_file"] = filename
    if config is not None:
        settings = config.recognizer if result.get("kind") == "command" else config.start_phrase
        metadata.update(max_distance=settings.max_distance, min_margin=settings.min_margin)
        if result.get("accepted") is False and not result.get("rejection_reason"):
            score, margin = metadata.get("score"), metadata.get("margin")
            if result.get("kind") == "ignored" and result.get("best_command") not in {None, config.start_phrase.name}:
                metadata["rejection_reason"] = "not_start_phrase"
            elif isinstance(score, (int, float)) and isinstance(margin, (int, float)) and math.isfinite(score) and math.isfinite(margin):
                distance_failed = score > settings.max_distance
                margin_failed = margin < settings.min_margin
                metadata["rejection_reason"] = (
                    "distance_and_margin" if distance_failed and margin_failed else
                    "distance" if distance_failed else "margin" if margin_failed else None)
    return metadata


class TriggerCaptureQueue:
    """Save accepted Start events and their next command in a bounded FIFO."""

    def __init__(
        self,
        recordings_dir: Path,
        *,
        sample_rate: int,
        max_events: int = DEFAULT_TRIGGER_CAPTURE_LIMIT,
        config: AppConfig | None = None,
    ) -> None:
        if max_events < 1:
            raise ValueError("max_events must be at least 1")
        self.recordings_dir = recordings_dir
        self.root = recordings_dir / "_diagnostics" / "trigger_candidates"
        self.sample_rate = sample_rate
        self.max_events = max_events
        self.config = config
        self._lock = threading.Lock()
        self._pending_event_id: str | None = None
        self.root.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._prune_locked()

    @property
    def command_timeout_seconds(self) -> float:
        return self.config.start_phrase.command_timeout_seconds if self.config else 5.0

    def arm_command_window(self) -> None:
        """Refresh diagnostic timing after acknowledgement playback, like the gate."""
        with self._lock:
            if self._pending_event_id is None:
                return
            event_dir = self._event_dir(self._pending_event_id)
            if not event_dir.is_dir():
                return
            metadata = self._read_metadata(event_dir)
            metadata["command_deadline_at"] = time.time() + self.command_timeout_seconds
            self._write_metadata(event_dir, metadata)

    def _close_pending_locked(self) -> None:
        if self._pending_event_id is not None:
            event_dir = self._event_dir(self._pending_event_id)
            if event_dir.is_dir():
                metadata = self._read_metadata(event_dir)
                if metadata.get("command") is None:
                    metadata["command_window_closed"] = True
                    self._write_metadata(event_dir, metadata)
            self._pending_event_id = None

    def _command_status(self, metadata: dict[str, Any]) -> str | None:
        command = metadata.get("command")
        if isinstance(command, dict):
            return "accepted" if command.get("accepted") else "rejected"
        if not metadata.get("start"):
            return None
        try:
            deadline = metadata.get("command_deadline_at")
            if deadline is None:
                # Old entries predate explicit timing; use their capture timestamp.
                deadline = datetime.fromisoformat(metadata["created_at"]).timestamp() + self.command_timeout_seconds
            waiting = not metadata.get("command_window_closed") and time.time() < float(deadline)
        except (ValueError, TypeError, KeyError, OverflowError):
            waiting = False
        return "waiting" if waiting else "nothing_detected"

    def review_status(self) -> dict[str, Any]:
        """Read the short-lived opt-in shared with the separate live listener."""
        try:
            setting = json.loads((self.root.parent / "review-mode.json").read_text("utf-8"))
            started, expires = float(setting["started_at"]), float(setting["expires_at"])
            now = time.time()
            active = math.isfinite(started) and math.isfinite(expires) and started <= now < expires <= started + 300
        except (OSError, ValueError, TypeError, KeyError):
            active, expires = False, 0
        return {"active": active, "remaining_seconds": max(0, math.ceil(expires - time.time())) if active else 0}

    def set_review_mode(self, enabled: bool) -> dict[str, Any]:
        if not isinstance(enabled, bool):
            raise ValueError("Review mode must be enabled or disabled")
        with self._lock:
            now = time.time()
            self._write_json(self.root.parent / "review-mode.json",
                             {"started_at": now, "expires_at": now + 300 if enabled else now})
        return self.review_status()

    def _review_path(self, event_id: str, clip: str) -> Path:
        self._event_dir(event_id)
        if clip not in {"start", "command", "attempt"}:
            raise ValueError("Choose a saved audio clip")
        return self.recordings_dir / "_review_metadata" / f"{event_id}_{clip}.json"

    def teach(self, event_id: str, clip: str, label: str, config: AppConfig) -> dict[str, Any]:
        """Teach exactly one clip; retain its sibling and make retries idempotent."""
        allowed = {*config.commands, config.calibration.name}
        if config.start_phrase.enabled:
            allowed.update({config.start_phrase.name, "_not_start_phrase"})
        if not isinstance(label, str) or label not in allowed:
            raise ValueError("Choose a configured command, wake phrase, or rejection label")
        review_path = self._review_path(event_id, clip)
        with self._lock:
            if review_path.exists():
                previous = json.loads(review_path.read_text("utf-8"))
                if previous["label"] != label:
                    raise ValueError("This clip was already taught; remove an incorrect saved training take and record a replacement")
                destination = self.recordings_dir / label / f"{event_id}_{clip}.wav"
                if not destination.is_file() or hashlib.sha256(destination.read_bytes()).hexdigest() != previous["sha256"]:
                    raise ValueError("The saved training take was removed or changed; record a new example")
                return {"label": label, "filename": destination.name, "already_taught": True}
            source = self.audio_path(event_id, f"{clip}.wav")
            metadata = self._read_metadata(source.parent)
            audio = load_wav(source, config.recognizer.sample_rate)
            trim = trim_spoken_phrase if label == config.start_phrase.name else trim_silence
            samples = trim(audio.samples, audio.sample_rate)
            if not .2 <= samples.size / audio.sample_rate <= 6 or not np.isfinite(samples).all():
                raise ValueError("Use a clear recording of 0.2–6 seconds; record a new example if needed")
            body = encode_wav(Audio(samples, audio.sample_rate))
            destination = self.recordings_dir / label / f"{event_id}_{clip}.wav"
            destination.parent.mkdir(parents=True, exist_ok=True)
            review_path.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise ValueError("A training take with this name already exists")
            created = False
            try:
                with destination.open("xb") as output:
                    created = True
                    output.write(body)
                self._write_json(review_path, {"label": label, "clip": clip, "event": metadata,
                    "filename": destination.name, "sha256": hashlib.sha256(body).hexdigest(),
                    "taught_at": datetime.now(timezone.utc).isoformat()})
            except Exception:
                if created:
                    destination.unlink(missing_ok=True)
                raise
            return {"label": label, "filename": destination.name, "already_taught": False}

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
        if kind == "ignored" or (kind == "command" and self._pending_event_id is None):
            with self._lock:
                # An ignored wake attempt starts a new interaction, not the old pair.
                self._close_pending_locked()
                if not self.review_status()["active"]:
                    return None
                event_id = self._new_event_id_locked()
                event_dir = self.root / event_id
                event_dir.mkdir(parents=True)
                save_wav(event_dir / "attempt.wav", Audio(np.asarray(samples, dtype=np.float32), self.sample_rate))
                self._write_metadata(event_dir, {"id": event_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "start": None, "command": None,
                    "attempt": _result_metadata(result, "attempt.wav", self.config)})
                self._prune_locked()
                return event_id
        if kind == "start_phrase" and result.get("accepted"):
            with self._lock:
                self._close_pending_locked()
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
                    "start": _result_metadata(result, "start.wav", self.config),
                    "command": None,
                    "command_deadline_at": time.time() + self.command_timeout_seconds,
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
                metadata["command"] = _result_metadata(result, "command.wav", self.config)
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
                metadata["command_status"] = self._command_status(metadata)
                if isinstance(start, dict) and (event_dir / "start.wav").is_file():
                    start["audio_url"] = f"/api/diagnostics/{event_id}/start.wav"
                if isinstance(command, dict) and (event_dir / "command.wav").is_file():
                    command["audio_url"] = (
                        f"/api/diagnostics/{event_id}/command.wav"
                    )
                events.append(metadata)
                attempt = metadata.get("attempt")
                if isinstance(attempt, dict) and (event_dir / "attempt.wav").is_file():
                    attempt["audio_url"] = f"/api/diagnostics/{event_id}/attempt.wav"
                for clip in ("start", "command", "attempt"):
                    if isinstance(metadata.get(clip), dict):
                        try:
                            metadata[clip]["taught_as"] = json.loads(self._review_path(event_id, clip).read_text("utf-8"))["label"]
                        except (OSError, ValueError, KeyError, TypeError):
                            pass
            return events

    def audio_path(self, event_id: str, filename: str) -> Path:
        if filename not in {"start.wav", "command.wav", "attempt.wav"}:
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
            if metadata.get("attempt") or any(self._review_path(event_id, clip).exists() for clip in ("start", "command")):
                raise ValueError("Label each clip separately with Teach as")
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
