"""Local voice enable/pause state, independent of external command execution."""
from __future__ import annotations

import json
from pathlib import Path
import threading

from .actions import ActionResult
from .chime import play_beeps


class VoiceMode:
    def __init__(self, listener, responses, *, output_device=None, state_file: Path | None = None):
        self.listener = listener
        self.responses = responses
        self.output_device = output_device
        self.state_file = state_file
        self._lock = threading.RLock()
        self._recognition = threading.local()
        self._playing: threading.Event | None = None
        self._enabled = True
        self.reset_recognition = lambda: None
        self._recognition_epoch = -1
        if state_file is not None and state_file.exists():
            data = json.loads(state_file.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or type(data.get("enabled")) is not bool:
                raise ValueError("Voice state must contain a boolean enabled value")
            self._enabled = data["enabled"]
        listener.set_input_paused(not self._enabled)

    def status(self):
        with self._lock:
            state = self.listener.snapshot()
            return {"enabled": self._enabled, "input_paused": not self._enabled,
                    "spoken_responses": self._enabled, "listener_phase": state["phase"],
                    "capture_paused": state.get("capture_paused", False)}

    def recognize(self, matcher, samples):
        epoch = self.listener.recognition_epoch
        self._recognition.epoch = epoch
        if not self.listener.input_is_current(epoch):
            return {"kind": "ignored", "accepted": False}
        if epoch != self._recognition_epoch:
            self.reset_recognition()
            self._recognition_epoch = epoch
        return matcher(samples)

    def execute(self, handler, command):
        epoch = getattr(self._recognition, "epoch", -1)
        if not self.listener.input_is_current(epoch):
            return ActionResult(status="unassigned", suppress_feedback=True)
        return handler(command)

    def play_spoken(self, group):
        """Recognition feedback must not become a beep after a concurrent pause."""
        cancelled = threading.Event()
        with self._lock:
            epoch = getattr(self._recognition, "epoch", -1)
            if not self._enabled or not self.listener.input_is_current(epoch):
                return False
            self._playing = cancelled
        try:
            if self.responses is not None and group:
                self.responses.play(group, cancelled=cancelled)
            return not cancelled.is_set() and self.listener.input_is_current(epoch)
        finally:
            with self._lock:
                if self._playing is cancelled:
                    self._playing = None

    def _queue(self, *, group=None, beeps=None):
        cancelled = threading.Event()
        if self._playing is not None:
            self._playing.set()

        def playback():
            with self._lock:
                if cancelled.is_set():
                    return
                self._playing = cancelled
                count = beeps if beeps is not None else (None if self._enabled else 1)
            try:
                if count is not None:
                    play_beeps(count, device=self.output_device, cancelled=cancelled)
                elif self.responses is not None:
                    self.responses.play(group, cancelled=cancelled)
            finally:
                with self._lock:
                    if self._playing is cancelled:
                        self._playing = None

        self.listener.enqueue_feedback(playback, replace_pending=True, cancel=cancelled.set)

    def queue_response(self, group):
        with self._lock:
            self._queue(group=group)

    def set_enabled(self, enabled: bool | None = None):
        with self._lock:
            if not self.listener.snapshot()["running"]:
                raise RuntimeError("Voice listener is not running")
            enabled = not self._enabled if enabled is None else enabled
            if self.state_file is not None:
                try:
                    self.state_file.parent.mkdir(parents=True, exist_ok=True)
                    temporary = self.state_file.with_suffix(self.state_file.suffix + ".tmp")
                    temporary.write_text(json.dumps({"enabled": enabled}) + "\n", encoding="utf-8")
                    temporary.replace(self.state_file)
                except OSError as exc:
                    raise RuntimeError(f"Cannot save voice state: {exc}") from exc
            self._enabled = enabled
            self.listener.set_input_paused(not enabled)
            self._queue(beeps=1 if enabled else 2)
            return self.status()

    def callbacks(self):
        return {"/voice/toggle": self.set_enabled,
                "/voice/pause": lambda: self.set_enabled(False),
                "/voice/resume": lambda: self.set_enabled(True)}
