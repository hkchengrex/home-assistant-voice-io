"""Continuous microphone capture with lightweight voice activity detection."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
import queue
import threading
import time
from typing import Any

import numpy as np

from .audio import trim_spoken_phrase


@dataclass(frozen=True)
class SegmentedUtterance:
    samples: np.ndarray
    hit_duration_limit: bool


class VoiceSegmenter:
    """Split an audio stream into short utterances separated by quiet pauses."""

    def __init__(
        self,
        sample_rate: int = 16_000,
        *,
        block_ms: int = 20,
        start_ms: int = 80,
        end_silence_ms: int = 300,
        pre_roll_ms: int = 700,
        max_utterance_seconds: float = 2.5,
        min_rms: float = 0.004,
        noise_multiplier: float = 3.0,
        calibration_ms: int = 0,
    ) -> None:
        self.sample_rate = sample_rate
        self.block_samples = round(sample_rate * block_ms / 1000)
        self.start_samples = round(sample_rate * start_ms / 1000)
        self.end_silence_samples = round(sample_rate * end_silence_ms / 1000)
        self.max_utterance_samples = round(sample_rate * max_utterance_seconds)
        self.min_rms = min_rms
        self.noise_multiplier = noise_multiplier
        self.noise_rms = 0.001
        self._calibration_samples_remaining = round(
            sample_rate * calibration_ms / 1000
        )
        self.active = False
        self.level_rms = 0.0
        self._voiced_run = 0
        self._silence_run = 0
        self._pre_roll: deque[np.ndarray] = deque(
            maxlen=max(1, round(pre_roll_ms / block_ms))
        )
        self._utterance: list[np.ndarray] = []

    def process(self, samples: np.ndarray) -> list[SegmentedUtterance]:
        block = np.asarray(samples, dtype=np.float32).reshape(-1).copy()
        if block.size == 0:
            return []
        self.level_rms = float(np.sqrt(np.mean(block * block) + 1e-12))
        if not self.active and self._calibration_samples_remaining > 0:
            self.noise_rms = 0.9 * self.noise_rms + 0.1 * self.level_rms
            self._calibration_samples_remaining -= block.size
            self._pre_roll.append(block)
            return []
        threshold = max(self.min_rms, self.noise_rms * self.noise_multiplier)
        voiced = self.level_rms >= threshold
        completed: list[SegmentedUtterance] = []

        if not self.active:
            self._pre_roll.append(block)
            if voiced:
                self._voiced_run += block.size
            else:
                self.noise_rms = 0.98 * self.noise_rms + 0.02 * self.level_rms
                self._voiced_run = 0
            if self._voiced_run >= self.start_samples:
                self.active = True
                self._utterance = list(self._pre_roll)
                self._silence_run = 0
            return completed

        self._utterance.append(block)
        if voiced:
            self._silence_run = 0
        else:
            self._silence_run += block.size

        utterance_size = sum(part.size for part in self._utterance)
        hit_duration_limit = utterance_size >= self.max_utterance_samples
        if self._silence_run >= self.end_silence_samples or hit_duration_limit:
            joined = np.concatenate(self._utterance)
            trimmed = trim_spoken_phrase(joined, self.sample_rate)
            if trimmed.size >= self.sample_rate // 5:
                completed.append(
                    SegmentedUtterance(
                        samples=trimmed,
                        hit_duration_limit=hit_duration_limit,
                    )
                )
            self._reset_after_utterance()
        return completed

    def _reset_after_utterance(self) -> None:
        self.active = False
        self._voiced_run = 0
        self._silence_run = 0
        self._pre_roll.clear()
        self._utterance = []


class ContinuousListener:
    """Own a sounddevice stream and expose thread-safe recognition state."""

    def __init__(
        self,
        sample_rate: int = 16_000,
        *,
        heartbeat_timeout_seconds: float | None = 3.0,
        min_rms: float = 0.004,
        noise_multiplier: float = 3.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.heartbeat_timeout_seconds = heartbeat_timeout_seconds
        self.min_rms = min_rms
        self.noise_multiplier = noise_multiplier
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._matcher: Callable[[np.ndarray], dict[str, Any]] | None = None
        self._phase_provider: Callable[[], str] | None = None
        self._device: int | str | None = None
        self._start_phrase_feedback: Callable[[], None] | None = None
        self._command_feedback: Callable[[str], None] | None = None
        self._capture_callback: (
            Callable[[np.ndarray, dict[str, Any]], str | None] | None
        ) = None
        self._external_feedback: queue.Queue[Callable[[], object]] = queue.Queue()
        self._last_heartbeat = 0.0
        self._state: dict[str, Any] = {
            "running": False,
            "phase": "stopped",
            "level_db": -60.0,
            "event_id": 0,
            "last_result": None,
            "error": None,
        }

    def start(
        self,
        *,
        device: int | str | None,
        matcher: Callable[[np.ndarray], dict[str, Any]],
        phase_provider: Callable[[], str] | None = None,
        start_phrase_feedback: Callable[[], None] | None = None,
        command_feedback: Callable[[str], None] | None = None,
        capture_callback: (
            Callable[[np.ndarray, dict[str, Any]], str | None] | None
        ) = None,
    ) -> None:
        with self._lock:
            if self._state["running"] or (
                self._thread is not None and self._thread.is_alive()
            ):
                raise RuntimeError("Continuous listening is already running")
            self._state.update(
                running=True,
                phase="starting",
                level_db=-60.0,
                last_result=None,
                error=None,
            )
            self._device = device
            self._matcher = matcher
            self._phase_provider = phase_provider
            self._start_phrase_feedback = start_phrase_feedback
            self._command_feedback = command_feedback
            self._capture_callback = capture_callback
            self._last_heartbeat = time.monotonic()
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run,
                name="continuous-voice-listener",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread and thread is not threading.current_thread():
            thread.join(timeout=10.0)
        with self._lock:
            self._state.update(running=False, phase="stopped", level_db=-60.0)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._state)

    def enqueue_feedback(self, feedback: Callable[[], object]) -> None:
        """Queue playback on the listener thread so captured speaker audio is discarded."""
        with self._lock:
            if not self._state["running"]:
                raise RuntimeError("Voice listener is not running")
        self._external_feedback.put_nowait(feedback)

    def heartbeat(self) -> None:
        with self._lock:
            if self._state["running"]:
                self._last_heartbeat = time.monotonic()

    def _idle_phase(self) -> str:
        provider = self._phase_provider
        return provider() if provider else "listening"

    def _new_segmenter(self, *, calibration_ms: int = 0) -> VoiceSegmenter:
        return VoiceSegmenter(
            self.sample_rate,
            min_rms=self.min_rms,
            noise_multiplier=self.noise_multiplier,
            calibration_ms=calibration_ms,
        )

    def _run(self) -> None:
        try:
            import sounddevice as sd

            audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=500)
            segmenter = self._new_segmenter(calibration_ms=1000)

            def callback(
                input_data: np.ndarray,
                frames: int,
                timing: Any,
                status: Any,
            ) -> None:
                del frames, timing
                if status:
                    with self._lock:
                        self._state["error"] = str(status)
                try:
                    audio_queue.put_nowait(input_data[:, 0].copy())
                except queue.Full:
                    pass

            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=segmenter.block_samples,
                device=self._device,
                callback=callback,
            ):
                with self._lock:
                    self._state["phase"] = self._idle_phase()
                while not self._stop.is_set():
                    try:
                        external_feedback = self._external_feedback.get_nowait()
                    except queue.Empty:
                        external_feedback = None
                    if external_feedback is not None:
                        with self._lock:
                            self._state["phase"] = "responding"
                        try:
                            external_feedback()
                        except Exception as exc:
                            with self._lock:
                                self._state["error"] = f"Response failed: {exc}"
                        while True:
                            try:
                                audio_queue.get_nowait()
                            except queue.Empty:
                                break
                        replacement = self._new_segmenter()
                        replacement.noise_rms = segmenter.noise_rms
                        segmenter = replacement
                        with self._lock:
                            self._state["phase"] = self._idle_phase()
                        continue
                    with self._lock:
                        heartbeat_expired = (
                            self.heartbeat_timeout_seconds is not None
                            and time.monotonic() - self._last_heartbeat
                            > self.heartbeat_timeout_seconds
                        )
                    if heartbeat_expired:
                        break
                    try:
                        block = audio_queue.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    utterances = segmenter.process(block)
                    level_db = (
                        20.0 * np.log10(segmenter.level_rms)
                        if segmenter.level_rms > 0
                        else -60.0
                    )
                    with self._lock:
                        self._state["level_db"] = max(-60.0, float(level_db))
                        self._state["phase"] = (
                            "hearing" if segmenter.active else self._idle_phase()
                        )
                    for captured in utterances:
                        utterance = captured.samples
                        with self._lock:
                            self._state["phase"] = "matching"
                        matcher = self._matcher
                        if matcher is None:
                            continue
                        if (
                            captured.hit_duration_limit
                            and self._idle_phase() == "waiting_for_start"
                        ):
                            result = {
                                "kind": "ignored",
                                "accepted": False,
                                "command": None,
                                "utterance": None,
                                "best_command": None,
                                "best_utterance": None,
                                "score": 0.0,
                                "margin": 0.0,
                                "scores": {},
                                "rejection_reason": "duration_limit",
                                "match_seconds": 0.0,
                                "audio_seconds": utterance.size / self.sample_rate,
                            }
                        else:
                            try:
                                match_started = time.perf_counter()
                                result = matcher(utterance)
                                result["match_seconds"] = (
                                    time.perf_counter() - match_started
                                )
                                result["audio_seconds"] = (
                                    utterance.size / self.sample_rate
                                )
                            except Exception as exc:
                                with self._lock:
                                    self._state["error"] = str(exc)
                                    self._state["phase"] = self._idle_phase()
                                continue
                        feedback_played = False
                        capture_callback = self._capture_callback
                        if capture_callback is not None:
                            try:
                                diagnostic_id = capture_callback(utterance, result)
                                if diagnostic_id is not None:
                                    result["diagnostic_id"] = diagnostic_id
                            except Exception as exc:
                                with self._lock:
                                    self._state["error"] = (
                                        f"Diagnostic capture failed: {exc}"
                                    )
                        if result.get("kind") == "start_phrase":
                            feedback = self._start_phrase_feedback
                            if feedback is not None:
                                feedback_played = True
                                try:
                                    feedback()
                                except Exception as exc:
                                    with self._lock:
                                        self._state["error"] = f"Response failed: {exc}"
                        elif (
                            result.get("kind") == "command"
                            and result.get("accepted")
                            and not result.get("suppress_feedback")
                        ):
                            feedback = self._command_feedback
                            command = result.get("command")
                            if feedback is not None and isinstance(command, str):
                                feedback_played = True
                                try:
                                    feedback(command)
                                except Exception as exc:
                                    with self._lock:
                                        self._state["error"] = f"Response failed: {exc}"
                        if feedback_played:
                            while True:
                                try:
                                    audio_queue.get_nowait()
                                except queue.Empty:
                                    break
                            replacement = self._new_segmenter()
                            replacement.noise_rms = segmenter.noise_rms
                            segmenter = replacement
                        with self._lock:
                            self._state["event_id"] += 1
                            self._state["last_result"] = result
                            self._state["phase"] = self._idle_phase()
        except Exception as exc:  # sounddevice errors vary by host API
            with self._lock:
                self._state["error"] = str(exc)
        finally:
            with self._lock:
                self._state.update(running=False, phase="stopped", level_db=-60.0)
