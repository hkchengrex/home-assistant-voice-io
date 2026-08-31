"""Local-only HTTP server for the browser recording studio."""

from __future__ import annotations

from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse

from .audio import (
    Audio,
    decode_wav,
    encode_wav,
    list_devices,
    record_audio,
    save_wav,
    trim_silence,
    trim_spoken_phrase,
)
from .chime import play_confirmation_chime
from .config import AppConfig, load_config
from .continuous import ContinuousListener
from .diagnostics import TriggerCaptureQueue
from .features import extract_command_features
from .library import add_command, archive_command, archive_recording
from .matcher import Template, classify, load_start_phrase_templates, load_templates
from .start_phrase import StartPhraseGate
from .services import ListenerManager, NoopListenerManager
from .response_http import handle_response_request
from .response_studio import ResponseStudio


MAX_UPLOAD_BYTES = 10 * 1024 * 1024
STATIC_DIR = Path(__file__).with_name("studio")
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/responses": ("responses.html", "text/html; charset=utf-8"),
    "/responses.js": ("responses.js", "text/javascript; charset=utf-8"),
    "/responses.css": ("responses.css", "text/css; charset=utf-8"),
}


def recording_names(config: AppConfig) -> set[str]:
    return {
        *config.commands,
        config.calibration.name,
        config.start_phrase.name,
    }


def template_counts(recordings_dir: Path) -> dict[str, int]:
    if not recordings_dir.exists():
        return {}
    return {
        directory.name: sum(1 for _ in directory.glob("*.wav"))
        for directory in recordings_dir.iterdir()
        if directory.is_dir()
    }


def has_command_recordings(config: AppConfig, recordings_dir: Path) -> bool:
    """Return whether the matcher has at least one actionable command template."""
    return any(
        any((recordings_dir / command_name).glob("*.wav"))
        for command_name in config.commands
    )


def recording_files(
    command_name: str, config: AppConfig, recordings_dir: Path
) -> list[Path]:
    allowed_names = recording_names(config)
    if command_name not in allowed_names:
        raise ValueError("Unknown command")
    return sorted((recordings_dir / command_name).glob("*.wav"))


def recording_file(
    command_name: str,
    filename: str,
    config: AppConfig,
    recordings_dir: Path,
) -> Path:
    if Path(filename).name != filename or not filename.lower().endswith(".wav"):
        raise ValueError("Invalid recording name")
    candidates = recording_files(command_name, config, recordings_dir)
    path = recordings_dir / command_name / filename
    if path not in candidates:
        raise FileNotFoundError("Recording not found")
    return path


def input_devices() -> list[dict[str, Any]]:
    """Return one entry per named input exposed by the local audio host."""
    devices: list[dict[str, Any]] = []
    seen: set[str] = set()
    reported_devices = list_devices()
    routed_prefixes = ("alsa_input.", "bluez_input.")
    has_routed_sources = any(
        str(device.get("name", "")).strip().casefold().startswith(routed_prefixes)
        and int(device.get("max_input_channels", 0)) > 0
        for device in reported_devices
    )
    for index, device in enumerate(reported_devices):
        channels = int(device.get("max_input_channels", 0))
        name = str(device.get("name", "")).strip()
        key = name.casefold()
        if not name or channels < 1 or key in seen:
            continue
        seen.add(key)
        comparison_eligible = (
            key.startswith(routed_prefixes)
            if has_routed_sources
            else (
                key not in {"default", "default source", "pipewire", "sysdefault"}
                and ".monitor" not in key
                and not key.endswith(" monitor")
            )
        )
        devices.append(
            {
                "id": index,
                "name": name,
                "channels": channels,
                "comparison_eligible": comparison_eligible,
            }
        )
    return devices


def state_payload(
    config: AppConfig,
    recordings_dir: Path,
    devices: list[dict[str, Any]] | None = None,
    *,
    can_manage_commands: bool = False,
) -> dict[str, Any]:
    counts = template_counts(recordings_dir)
    return {
        "sample_rate": config.recognizer.sample_rate,
        "devices": devices or [],
        "commands": [
            {
                "name": command.name,
                "description": command.description,
                "utterance": command.utterance,
                "target_samples": command.target_samples,
                "count": counts.get(command.name, 0),
                "is_negative": False,
                "is_start_phrase": False,
            }
            for command in config.commands.values()
        ]
        + [
            {
                "name": config.calibration.name,
                "description": config.calibration.description,
                "utterance": config.calibration.utterance,
                "target_samples": config.calibration.target_samples,
                "count": counts.get(config.calibration.name, 0),
                "is_negative": True,
                "is_start_phrase": False,
            }
        ]
        + [
            {
                "name": config.start_phrase.name,
                "description": config.start_phrase.description,
                "utterance": config.start_phrase.utterance,
                "target_samples": config.start_phrase.target_samples,
                "count": counts.get(config.start_phrase.name, 0),
                "is_negative": False,
                "is_start_phrase": True,
            }
        ],
        "start_phrase_ready": (
            config.start_phrase.enabled
            and
            counts.get(config.start_phrase.name, 0)
            >= config.start_phrase.target_samples
        ),
        "start_phrase_enabled": config.start_phrase.enabled,
        "command_timeout_seconds": config.start_phrase.command_timeout_seconds,
        "can_manage_commands": can_manage_commands,
    }


def store_recording(
    *,
    wav_data: bytes,
    command_name: str,
    config: AppConfig,
    recordings_dir: Path,
) -> Path:
    allowed_names = recording_names(config)
    if command_name not in allowed_names:
        raise ValueError("Unknown command")
    if not wav_data or len(wav_data) > MAX_UPLOAD_BYTES:
        raise ValueError("Recording is empty or too large")

    audio = decode_wav(wav_data, config.recognizer.sample_rate)
    samples = (
        trim_spoken_phrase(audio.samples, audio.sample_rate)
        if command_name == config.start_phrase.name
        else trim_silence(audio.samples, audio.sample_rate)
    )
    duration = samples.size / audio.sample_rate
    if duration < 0.2:
        raise ValueError("No clear speech was detected")
    if duration > 6.0:
        raise ValueError("Recording is longer than 6 seconds")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = recordings_dir / command_name / f"{stamp}.wav"
    save_wav(output, Audio(samples=samples, sample_rate=audio.sample_rate))
    return output


def capture_cued_recording(
    *,
    command_name: str,
    device: int,
    config: AppConfig,
    recordings_dir: Path,
) -> Path:
    """Play the enrollment cue, capture one full take, and save it immediately."""
    play_confirmation_chime(sample_rate=config.recognizer.sample_rate)
    audio = record_audio(
        seconds=3.0,
        sample_rate=config.recognizer.sample_rate,
        device=device,
        countdown=0,
        trim=False,
    )
    return store_recording(
        wav_data=encode_wav(audio),
        command_name=command_name,
        config=config,
        recordings_dir=recordings_dir,
    )


def replace_recording(
    *,
    wav_data: bytes,
    command_name: str,
    filename: str,
    config: AppConfig,
    recordings_dir: Path,
) -> Path:
    """Save a replacement before removing the selected existing take."""
    previous = recording_file(command_name, filename, config, recordings_dir)
    replacement = store_recording(
        wav_data=wav_data,
        command_name=command_name,
        config=config,
        recordings_dir=recordings_dir,
    )
    try:
        previous.unlink()
    except OSError:
        replacement.unlink(missing_ok=True)
        raise
    return replacement


def match_recording(
    *,
    wav_data: bytes,
    config: AppConfig,
    recordings_dir: Path,
) -> dict[str, Any]:
    if not wav_data or len(wav_data) > MAX_UPLOAD_BYTES:
        raise ValueError("Recording is empty or too large")
    audio = decode_wav(wav_data, config.recognizer.sample_rate)
    samples = trim_silence(audio.samples, audio.sample_rate)
    if samples.size / audio.sample_rate < 0.2:
        raise ValueError("No clear speech was detected")

    templates = load_templates(recordings_dir, config.recognizer.sample_rate)
    if not templates:
        raise ValueError("Record command examples before testing")
    return match_samples(samples=samples, config=config, templates=templates)


def match_samples(
    *,
    samples: Any,
    config: AppConfig,
    templates: list[Template],
) -> dict[str, Any]:
    if not templates:
        raise ValueError("Record command examples before testing")
    samples = trim_silence(samples, config.recognizer.sample_rate)
    if samples.size / config.recognizer.sample_rate < 0.2:
        raise ValueError("No clear speech was detected")
    features = extract_command_features(samples, config.recognizer.sample_rate)
    return match_features(features=features, config=config, templates=templates)


def match_features(
    *,
    features: Any,
    config: AppConfig,
    templates: list[Template],
) -> dict[str, Any]:
    result = classify(
        features,
        templates,
        max_distance=config.recognizer.max_distance,
        min_margin=config.recognizer.min_margin,
        top_k=config.recognizer.top_k,
        default_template_limit=8,
    )
    intent_groups: dict[str, tuple[str, float]] = {}
    for command_name, score in result.per_command.items():
        command = config.commands.get(command_name)
        if command is None:
            continue
        intent_key = command.intent_group
        current = intent_groups.get(intent_key)
        if current is None or score < current[1]:
            intent_groups[intent_key] = (command_name, score)

    intent_ranking = sorted(intent_groups.values(), key=lambda item: item[1])
    if not intent_ranking:
        raise ValueError("Record command examples before testing")
    best_command, best_score = intent_ranking[0]
    competitors = intent_ranking[1:]
    negative_score = result.per_command.get("_not_command")
    if negative_score is not None:
        competitors.append(("_not_command", negative_score))
    competitors.sort(key=lambda item: item[1])
    if competitors:
        runner_command, runner_score = competitors[0]
        action_margin = max(
            0.0,
            (runner_score - best_score) / max(runner_score, 1e-9),
        )
    else:
        runner_command, runner_score, action_margin = None, float("inf"), 1.0
    accepted = (
        best_score <= config.recognizer.max_distance
        and action_margin >= config.recognizer.min_margin
    )
    matched = config.commands.get(best_command) if accepted else None
    best = config.commands.get(best_command)
    return {
        "accepted": accepted,
        "command": best_command if accepted else None,
        "utterance": matched.utterance if matched else None,
        "best_command": best_command,
        "best_utterance": best.utterance if best else best_command,
        "runner_command": runner_command,
        "runner_score": runner_score,
        "score": best_score,
        "margin": action_margin,
        "scores": result.per_command,
    }


class StudioServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        config: AppConfig,
        recordings_dir: Path,
        *,
        config_path: Path | None = None,
        listener_manager: ListenerManager | None = None,
        assets_dir: Path | None = None,
        response_dir: Path | None = None,
    ) -> None:
        super().__init__(server_address, StudioHandler)
        self.app_config = config
        self.recordings_dir = recordings_dir
        self.config_path = config_path
        self.assets_dir = assets_dir or recordings_dir.parent / "assets"
        self.response_dir = response_dir or recordings_dir.parent / "response-studio"
        self._responses = None
        self._responses_lock = threading.Lock()
        self.listener_manager = listener_manager or NoopListenerManager()
        self.listener = ContinuousListener(config.recognizer.sample_rate)
        self.trigger_captures = TriggerCaptureQueue(
            recordings_dir,
            sample_rate=config.recognizer.sample_rate,
        )

    def response_workspace(self):
        with self._responses_lock:
            if self._responses is None:
                self._responses = ResponseStudio(self.app_config, self.response_dir, self.assets_dir)
            self._responses.config = self.app_config
            return self._responses

    def server_close(self):
        if self._responses is not None:
            self._responses.close()
        super().server_close()

    def reload_config(self) -> None:
        if self.config_path is None:
            raise RuntimeError("This studio cannot edit its command configuration")
        self.app_config = load_config(self.config_path)

    def update_managed_listener(self) -> dict[str, Any]:
        """Restart an optional external listener after library changes."""
        has_command_templates = has_command_recordings(
            self.app_config,
            self.recordings_dir,
        )
        result = self.listener_manager.update(has_templates=has_command_templates)
        return {
            "listener_updated": result.updated,
            "listener_error": result.error,
            "listener_waiting_for_templates": result.waiting_for_templates,
        }

    def pause_managed_listener(self) -> dict[str, Any]:
        """Keep spoken enrollment audio away from the always-on recognizer."""
        result = self.listener_manager.pause()
        return {"listener_paused": result.paused, "listener_error": result.error}

    def start_listener(self, device: int) -> None:
        templates = load_templates(
            self.recordings_dir, self.app_config.recognizer.sample_rate
        )
        if not templates:
            raise ValueError("Record command examples before continuous listening")

        counts = template_counts(self.recordings_dir)
        start_phrase_ready = (
            self.app_config.start_phrase.enabled
            and
            counts.get(self.app_config.start_phrase.name, 0)
            >= self.app_config.start_phrase.target_samples
        )
        if start_phrase_ready:
            start_templates = load_start_phrase_templates(
                self.recordings_dir,
                start_phrase_name=self.app_config.start_phrase.name,
                negative_names={
                    *self.app_config.commands,
                    self.app_config.calibration.name,
                },
                sample_rate=self.app_config.recognizer.sample_rate,
            )

            def command_matcher(features: Any) -> dict[str, Any]:
                return match_features(
                    features=features,
                    config=self.app_config,
                    templates=templates,
                )

            gate = StartPhraseGate(
                sample_rate=self.app_config.recognizer.sample_rate,
                start_label=self.app_config.start_phrase.name,
                display_name=self.app_config.start_phrase.utterance,
                templates=start_templates,
                max_distance=self.app_config.start_phrase.max_distance,
                min_margin=self.app_config.start_phrase.min_margin,
                top_k=self.app_config.start_phrase.top_k,
                command_timeout_seconds=(
                    self.app_config.start_phrase.command_timeout_seconds
                ),
                min_audio_seconds=self.app_config.start_phrase.min_audio_seconds,
                max_audio_seconds=self.app_config.start_phrase.max_audio_seconds,
                min_command_audio_seconds=(
                    self.app_config.start_phrase.min_command_audio_seconds
                ),
                command_matcher=command_matcher,
            )
            def start_phrase_feedback() -> None:
                play_confirmation_chime(
                    sample_rate=self.app_config.recognizer.sample_rate
                )
                gate.arm_command_window()

            self.listener.start(
                device=device,
                matcher=gate,
                phase_provider=lambda: gate.phase,
                start_phrase_feedback=start_phrase_feedback,
                capture_callback=self.trigger_captures.capture,
            )
            return

        def direct_matcher(samples: Any) -> dict[str, Any]:
            payload = match_samples(
                samples=samples,
                config=self.app_config,
                templates=templates,
            )
            payload["kind"] = "command"
            return payload

        self.listener.start(device=device, matcher=direct_matcher, capture_callback=self.trigger_captures.capture)


class StudioHandler(BaseHTTPRequestHandler):
    server: StudioServer

    def log_message(self, message: str, *args: object) -> None:
        print(f"studio: {message % args}")

    def _review_diagnostic(self, path: str) -> None:
        host = self.headers.get("Host", "")
        origin = self.headers.get("Origin")
        try:
            allowed = {"localhost", "127.0.0.1", "::1", self.server.server_address[0]}
            safe = urlparse("http://" + host).hostname in allowed and (
                origin is None or origin in {"http://" + host, "https://" + host})
        except ValueError:
            safe = False
        if not safe or self.headers.get("X-Voice-IO") != "diagnostic-review":
            # Drain small rejected bodies before closing to avoid a TCP reset
            # obscuring the 403 response on Windows.
            try:
                rejected_size = int(self.headers.get("Content-Length", "0"))
                if 0 < rejected_size <= 4096:
                    self.rfile.read(rejected_size)
            except ValueError:
                pass
            self._send_json(HTTPStatus.FORBIDDEN, {"error": "Use the local Studio to review recordings"})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 4096 or self.headers.get_content_type() != "application/json":
                raise ValueError("Send a small JSON review request")
            payload = json.loads(self.rfile.read(size))
            if not isinstance(payload, dict):
                raise ValueError("Invalid review request")
            if path == "/api/diagnostics/review-mode":
                result = {"review_mode": self.server.trigger_captures.set_review_mode(payload.get("enabled"))}
            else:
                if self.server.listener.snapshot()["running"]:
                    self._send_json(HTTPStatus.CONFLICT, {"error": "Stop continuous testing before teaching a clip"})
                    return
                event_id = unquote(path[len("/api/diagnostics/"):-len("/teach")])
                result = self.server.trigger_captures.teach(event_id, payload.get("clip"), payload.get("label"), self.server.app_config)
                result.update(self.server.update_managed_listener())
            self._send_json(HTTPStatus.OK, result)
        except (ValueError, TypeError, KeyError, FileNotFoundError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except OSError:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "Could not save the review; check disk space and permissions"})

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _send_wav(self, body: bytes) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlparse(self.path).path
        if handle_response_request(self, path, "GET"):
            return
        if path == "/api/state":
            self._send_json(
                HTTPStatus.OK,
                state_payload(
                    self.server.app_config,
                    self.server.recordings_dir,
                    input_devices(),
                    can_manage_commands=self.server.config_path is not None,
                ),
            )
            return

        if path == "/api/listener":
            self.server.listener.heartbeat()
            self._send_json(HTTPStatus.OK, self.server.listener.snapshot())
            return

        if path == "/api/diagnostics":
            events = self.server.trigger_captures.list_events()
            self._send_json(
                HTTPStatus.OK,
                {
                    "events": events,
                    "count": len(events),
                    "capacity": self.server.trigger_captures.max_events,
                    "review_mode": self.server.trigger_captures.review_status(),
                },
            )
            return

        diagnostic_prefix = "/api/diagnostics/"
        if path.startswith(diagnostic_prefix):
            parts = path[len(diagnostic_prefix) :].split("/", 1)
            if len(parts) != 2:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                body = self.server.trigger_captures.audio_path(
                    unquote(parts[0]),
                    unquote(parts[1]),
                ).read_bytes()
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            except (FileNotFoundError, OSError):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._send_wav(body)
            return

        samples_prefix = "/api/samples/"
        if path.startswith(samples_prefix):
            command_name = unquote(path[len(samples_prefix) :])
            try:
                samples = recording_files(
                    command_name,
                    self.server.app_config,
                    self.server.recordings_dir,
                )
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            self._send_json(
                HTTPStatus.OK,
                {
                    "command": command_name,
                    "samples": [
                        {
                            "filename": sample.name,
                            "url": (
                                f"/api/audio/{quote(command_name, safe='')}"
                                f"/{quote(sample.name, safe='')}"
                            ),
                        }
                        for sample in samples
                    ],
                },
            )
            return

        audio_prefix = "/api/audio/"
        if path.startswith(audio_prefix):
            parts = path[len(audio_prefix) :].split("/", 1)
            if len(parts) != 2:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                sample = recording_file(
                    unquote(parts[0]),
                    unquote(parts[1]),
                    self.server.app_config,
                    self.server.recordings_dir,
                )
                body = sample.read_bytes()
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            except (FileNotFoundError, OSError):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._send_wav(body)
            return

        static = STATIC_FILES.get(path)
        if static is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        filename, content_type = static
        try:
            body = (STATIC_DIR / filename).read_bytes()
        except OSError:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; media-src 'self' blob:; connect-src 'self'; style-src 'self'; script-src 'self'")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlparse(self.path).path
        if handle_response_request(self, path, "POST"):
            return
        if path == "/api/diagnostics/review-mode" or (path.startswith("/api/diagnostics/") and path.endswith("/teach")):
            self._review_diagnostic(path)
            return
        if path == "/api/commands":
            self._create_command()
            return
        if path == "/api/capture":
            self._capture_audio()
            return
        if path == "/api/match":
            self._match_audio()
            return
        if path == "/api/listener/start":
            self._start_listener()
            return
        if path == "/api/listener/stop":
            self.server.listener.stop()
            self._send_json(HTTPStatus.OK, self.server.listener.snapshot())
            return
        if path == "/api/batch/finish":
            self._send_json(HTTPStatus.OK, self.server.update_managed_listener())
            return
        diagnostic_prefix = "/api/diagnostics/"
        if path.startswith(diagnostic_prefix) and path.endswith("/false-trigger"):
            event_id = unquote(
                path[len(diagnostic_prefix) : -len("/false-trigger")]
            )
            if self.server.listener.snapshot()["running"]:
                self._send_json(
                    HTTPStatus.CONFLICT,
                    {"error": "Stop continuous testing before teaching a rejection"},
                )
                return
            try:
                promoted = self.server.trigger_captures.promote_false_trigger(
                    event_id
                )
            except (ValueError, FileNotFoundError, OSError) as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            self._send_json(
                HTTPStatus.OK,
                {
                    **promoted,
                    "marked_false": True,
                    **self.server.update_managed_listener(),
                },
            )
            return
        batch_prefix = "/api/batch/capture/"
        if path.startswith(batch_prefix):
            self._capture_batch(unquote(path[len(batch_prefix) :]))
            return

        prefix = "/api/recordings/"
        if not path.startswith(prefix):
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        content_length = self.headers.get("Content-Length")
        if content_length is None:
            self._send_json(HTTPStatus.LENGTH_REQUIRED, {"error": "Missing recording size"})
            return
        try:
            size = int(content_length)
        except ValueError:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid recording size"})
            return
        if size <= 0 or size > MAX_UPLOAD_BYTES:
            self._send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "Recording is empty or too large"})
            return

        recording_parts = path[len(prefix) :].split("/", 1)
        command_name = unquote(recording_parts[0])
        replace_filename = (
            unquote(recording_parts[1]) if len(recording_parts) == 2 else None
        )
        wav_data = self.rfile.read(size)
        try:
            if replace_filename is None:
                store_recording(
                    wav_data=wav_data,
                    command_name=command_name,
                    config=self.server.app_config,
                    recordings_dir=self.server.recordings_dir,
                )
            else:
                replace_recording(
                    wav_data=wav_data,
                    command_name=command_name,
                    filename=replace_filename,
                    config=self.server.app_config,
                    recordings_dir=self.server.recordings_dir,
                )
        except (ValueError, OSError, EOFError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return

        counts = template_counts(self.server.recordings_dir)
        listener_status = self.server.update_managed_listener()
        self._send_json(
            HTTPStatus.CREATED,
            {
                "saved": True,
                "replaced": replace_filename is not None,
                "command": command_name,
                "count": counts[command_name],
                **listener_status,
            },
        )

    def do_DELETE(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlparse(self.path).path
        diagnostic_prefix = "/api/diagnostics/"
        if path.startswith(diagnostic_prefix):
            event_id = unquote(path[len(diagnostic_prefix) :])
            try:
                self.server.trigger_captures.dismiss(event_id)
            except (ValueError, FileNotFoundError, OSError) as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            self._send_json(
                HTTPStatus.OK,
                {"dismissed": True, "event_id": event_id},
            )
            return
        command_prefix = "/api/commands/"
        if path.startswith(command_prefix):
            command_name = unquote(path[len(command_prefix) :])
            if self.headers.get("X-Confirm-Command") != command_name:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "Confirm the command before removing it"},
                )
                return
            if self.server.listener.snapshot()["running"]:
                self._send_json(
                    HTTPStatus.CONFLICT,
                    {"error": "Stop continuous testing before removing a command"},
                )
                return
            if self.server.config_path is None:
                self._send_json(
                    HTTPStatus.FORBIDDEN,
                    {"error": "Command editing is unavailable in this studio"},
                )
                return
            try:
                archive_command(
                    self.server.config_path,
                    self.server.recordings_dir,
                    command_name=command_name,
                )
                self.server.reload_config()
            except (ValueError, OSError) as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            self._send_json(
                HTTPStatus.OK,
                {
                    "removed": True,
                    "command": command_name,
                    "recoverable": True,
                    **self.server.update_managed_listener(),
                },
            )
            return

        recording_prefix = "/api/recordings/"
        if path.startswith(recording_prefix):
            parts = path[len(recording_prefix) :].split("/", 1)
            if len(parts) != 2:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            command_name, filename = map(unquote, parts)
            try:
                recording = recording_file(
                    command_name,
                    filename,
                    self.server.app_config,
                    self.server.recordings_dir,
                )
                archive_recording(
                    self.server.recordings_dir,
                    command_name=command_name,
                    recording=recording,
                )
            except (ValueError, FileNotFoundError, OSError) as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            count = len(
                recording_files(
                    command_name,
                    self.server.app_config,
                    self.server.recordings_dir,
                )
            )
            self._send_json(
                HTTPStatus.OK,
                {
                    "removed": True,
                    "command": command_name,
                    "count": count,
                    "recoverable": True,
                    **self.server.update_managed_listener(),
                },
            )
            return

        self.send_error(HTTPStatus.NOT_FOUND)

    def _create_command(self) -> None:
        if self.server.listener.snapshot()["running"]:
            self._send_json(
                HTTPStatus.CONFLICT,
                {"error": "Stop continuous testing before adding a command"},
            )
            return
        if self.server.config_path is None:
            self._send_json(
                HTTPStatus.FORBIDDEN,
                {"error": "Command editing is unavailable in this studio"},
            )
            return
        content_length = self.headers.get("Content-Length")
        try:
            size = int(content_length or "0")
        except ValueError:
            size = 0
        if size <= 0 or size > 8192:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid command request"})
            return
        try:
            payload = json.loads(self.rfile.read(size))
            command = add_command(
                self.server.config_path,
                utterance=str(payload.get("utterance", "")),
                description=str(payload.get("description", "")),
                target_samples=int(payload.get("target_samples", 10)),
            )
            self.server.reload_config()
        except (TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        self._send_json(
            HTTPStatus.CREATED,
            {
                "created": True,
                "command": command.name,
                "utterance": command.utterance,
                **self.server.update_managed_listener(),
            },
        )

    def _capture_audio(self) -> None:
        content_length = self.headers.get("Content-Length")
        try:
            size = int(content_length or "0")
        except ValueError:
            size = 0
        if size <= 0 or size > 4096:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid capture request"})
            return
        try:
            payload = json.loads(self.rfile.read(size))
            device = int(payload["device"])
            preserve_silence = payload.get("preserve_silence") is True
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Choose a valid microphone"})
            return

        valid_devices = {item["id"] for item in input_devices()}
        if device not in valid_devices:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "The selected microphone is unavailable"})
            return
        try:
            audio = record_audio(
                seconds=3.0,
                sample_rate=self.server.app_config.recognizer.sample_rate,
                device=device,
                countdown=0,
                trim=not preserve_silence,
            )
        except (RuntimeError, ValueError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        self._send_wav(encode_wav(audio))

    def _capture_batch(self, command_name: str) -> None:
        if self.server.listener.snapshot()["running"]:
            self._send_json(
                HTTPStatus.CONFLICT,
                {"error": "Stop continuous testing before batch recording"},
            )
            return
        if command_name not in recording_names(self.server.app_config):
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Unknown command"})
            return
        content_length = self.headers.get("Content-Length")
        try:
            size = int(content_length or "0")
        except ValueError:
            size = 0
        if size <= 0 or size > 4096:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid batch request"})
            return
        try:
            payload = json.loads(self.rfile.read(size))
            device = int(payload["device"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Choose a valid microphone"})
            return
        if device not in {item["id"] for item in input_devices()}:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "The selected microphone is unavailable"},
            )
            return

        pause_status = self.server.pause_managed_listener()
        if pause_status["listener_error"]:
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": pause_status["listener_error"]},
            )
            return
        try:
            output = capture_cued_recording(
                command_name=command_name,
                device=device,
                config=self.server.app_config,
                recordings_dir=self.server.recordings_dir,
            )
        except (RuntimeError, ValueError, OSError, EOFError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        counts = template_counts(self.server.recordings_dir)
        self._send_json(
            HTTPStatus.CREATED,
            {
                "saved": True,
                "command": command_name,
                "filename": output.name,
                "count": counts.get(command_name, 0),
                **pause_status,
            },
        )

    def _match_audio(self) -> None:
        content_length = self.headers.get("Content-Length")
        try:
            size = int(content_length or "0")
        except ValueError:
            size = 0
        if size <= 0 or size > MAX_UPLOAD_BYTES:
            self._send_json(
                HTTPStatus.BAD_REQUEST, {"error": "Recording is empty or too large"}
            )
            return
        try:
            payload = match_recording(
                wav_data=self.rfile.read(size),
                config=self.server.app_config,
                recordings_dir=self.server.recordings_dir,
            )
        except (ValueError, OSError, EOFError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        self._send_json(HTTPStatus.OK, payload)

    def _start_listener(self) -> None:
        content_length = self.headers.get("Content-Length")
        try:
            size = int(content_length or "0")
        except ValueError:
            size = 0
        if size <= 0 or size > 4096:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid request"})
            return
        try:
            payload = json.loads(self.rfile.read(size))
            device = int(payload["device"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Choose a valid microphone"})
            return
        if device not in {item["id"] for item in input_devices()}:
            self._send_json(
                HTTPStatus.BAD_REQUEST, {"error": "The selected microphone is unavailable"}
            )
            return
        try:
            self.server.start_listener(device)
        except (RuntimeError, ValueError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        self._send_json(HTTPStatus.OK, self.server.listener.snapshot())


def serve_studio(
    *,
    config_path: Path,
    recordings_dir: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    listener_manager: ListenerManager | None = None,
    assets_dir: Path | None = None,
    response_dir: Path | None = None,
) -> None:
    config = load_config(config_path)
    server = StudioServer(
        (host, port),
        config,
        recordings_dir,
        config_path=config_path,
        listener_manager=listener_manager,
        assets_dir=assets_dir,
        response_dir=response_dir,
    )
    print(f"Recording studio: http://{host}:{server.server_port}")
    print("Recordings stay on this computer. Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping recording studio.")
    finally:
        server.listener.stop()
        server.server_close()
