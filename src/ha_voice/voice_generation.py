"""Optional Hugging Face Space client for cloned response audio."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
import wave
from typing import Any, Callable, Sequence

from .asset_processing import ProcessingResult, publish_asset_library


DEFAULT_SPACE_ID = "hkchengrex/OmniVoice"
DEFAULT_SPACE_URL = "https://hkchengrex-omnivoice.hf.space"
DEFAULT_API_NAME = "/_clone_fn"
DEFAULT_BATCH_API_NAME = "/clone_batch"
_SAFE_GROUP = re.compile(r"[A-Za-z0-9_-]+")


class VoiceGenerationError(RuntimeError):
    """A remote voice-generation request failed or returned invalid output."""


@dataclass(frozen=True)
class CloneSettings:
    language: str = "Auto"
    reference_text: str = ""
    instruct: str = ""
    inference_steps: int = 32
    guidance_scale: float = 2.0
    denoise: bool = True
    speed: float = 1.0
    duration: float | None = None
    preprocess_prompt: bool = True
    postprocess_output: bool = True


@dataclass(frozen=True)
class CloneResult:
    source_path: Path
    published: tuple[tuple[Path, ProcessingResult], ...]
    remote_status: str


@dataclass(frozen=True)
class BatchCloneResult:
    source_paths: tuple[Path, ...]
    published: tuple[tuple[Path, ProcessingResult], ...]
    remote_metrics: dict[str, Any]


@dataclass(frozen=True)
class GeneratedBatch:
    audio_paths: tuple[Path, ...]
    remote_metrics: dict[str, Any]


ClientFactory = Callable[[str, str | None], Any]
FileWrapper = Callable[[str], Any]
Sleep = Callable[[float], None]


def _transient_remote_error(exc: Exception) -> bool:
    message = str(exc).casefold()
    return any(
        marker in message
        for marker in (
            "no gpu was available",
            "temporarily unavailable",
            "timed out",
            "timeout",
            "connection reset",
            "forcibly closed",
            " 502",
            " 503",
            " 504",
        )
    )


def _gradio_components() -> tuple[ClientFactory, FileWrapper]:
    try:
        from gradio_client import Client, handle_file
    except ImportError as exc:
        raise VoiceGenerationError(
            "Install the 'voice-clone' extra to use Hugging Face voice generation"
        ) from exc

    def create_client(space_id: str, token: str | None) -> Any:
        source = DEFAULT_SPACE_URL if space_id == DEFAULT_SPACE_ID else space_id
        return Client(source, token=token, verbose=False)

    return create_client, handle_file


def _audio_path(value: Any) -> Path:
    if isinstance(value, (str, os.PathLike)):
        path = Path(value)
    elif isinstance(value, dict):
        candidate = value.get("path") or value.get("name")
        if not isinstance(candidate, (str, os.PathLike)):
            raise VoiceGenerationError("OmniVoice returned audio without a file path")
        path = Path(candidate)
    else:
        raise VoiceGenerationError("OmniVoice returned an unsupported audio result")
    if not path.is_file():
        raise VoiceGenerationError("The generated audio download is unavailable")
    return path


def _next_source_path(group_dir: Path) -> Path:
    group_dir.mkdir(parents=True, exist_ok=True)
    for index in range(1, 100_000):
        candidate = group_dir / f"omnivoice_{index:04d}.wav"
        if not candidate.exists():
            return candidate
    raise VoiceGenerationError("The response group contains too many generated clips")


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copyfile(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_batch(
    generated: list[Path], assets_dir: Path, prefix: str,
) -> tuple[tuple[Path, ...], tuple[tuple[Path, ProcessingResult], ...]]:
    with tempfile.TemporaryDirectory(prefix="voice-io-batch-") as directory:
        staging = Path(directory) / "staging"
        group = staging / prefix
        group.mkdir(parents=True)
        existing = assets_dir / prefix
        if existing.exists():
            for source in existing.glob("*.wav"):
                shutil.copyfile(source, group / source.name)
        sources = []
        for source in generated:
            destination = _next_source_path(group)
            shutil.copyfile(source, destination)
            sources.append(destination)
        # Validate every new and existing clip before changing the live library.
        prepared = publish_asset_library(staging, prefixes={prefix})
        changes = [(source, assets_dir / source.relative_to(staging)) for source in sources]
        changes.extend((source, assets_dir / source.name) for source, _ in prepared)
        backups: dict[Path, Path] = {}
        for index, (_, destination) in enumerate(changes):
            if destination.exists():
                backup = Path(directory) / f"backup-{index}.wav"
                shutil.copyfile(destination, backup)
                backups[destination] = backup
        changed: list[Path] = []
        try:
            for source, destination in changes:
                _atomic_copy(source, destination)
                changed.append(destination)
        except Exception:
            for destination in reversed(changed):
                if destination in backups:
                    _atomic_copy(backups[destination], destination)
                else:
                    destination.unlink(missing_ok=True)
            raise
        return (
            tuple(assets_dir / source.relative_to(staging) for source in sources),
            tuple((assets_dir / source.name, result) for source, result in prepared),
        )


class HuggingFaceSpaceVoiceCloner:
    """Generate and publish response clips through a Gradio Hugging Face Space."""

    def __init__(
        self,
        *,
        space_id: str = DEFAULT_SPACE_ID,
        api_name: str = DEFAULT_API_NAME,
        batch_api_name: str = DEFAULT_BATCH_API_NAME,
        token: str | None = None,
        client: Any | None = None,
        file_wrapper: FileWrapper | None = None,
        max_attempts: int = 2,
        retry_delay_seconds: float = 5.0,
        sleep: Sleep = time.sleep,
    ) -> None:
        if not space_id.strip():
            raise ValueError("space_id cannot be empty")
        if not api_name.startswith("/"):
            raise ValueError("api_name must start with '/'")
        if not batch_api_name.startswith("/"):
            raise ValueError("batch_api_name must start with '/'")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds cannot be negative")
        if client is None or file_wrapper is None:
            client_factory, default_file_wrapper = _gradio_components()
            try:
                client = client or client_factory(space_id, token)
            except Exception as exc:
                raise VoiceGenerationError(
                    "Could not connect to the OmniVoice Space. If it requires "
                    "authentication, set HF_TOKEN in the environment."
                ) from exc
            file_wrapper = file_wrapper or default_file_wrapper
        self.space_id = space_id
        self.api_name = api_name
        self.batch_api_name = batch_api_name
        self.client = client
        self.file_wrapper = file_wrapper
        self.max_attempts = max_attempts
        self.retry_delay_seconds = retry_delay_seconds
        self.sleep = sleep

    def _predict(self, *args: Any, api_name: str) -> Any:
        for attempt in range(1, self.max_attempts + 1):
            try:
                return self.client.predict(*args, api_name=api_name)
            except Exception as exc:
                if attempt < self.max_attempts and _transient_remote_error(exc):
                    self.sleep(self.retry_delay_seconds * attempt)
                    continue
                detail = " ".join(str(exc).split())[:300]
                suffix = f" ({type(exc).__name__}: {detail})" if detail else ""
                raise VoiceGenerationError(
                    "OmniVoice request failed"
                    f"{suffix}. If the Space requires authentication, set HF_TOKEN "
                    "in the environment."
                ) from exc

    def clone_batch_to_library(
        self,
        *,
        texts: Sequence[str],
        reference_audio: Path,
        assets_dir: Path,
        response_prefix: str,
        settings: CloneSettings | None = None,
        batch_size: int = 4,
        consent_to_upload: bool = False,
    ) -> BatchCloneResult:
        """Generate up to eight variations in one request, then publish the group.

        All lines share the reference and generation settings. Output order matches
        input order. Validate and normalize the entire batch before changing assets.
        This endpoint must be installed on the Space; there is no silent fallback
        to independent requests. Do not run simultaneous publishers on one library.
        """
        if not _SAFE_GROUP.fullmatch(response_prefix):
            raise ValueError("response_prefix contains unsupported characters")
        generated = self.generate_batch(
            texts=texts, reference_audio=reference_audio, settings=settings,
            batch_size=batch_size, consent_to_upload=consent_to_upload,
        )
        try:
            paths, published = _publish_batch(
                list(generated.audio_paths), assets_dir.expanduser().resolve(), response_prefix
            )
        except (wave.Error, EOFError) as exc:
            raise VoiceGenerationError("The batch contains invalid WAV audio") from exc
        return BatchCloneResult(paths, published, generated.remote_metrics)

    def generate_batch(
        self, *, texts: Sequence[str], reference_audio: Path,
        settings: CloneSettings | None = None, batch_size: int = 4,
        consent_to_upload: bool = False,
    ) -> GeneratedBatch:
        """Download candidate clips without publishing them to a playback library."""
        if not consent_to_upload:
            raise ValueError("Uploading reference voice audio requires explicit consent")
        if isinstance(texts, (str, bytes)) or not 1 <= len(texts) <= 8:
            raise ValueError("texts must contain 1–8 strings")
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Each text must be a non-empty string")
        lines = [text.strip() for text in texts]
        if any(len(text) > 300 for text in lines) or sum(map(len, lines)) > 1200:
            raise ValueError("Batch text is limited to 300 characters per line and 1200 total")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 8:
            raise ValueError("batch_size must be an integer between 1 and 8")
        reference_audio = reference_audio.expanduser().resolve()
        if not reference_audio.is_file():
            raise ValueError(f"Reference audio does not exist: {reference_audio}")
        settings = settings or CloneSettings()
        if isinstance(settings.inference_steps, bool) or not isinstance(settings.inference_steps, int) or not 4 <= settings.inference_steps <= 64:
            raise ValueError("inference_steps must be an integer between 4 and 64")
        if not 0 <= settings.guidance_scale <= 4:
            raise ValueError("guidance_scale must be between 0 and 4")
        if not 0.5 <= settings.speed <= 1.5:
            raise ValueError("speed must be between 0.5 and 1.5")
        if settings.duration is not None and not 0 < settings.duration <= 15:
            raise ValueError("duration must be positive and at most 15 seconds per clip")
        remote = self._predict(
            lines, settings.language, self.file_wrapper(str(reference_audio)),
            settings.reference_text, settings.instruct, settings.inference_steps,
            settings.guidance_scale, settings.denoise, settings.speed, settings.duration,
            settings.preprocess_prompt, settings.postprocess_output, batch_size,
            api_name=self.batch_api_name,
        )
        if not isinstance(remote, (list, tuple)) or len(remote) != 2:
            raise VoiceGenerationError("OmniVoice returned an unexpected batch response")
        files, metrics = remote
        if (
            not isinstance(metrics, dict) or metrics.get("schema_version") != 1
            or metrics.get("status") != "done" or metrics.get("count") != len(lines)
            or not isinstance(files, (list, tuple)) or len(files) != len(lines)
        ):
            raise VoiceGenerationError("OmniVoice returned an incomplete or failed batch")
        return GeneratedBatch(tuple(_audio_path(file) for file in files), dict(metrics))

    def clone_to_library(
        self,
        *,
        text: str,
        reference_audio: Path,
        assets_dir: Path,
        response_prefix: str,
        settings: CloneSettings | None = None,
        consent_to_upload: bool = False,
    ) -> CloneResult:
        """Upload a reference, generate one clip, and publish its response group."""
        if not consent_to_upload:
            raise ValueError(
                "Uploading reference voice audio requires explicit consent"
            )
        if not text.strip():
            raise ValueError("text cannot be empty")
        if not _SAFE_GROUP.fullmatch(response_prefix):
            raise ValueError("response_prefix contains unsupported characters")
        reference_audio = reference_audio.expanduser().resolve()
        if not reference_audio.is_file():
            raise ValueError(f"Reference audio does not exist: {reference_audio}")
        if settings is None:
            settings = CloneSettings()
        if not 4 <= settings.inference_steps <= 64:
            raise ValueError("inference_steps must be between 4 and 64")
        if not 0 <= settings.guidance_scale <= 4:
            raise ValueError("guidance_scale must be between 0 and 4")
        if not 0.5 <= settings.speed <= 1.5:
            raise ValueError("speed must be between 0.5 and 1.5")
        if settings.duration is not None and settings.duration <= 0:
            raise ValueError("duration must be positive")

        upload = self.file_wrapper(str(reference_audio))
        remote = self._predict(
            text.strip(),
            settings.language,
            upload,
            settings.reference_text,
            settings.instruct,
            settings.inference_steps,
            settings.guidance_scale,
            settings.denoise,
            settings.speed,
            settings.duration,
            settings.preprocess_prompt,
            settings.postprocess_output,
            api_name=self.api_name,
        )

        if not isinstance(remote, (list, tuple)) or len(remote) < 2:
            raise VoiceGenerationError("OmniVoice returned an unexpected response")
        generated = _audio_path(remote[0])
        status = str(remote[1])
        if not status.casefold().startswith("done"):
            raise VoiceGenerationError(f"OmniVoice did not generate audio: {status}")

        assets_dir = assets_dir.expanduser().resolve()
        source_path = _next_source_path(assets_dir / response_prefix)
        temporary = source_path.with_suffix(".tmp.wav")
        try:
            shutil.copyfile(generated, temporary)
            temporary.replace(source_path)
            published = publish_asset_library(
                assets_dir, prefixes={response_prefix}
            )
        except Exception:
            source_path.unlink(missing_ok=True)
            raise
        finally:
            temporary.unlink(missing_ok=True)
        return CloneResult(
            source_path=source_path,
            published=tuple(published),
            remote_status=status,
        )
