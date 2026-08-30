"""Optional Hugging Face Space client for cloned response audio."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any, Callable

from .asset_processing import ProcessingResult, publish_asset_library


DEFAULT_SPACE_ID = "hkchengrex/OmniVoice"
DEFAULT_SPACE_URL = "https://hkchengrex-omnivoice.hf.space"
DEFAULT_API_NAME = "/_clone_fn"
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


class HuggingFaceSpaceVoiceCloner:
    """Generate and publish response clips through a Gradio Hugging Face Space."""

    def __init__(
        self,
        *,
        space_id: str = DEFAULT_SPACE_ID,
        api_name: str = DEFAULT_API_NAME,
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
        self.client = client
        self.file_wrapper = file_wrapper
        self.max_attempts = max_attempts
        self.retry_delay_seconds = retry_delay_seconds
        self.sleep = sleep

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
        remote: Any = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                remote = self.client.predict(
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
                break
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
