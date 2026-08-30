from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest

from ha_voice.audio import Audio, load_wav, save_wav
from ha_voice.voice_generation import (
    DEFAULT_SPACE_ID,
    DEFAULT_SPACE_URL,
    CloneSettings,
    HuggingFaceSpaceVoiceCloner,
    VoiceGenerationError,
    _gradio_components,
)


class FakeClient:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def predict(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


def test_gradio_client_receives_environment_token(monkeypatch) -> None:
    captured: dict[str, object] = {}
    fake_module = ModuleType("gradio_client")

    class CapturingClient:
        def __init__(
            self,
            src: str,
            token: str | None = None,
            verbose: bool = True,
        ) -> None:
            captured.update(src=src, token=token, verbose=verbose)

    fake_module.Client = CapturingClient
    fake_module.handle_file = lambda path: path
    monkeypatch.setitem(sys.modules, "gradio_client", fake_module)

    factory, file_wrapper = _gradio_components()
    factory("owner/space", "hf_test")

    assert captured == {
        "src": "owner/space",
        "token": "hf_test",
        "verbose": False,
    }
    assert file_wrapper("reference.wav") == "reference.wav"

    factory(DEFAULT_SPACE_ID, None)
    assert captured["src"] == DEFAULT_SPACE_URL
    assert captured["token"] is None


def _speech_wav(path: Path, *, sample_rate: int = 24_000) -> None:
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    samples = np.concatenate(
        (
            np.zeros(sample_rate // 10, dtype=np.float32),
            0.08 * np.sin(2 * np.pi * 180 * time),
        )
    )
    save_wav(path, Audio(samples=samples, sample_rate=sample_rate))


def test_clone_requires_explicit_upload_consent(tmp_path: Path) -> None:
    generated = tmp_path / "remote.wav"
    reference = tmp_path / "reference.wav"
    _speech_wav(generated)
    _speech_wav(reference)
    client = FakeClient((str(generated), "Done."))
    cloner = HuggingFaceSpaceVoiceCloner(
        client=client,
        file_wrapper=lambda path: path,
    )

    with pytest.raises(ValueError, match="explicit consent"):
        cloner.clone_to_library(
            text="Welcome home",
            reference_audio=reference,
            assets_dir=tmp_path / "assets",
            response_prefix="welcome",
        )

    assert client.calls == []


def test_clone_uploads_expected_inputs_and_publishes_response(tmp_path: Path) -> None:
    generated = tmp_path / "remote.wav"
    reference = tmp_path / "reference.wav"
    _speech_wav(generated)
    _speech_wav(reference)
    client = FakeClient(({"path": str(generated)}, "Done."))
    cloner = HuggingFaceSpaceVoiceCloner(
        client=client,
        file_wrapper=lambda path: {"upload": path},
    )

    result = cloner.clone_to_library(
        text="  Welcome home  ",
        reference_audio=reference,
        assets_dir=tmp_path / "assets",
        response_prefix="welcome",
        settings=CloneSettings(
            language="English",
            reference_text="This is my reference.",
            instruct="warm",
            inference_steps=24,
            guidance_scale=1.8,
            speed=0.95,
        ),
        consent_to_upload=True,
    )

    args, kwargs = client.calls[0]
    assert args == (
        "Welcome home",
        "English",
        {"upload": str(reference.resolve())},
        "This is my reference.",
        "warm",
        24,
        1.8,
        True,
        0.95,
        None,
        True,
        True,
    )
    assert kwargs == {"api_name": "/_clone_fn"}
    assert result.source_path.name == "omnivoice_0001.wav"
    assert result.source_path.parent == tmp_path / "assets" / "welcome"
    assert result.source_path.exists()
    assert [path.name for path, _ in result.published] == ["welcome_1.wav"]
    published = load_wav(tmp_path / "assets" / "welcome_1.wav")
    assert published.samples.size > 0


def test_clone_rejects_remote_error_without_writing_library(tmp_path: Path) -> None:
    generated = tmp_path / "remote.wav"
    reference = tmp_path / "reference.wav"
    _speech_wav(generated)
    _speech_wav(reference)
    cloner = HuggingFaceSpaceVoiceCloner(
        client=FakeClient((str(generated), "Error: generation failed")),
        file_wrapper=lambda path: path,
    )

    with pytest.raises(VoiceGenerationError, match="did not generate"):
        cloner.clone_to_library(
            text="Welcome home",
            reference_audio=reference,
            assets_dir=tmp_path / "assets",
            response_prefix="welcome",
            consent_to_upload=True,
        )

    assert not (tmp_path / "assets").exists()


def test_clone_reports_sanitized_remote_exception(tmp_path: Path) -> None:
    reference = tmp_path / "reference.wav"
    _speech_wav(reference)

    class FailingClient:
        def predict(self, *args, **kwargs):
            raise RuntimeError("queue\nfailed")

    cloner = HuggingFaceSpaceVoiceCloner(
        client=FailingClient(),
        file_wrapper=lambda path: path,
    )

    with pytest.raises(
        VoiceGenerationError, match=r"RuntimeError: queue failed"
    ):
        cloner.clone_to_library(
            text="Welcome home",
            reference_audio=reference,
            assets_dir=tmp_path / "assets",
            response_prefix="welcome",
            consent_to_upload=True,
        )


def test_clone_retries_transient_zero_gpu_error(tmp_path: Path) -> None:
    generated = tmp_path / "remote.wav"
    reference = tmp_path / "reference.wav"
    _speech_wav(generated)
    _speech_wav(reference)

    class EventuallyAvailableClient:
        def __init__(self) -> None:
            self.calls = 0

        def predict(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("No GPU was available after 60s. Retry later")
            return str(generated), "Done."

    client = EventuallyAvailableClient()
    delays: list[float] = []
    cloner = HuggingFaceSpaceVoiceCloner(
        client=client,
        file_wrapper=lambda path: path,
        max_attempts=2,
        retry_delay_seconds=3,
        sleep=delays.append,
    )

    result = cloner.clone_to_library(
        text="Welcome home",
        reference_audio=reference,
        assets_dir=tmp_path / "assets",
        response_prefix="welcome",
        consent_to_upload=True,
    )

    assert client.calls == 2
    assert delays == [3]
    assert result.source_path.exists()
