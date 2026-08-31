from pathlib import Path
import threading
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from ha_voice.asset_processing import publish_asset_library
from ha_voice.audio import Audio, encode_wav
from ha_voice.config import load_config
from ha_voice.response_studio import ResponseStudio

ROOT = Path(__file__).resolve().parents[1]


def wav_bytes(seconds=2):
    samples = .1 * np.sin(np.arange(24000 * seconds) * 2 * np.pi * 180 / 24000)
    return encode_wav(Audio(samples.astype(np.float32), 24000))


class FakeCloner:
    def __init__(self, root):
        self.root = root
        self.calls = []
        self.fail_call = None

    def generate_batch(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == self.fail_call:
            raise RuntimeError("Simulated GPU failure")
        paths = []
        for _ in kwargs["texts"]:
            path = self.root / f"{uuid4().hex}.wav"
            path.write_bytes(wav_bytes())
            paths.append(path)
        return SimpleNamespace(audio_paths=tuple(paths), remote_metrics={})


@pytest.fixture
def workspace(tmp_path):
    fake = FakeCloner(tmp_path)
    studio = ResponseStudio(load_config(ROOT / "commands.toml"), tmp_path / "response-studio",
                            tmp_path / "assets", cloner_factory=lambda: fake)
    studio.upload_reference(wav_bytes())
    yield studio, fake
    studio.close()
    if studio.thread:
        studio.thread.join(timeout=5)


def request(studio, **overrides):
    phrase = next(p for p in studio.snapshot()["phrases"] if p["group"] == "lights_on")
    return {"request_id": uuid4().hex, "phrase_ids": [phrase["id"]], "takes": 2,
            "batch_size": 4, "consent_to_upload": True, **overrides}


def wait(studio):
    studio.thread.join(timeout=5)
    assert not studio.thread.is_alive()
    return studio.snapshot()


def test_examples_cover_every_command_and_event(workspace):
    studio, _ = workspace
    state = studio.snapshot()
    assert {p["group"] for p in state["phrases"]} == set(studio.config.responses)
    for command in studio.config.commands.values():
        assert sum(p["group"] == command.response for p in state["phrases"]) >= 4
    assert len(state["phrases"]) == 32


def test_phrase_edits_archive_restore_and_persistence(workspace):
    studio, fake = workspace
    phrase = studio.save_phrase({"group": "lights_on", "text": "Shine on."})
    studio.save_phrase({**phrase, "text": "Here comes the light."})
    studio.archive_phrase(phrase["id"], True)
    reopened = ResponseStudio(studio.config, studio.directory, studio.assets_dir, cloner_factory=lambda: fake)
    saved = next(p for p in reopened.snapshot()["phrases"] if p["id"] == phrase["id"])
    assert saved["text"] == "Here comes the light."
    assert saved["archived"]
    reopened.archive_phrase(phrase["id"], False)
    assert not reopened._find("phrases", phrase["id"])["archived"]


def test_multiple_takes_are_drafts_with_frozen_settings(workspace):
    studio, fake = workspace
    payload = request(studio, takes=3, settings={"speed": .9, "guidance_scale": 1.6})
    studio.generate(payload)
    state = wait(studio)
    assert state["jobs"][-1]["status"] == "completed"
    assert len(state["candidates"]) == 3
    assert all(c["status"] == "pending" and not c["published"] for c in state["candidates"])
    assert fake.calls[0]["texts"] == ["Done."] * 3
    assert fake.calls[0]["settings"].guidance_scale == 1.6
    assert not studio.assets_dir.exists()
    studio.save_phrase({"id": payload["phrase_ids"][0], "group": "lights_on", "text": "New text"})
    assert state["candidates"][0]["text"] == "Done."
    assert studio.snapshot()["candidates"][0]["text"] == "Done."


def test_select_keep_publish_and_unpublish_preserves_other_audio(workspace):
    studio, _ = workspace
    studio.assets_dir.mkdir()
    unrelated = studio.assets_dir / "lights_on_1.wav"
    unrelated.write_bytes(wav_bytes())
    before = unrelated.read_bytes()
    studio.generate(request(studio))
    one, two = wait(studio)["candidates"]
    with pytest.raises(ValueError, match="Only kept"):
        studio.publish({"candidate_ids": [one["id"]]})
    studio.review({"candidate_ids": [one["id"]], "status": "kept"})
    studio.review({"candidate_ids": [two["id"]], "status": "rejected"})
    studio.publish({"candidate_ids": [one["id"]]})
    assert len(list(studio.assets_dir.glob("*.wav"))) == 2
    assert unrelated.read_bytes() == before
    with pytest.raises(ValueError, match="Remove"):
        studio.review({"candidate_ids": [one["id"]], "status": "rejected"})
    studio.unpublish(one["id"])
    assert list(studio.assets_dir.glob("*.wav")) == [unrelated]
    assert studio.audio_path(one["id"]).exists()
    assert studio.audio_path(two["id"]).exists()


def test_regenerate_rejected_keeps_history_and_uses_current_controls(workspace):
    studio, fake = workspace
    studio.generate(request(studio, takes=1))
    old = wait(studio)["candidates"][0]
    studio.review({"candidate_ids": [old["id"]], "status": "rejected"})
    studio.generate(request(studio, candidate_ids=[old["id"]], settings={"speed": 1.2, "guidance_scale": 3.0}))
    state = wait(studio)
    assert len(state["candidates"]) == 2
    assert state["candidates"][0]["replacement_id"] == state["candidates"][1]["id"]
    assert state["candidates"][0]["status"] == "rejected"
    assert state["candidates"][1]["status"] == "pending"
    assert fake.calls[-1]["settings"].speed == 1.2
    assert studio.audio_path(old["id"]).exists()


def test_failure_preserves_completed_batches_and_retry_only_remaining(workspace):
    studio, fake = workspace
    phrases = [p["id"] for p in studio.snapshot()["phrases"] if p["group"] == "lights_on"]
    fake.fail_call = 2
    first = studio.generate(request(studio, phrase_ids=phrases, takes=3))
    state = wait(studio)
    assert state["jobs"][-1]["status"] == "failed"
    assert state["jobs"][-1]["completed"] == 8
    assert len(state["candidates"]) == 8
    studio.generate(request(studio, retry_job_id=first["id"]))
    state = wait(studio)
    assert state["jobs"][-1]["status"] == "completed"
    assert state["jobs"][-1]["total"] == 4
    assert len(state["candidates"]) == 12
    with pytest.raises(ValueError, match="already retried"):
        studio.generate(request(studio, retry_job_id=first["id"]))


def test_duplicate_request_does_not_charge_for_another_job(workspace):
    studio, fake = workspace
    payload = request(studio)
    first = studio.generate(payload)
    wait(studio)
    second = studio.generate(payload)
    assert second["id"] == first["id"]
    assert len(fake.calls) == 1


def test_cancel_retains_inflight_batch_and_reference_snapshot(workspace):
    studio, fake = workspace
    started, release = threading.Event(), threading.Event()
    original = fake.generate_batch
    def paused(**kwargs):
        started.set()
        assert release.wait(timeout=5)
        return original(**kwargs)
    fake.generate_batch = paused
    phrases = [p["id"] for p in studio.snapshot()["phrases"] if p["group"] == "lights_on"]
    reference = studio.audio_path()
    job = studio.generate(request(studio, phrase_ids=phrases, takes=3))
    assert started.wait(timeout=5)
    with pytest.raises(ValueError, match="already running"):
        studio.generate(request(studio))
    studio.upload_reference(wav_bytes(seconds=3))
    studio.cancel(job["id"])
    release.set()
    state = wait(studio)
    assert state["jobs"][-1]["status"] == "cancelled"
    assert state["jobs"][-1]["completed"] == 8
    assert fake.calls[0]["reference_audio"] == reference


@pytest.mark.parametrize("change", [{"consent_to_upload": False}, {"takes": 0}, {"takes": 1.5},
    {"batch_size": 9}, {"phrase_ids": ["../oops"]}, {"settings": {"token": "forbidden"}},
    {"settings": {"guidance_scale": float("nan")}}, {"settings": {"duration": 16}},
    {"settings": {"denoise": "false"}}, {"settings": {"speed": 2}}])
def test_invalid_generation_never_calls_remote(workspace, change):
    studio, fake = workspace
    with pytest.raises(ValueError):
        studio.generate(request(studio, **change))
    assert not fake.calls
    assert not studio.snapshot()["jobs"]


def test_clip_length_limits_partition_long_phrases(workspace):
    studio, fake = workspace
    phrase = studio.save_phrase({"group": "lights_on", "text": "a" * 300})
    studio.generate(request(studio, phrase_ids=[phrase["id"]], takes=8))
    assert wait(studio)["jobs"][-1]["status"] == "completed"
    assert [len(c["texts"]) for c in fake.calls] == [4, 4]


def test_published_file_modified_elsewhere_is_not_removed(workspace):
    studio, _ = workspace
    studio.generate(request(studio, takes=1))
    candidate = wait(studio)["candidates"][0]
    studio.review({"candidate_ids": [candidate["id"]], "status": "kept"})
    studio.publish({"candidate_ids": [candidate["id"]]})
    path = next(studio.assets_dir.glob("*.wav"))
    path.write_bytes(b"external change")
    with pytest.raises(ValueError, match="changed outside"):
        studio.unpublish(candidate["id"])
    assert path.read_bytes() == b"external change"


def test_cli_library_publication_preserves_reviewed_studio_takes(tmp_path):
    source = tmp_path / "welcome/one.wav"
    source.parent.mkdir()
    source.write_bytes(wav_bytes())
    reviewed = tmp_path / "welcome_studio_abc.wav"
    reviewed.write_bytes(wav_bytes())
    publish_asset_library(tmp_path, prefixes={"welcome"})
    assert reviewed.exists()


def test_reference_validation_and_identifier_paths(workspace):
    studio, _ = workspace
    before = studio.audio_path()
    with pytest.raises(ValueError):
        studio.upload_reference(b"not wav")
    with pytest.raises(ValueError):
        studio.upload_reference(wav_bytes(seconds=21))
    with pytest.raises(ValueError):
        studio.audio_path("../references")
    assert studio.audio_path() == before


def test_interrupted_jobs_are_not_automatically_replayed(workspace):
    studio, fake = workspace
    studio.data["jobs"].append({"id": uuid4().hex, "status": "running"})
    studio._save()
    reopened = ResponseStudio(studio.config, studio.directory, studio.assets_dir, cloner_factory=lambda: fake)
    assert reopened.snapshot()["jobs"][-1]["status"] == "interrupted"
    assert fake.calls == []


def test_incomplete_batch_leaves_no_partial_candidates(workspace):
    studio, fake = workspace
    original = fake.generate_batch
    def incomplete(**kwargs):
        result = original(**kwargs)
        return SimpleNamespace(audio_paths=result.audio_paths[:-1], remote_metrics={})
    fake.generate_batch = incomplete
    studio.generate(request(studio))
    state = wait(studio)
    assert state["jobs"][-1]["status"] == "failed"
    assert state["candidates"] == []


def test_invalid_audio_rolls_back_entire_chunk(workspace):
    studio, fake = workspace
    original = fake.generate_batch
    def invalid(**kwargs):
        result = original(**kwargs)
        result.audio_paths[-1].write_bytes(b"not audio")
        return result
    fake.generate_batch = invalid
    studio.generate(request(studio))
    state = wait(studio)
    assert state["jobs"][-1]["status"] == "failed"
    assert state["candidates"] == []
    assert not list((studio.directory / "candidates").glob("*.wav"))


def test_publication_failure_restores_files_and_reviews(workspace, monkeypatch):
    import ha_voice.response_studio as module
    studio, _ = workspace
    studio.generate(request(studio))
    ids = [c["id"] for c in wait(studio)["candidates"]]
    studio.review({"candidate_ids": ids, "status": "kept"})
    copy = module._atomic_copy
    calls = []
    def failing_copy(source, destination):
        calls.append(destination)
        if len(calls) == 2:
            raise OSError("Simulated full disk")
        copy(source, destination)
    monkeypatch.setattr(module, "_atomic_copy", failing_copy)
    with pytest.raises(OSError):
        studio.publish({"candidate_ids": ids})
    assert not list(studio.assets_dir.glob("*.wav"))
    assert all(not c["published"] and c["status"] == "kept" for c in studio.snapshot()["candidates"])


def test_review_save_failure_preserves_previous_state(workspace, monkeypatch):
    studio, _ = workspace
    studio.generate(request(studio, takes=1))
    candidate = wait(studio)["candidates"][0]
    persisted = studio.path.read_bytes()
    def failing_save():
        raise OSError("Simulated full disk")
    monkeypatch.setattr(studio, "_save", failing_save)
    with pytest.raises(OSError):
        studio.review({"candidate_ids": [candidate["id"]], "status": "kept"})
    assert studio.snapshot()["candidates"][0]["status"] == "pending"
    assert studio.path.read_bytes() == persisted
