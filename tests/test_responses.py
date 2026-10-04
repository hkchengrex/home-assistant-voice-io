from pathlib import Path
import random

import numpy as np
import pytest

from ha_voice.audio import Audio, save_wav
from ha_voice.responses import VoiceResponsePlayer


GROUPS = {"start": "starting", "day": "lights_on", "bye": "bye"}


def _response_assets(root: Path, *, variants: int = 2) -> None:
    for prefix in GROUPS.values():
        for index in range(variants):
            samples = np.full(2400, 0.01 * (index + 1), dtype=np.float32)
            save_wav(root / f"{prefix}_{index + 1}.wav", Audio(samples, 24000))


def test_player_loads_caller_defined_groups_and_avoids_immediate_repeat(tmp_path: Path) -> None:
    _response_assets(tmp_path)
    played = []
    player = VoiceResponsePlayer(
        tmp_path,
        GROUPS,
        rng=random.Random(2),
        playback=lambda samples, rate, device: played.append((rate, device)),
        device="speaker",
    )

    first = player.play("start")
    second = player.play("start")
    day = player.play("day")
    bye = player.play("bye")

    assert first != second
    assert day.name.startswith("lights_on_")
    assert bye.name.startswith("bye_")
    assert played == [(24000, "speaker")] * 4


def test_player_requires_every_configured_group(tmp_path: Path) -> None:
    samples = np.zeros(2400, dtype=np.float32)
    save_wav(tmp_path / "starting_1.wav", Audio(samples, 24000))

    with pytest.raises(ValueError, match="No voice responses"):
        VoiceResponsePlayer(tmp_path, GROUPS)


def test_standard_playback_can_be_interrupted_without_waiting_for_clip(monkeypatch) -> None:
    import sys
    import threading
    import types
    from ha_voice.responses import _sounddevice_playback
    started = threading.Event()
    cancelled = threading.Event()
    stopped = threading.Event()
    class Stream:
        @property
        def active(self):
            return not stopped.is_set()
    fake = types.SimpleNamespace(
        play=lambda *args, **kwargs: started.set(),
        get_stream=lambda: Stream(),
        stop=stopped.set,
        wait=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    worker = threading.Thread(target=_sounddevice_playback,
        args=(np.zeros(24000, dtype=np.float32), 24000, None),
        kwargs={"cancelled": cancelled})
    worker.start()
    try:
        assert started.wait(1)
        cancelled.set()
        worker.join(1)
        assert not worker.is_alive()
        assert stopped.is_set()
    finally:
        cancelled.set()
        worker.join(1)


def test_cancelled_pending_response_never_starts(tmp_path: Path) -> None:
    import threading
    _response_assets(tmp_path)
    played = []
    player = VoiceResponsePlayer(tmp_path, GROUPS,
        playback=lambda *args: played.append(True))
    cancelled = threading.Event()
    cancelled.set()
    player.play("day", cancelled=cancelled)
    assert played == []
