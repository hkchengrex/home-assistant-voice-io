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
