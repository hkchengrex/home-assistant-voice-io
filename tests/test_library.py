from pathlib import Path

import pytest

from ha_voice.config import load_config
from ha_voice.library import add_command, archive_command, archive_recording


ROOT = Path(__file__).parents[1]


def _config_copy(tmp_path: Path) -> Path:
    path = tmp_path / "commands.toml"
    path.write_bytes((ROOT / "commands.toml").read_bytes())
    return path


def test_add_command_persists_a_unique_safe_name(tmp_path: Path) -> None:
    path = _config_copy(tmp_path)

    first = add_command(path, utterance="Coffee time", target_samples=8)
    second = add_command(path, utterance="Coffee time", target_samples=12)

    config = load_config(path)
    assert first.name == "coffee_time"
    assert second.name == "coffee_time_2"
    assert config.commands[first.name].target_samples == 8
    assert config.commands[second.name].utterance == "Coffee time"


def test_add_command_supports_non_latin_phrases(tmp_path: Path) -> None:
    path = _config_copy(tmp_path)

    command = add_command(path, utterance="晚安")

    assert command.name == "command"
    assert load_config(path).commands[command.name].utterance == "晚安"


@pytest.mark.parametrize("target", [0, 101])
def test_add_command_validates_target_count(tmp_path: Path, target: int) -> None:
    with pytest.raises(ValueError, match="between 1 and 100"):
        add_command(_config_copy(tmp_path), utterance="Coffee", target_samples=target)


def test_archive_recording_is_recoverable(tmp_path: Path) -> None:
    recording = tmp_path / "recordings" / "good_night" / "one.wav"
    recording.parent.mkdir(parents=True)
    recording.write_bytes(b"audio")

    archived = archive_recording(
        tmp_path / "recordings", command_name="good_night", recording=recording
    )

    assert not recording.exists()
    assert archived.read_bytes() == b"audio"
    assert "_trash" in archived.parts


def test_archive_command_removes_config_and_preserves_material(tmp_path: Path) -> None:
    path = _config_copy(tmp_path)
    recordings = tmp_path / "recordings"
    sample = recordings / "good_night" / "one.wav"
    sample.parent.mkdir(parents=True)
    sample.write_bytes(b"audio")

    archive = archive_command(
        path, recordings, command_name="good_night"
    )

    assert "good_night" not in load_config(path).commands
    assert (archive / "recordings" / "one.wav").read_bytes() == b"audio"
    assert "[commands.good_night]" in (archive / "command.toml").read_text()
