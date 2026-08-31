"""Keep package metadata, commands, and portable templates under one name."""

from importlib import metadata
from pathlib import Path
import plistlib
import tomllib

import ha_voice
from ha_voice.cli import _parser


ROOT = Path(__file__).resolve().parents[1]


def test_installed_package_identity() -> None:
    distribution = metadata.distribution("voice-io")
    assert distribution.metadata["Name"] == "voice-io"
    assert ha_voice.__version__ == distribution.version
    assert {
        item.name: item.value
        for item in distribution.entry_points
        if item.group == "console_scripts"
    } == {"voice-io": "ha_voice.cli:main"}


def test_cli_uses_project_name() -> None:
    help_text = _parser().format_help()
    assert help_text.startswith("usage: voice-io ")
    assert "Home Assistant Voice IO" in help_text


def test_portable_service_template_names() -> None:
    linux = (ROOT / "deploy/systemd/voice-io.service").read_text()
    assert "Description=Home Assistant Voice IO" in linux
    assert "/.venv/bin/voice-io " in linux
    with (ROOT / "deploy/launchd/io.havoice.voiceio.plist").open("rb") as handle:
        macos = plistlib.load(handle)
    assert macos["Label"] == "io.havoice.voiceio"
    assert macos["ProgramArguments"][0] == "REPLACE_WITH_VENV/voice-io"
    windows = (ROOT / "deploy/windows/run-voice-io.ps1").read_text()
    assert "voice-io.exe" in windows
    assert '"HAVoiceIO"' in windows


def test_declared_package_identity() -> None:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)["project"]
    assert project["name"] == "voice-io"
    assert project["scripts"] == {"voice-io": "ha_voice.cli:main"}


def test_studio_displays_short_project_name() -> None:
    studio = (ROOT / "src/ha_voice/studio/index.html").read_text(encoding="utf-8")
    assert "<title>HA Voice IO Studio</title>" in studio
    assert '<span>HA Voice IO</span>' in studio
