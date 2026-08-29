import shutil

import pytest

from ha_voice.services import NoopListenerManager, SystemdUserServiceManager


def test_noop_manager_is_portable() -> None:
    manager = NoopListenerManager()
    assert manager.update(has_templates=True).updated is False
    assert manager.update(has_templates=False).waiting_for_templates is True
    assert manager.pause().paused is False


def test_systemd_manager_reports_unsupported_platform(monkeypatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="not available"):
        SystemdUserServiceManager("voice.service")
