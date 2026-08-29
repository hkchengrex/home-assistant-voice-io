from urllib.error import HTTPError
from urllib.request import Request, urlopen

from ha_voice.control_server import VoiceControlServer


def test_control_server_dispatches_configured_events_and_reports_health() -> None:
    events = []
    server = VoiceControlServer(
        "127.0.0.1",
        0,
        {"/arrive": lambda: events.append("arrive"), "/morning": lambda: events.append("morning")},
    )
    server.start()
    host, port = server.address
    try:
        with urlopen(f"http://{host}:{port}/health", timeout=2) as response:
            assert response.status == 200
            assert response.read() == b"ok\n"
        for route in ("/arrive", "/morning"):
            request = Request(f"http://{host}:{port}{route}", method="POST")
            with urlopen(request, timeout=2) as response:
                assert response.status == 202
    finally:
        server.stop()

    assert events == ["arrive", "morning"]


def test_control_server_rejects_unknown_path() -> None:
    server = VoiceControlServer("127.0.0.1", 0, {})
    server.start()
    host, port = server.address
    try:
        request = Request(f"http://{host}:{port}/other", method="POST")
        try:
            urlopen(request, timeout=2)
        except HTTPError as exc:
            assert exc.code == 404
        else:
            raise AssertionError("Unknown route should return 404")
    finally:
        server.stop()
