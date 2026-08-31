"""Same-origin HTTP routes for the local response-generation workspace."""

from http import HTTPStatus
import json
from urllib.parse import urlparse


def handle_response_request(handler, path, method):
    if not path.startswith("/api/responses/"):
        return False
    host = handler.headers.get("Host", "")
    allowed = {"localhost", "127.0.0.1", "::1", handler.server.server_address[0], handler.connection.getsockname()[0]}
    origin = handler.headers.get("Origin")
    try:
        hostname = urlparse("http://" + host).hostname
        valid_origin = not origin or urlparse(origin).netloc == host
    except ValueError:
        hostname, valid_origin = None, False
    if hostname not in allowed or not valid_origin:
        handler._send_json(HTTPStatus.FORBIDDEN, {"error": "Open this workspace directly from its local Studio address"})
        return True
    if method != "GET" and handler.headers.get("X-Voice-IO") != "response-studio":
        handler._send_json(HTTPStatus.FORBIDDEN, {"error": "This action requires the Studio interface"})
        return True
    try:
        studio = handler.server.response_workspace()
        action = path[len("/api/responses/"):]
        if method == "GET":
            if action == "state":
                handler._send_json(HTTPStatus.OK, studio.snapshot())
            elif action == "reference":
                handler._send_wav(studio.audio_path().read_bytes())
            elif action.startswith("audio/"):
                handler._send_wav(studio.audio_path(action[len("audio/"):]).read_bytes())
            else:
                handler._send_json(HTTPStatus.NOT_FOUND, {"error": "Unknown response route"})
            return True
        if method != "POST":
            handler._send_json(HTTPStatus.METHOD_NOT_ALLOWED, {"error": "Unsupported method"})
            return True
        size = int(handler.headers.get("Content-Length", "0"))
        limit = 10 * 1024 * 1024 if action == "reference" else 65536
        if not 0 < size <= limit:
            raise ValueError("Request is empty or too large")
        body = handler.rfile.read(size)
        if len(body) != size:
            raise ValueError("Incomplete request")
        if action == "reference":
            studio.upload_reference(body)
        else:
            if handler.headers.get_content_type() != "application/json":
                raise ValueError("Use a JSON request")
            payload = json.loads(body)
            if not isinstance(payload, dict):
                raise ValueError("Use a JSON object")
            if action == "phrases":
                studio.save_phrase(payload)
            elif action == "archive":
                studio.archive_phrase(payload.get("id"), payload.get("archived"))
            elif action == "settings":
                studio.save_settings(payload.get("settings"))
            elif action == "generate":
                job = studio.generate(payload)
                handler._send_json(HTTPStatus.ACCEPTED, {"job": job})
                return True
            elif action == "cancel":
                studio.cancel(payload.get("job_id"))
            elif action == "review":
                studio.review(payload)
            elif action in {"publish", "unpublish"}:
                if action == "publish":
                    studio.publish(payload)
                else:
                    studio.unpublish(payload.get("id"))
                # The live listener preloads responses, so refresh it explicitly.
                handler._send_json(HTTPStatus.OK, {"state": studio.snapshot(), **handler.server.update_managed_listener()})
                return True
            else:
                handler._send_json(HTTPStatus.NOT_FOUND, {"error": "Unknown response action"})
                return True
        handler._send_json(HTTPStatus.OK, {"state": studio.snapshot()})
    except (ValueError, TypeError, KeyError) as exc:
        handler._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
    except FileNotFoundError:
        handler._send_json(HTTPStatus.NOT_FOUND, {"error": "The audio file is unavailable"})
    except OSError:
        handler._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "Local files could not be updated; check disk space and permissions"})
    return True
