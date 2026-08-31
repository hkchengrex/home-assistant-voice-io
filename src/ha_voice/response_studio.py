"""Persistent local phrase lists, generation jobs, and reviewed response takes."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, fields
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import re
import tempfile
import threading
from uuid import uuid4
import wave

from .asset_processing import process_wav
from .config import AppConfig
from .reference_audio import normalize_reference_audio
from .voice_generation import CloneSettings, HuggingFaceSpaceVoiceCloner, _atomic_copy


def _now():
    return datetime.now(timezone.utc).isoformat()


def _identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{32}", value):
        raise ValueError("Invalid item identifier")
    return value


def _integer(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer between {low} and {high}")
    return value


def validate_settings(raw):
    if not isinstance(raw, dict) or set(raw) - {f.name for f in fields(CloneSettings)}:
        raise ValueError("Unsupported voice settings")
    settings = CloneSettings(**raw)
    _integer(settings.inference_steps, "Inference steps", 4, 64)
    for name, value, low, high in (
        ("CFG scale", settings.guidance_scale, 0, 4),
        ("Speed", settings.speed, .5, 1.5),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{name} must be between {low} and {high}")
    if settings.duration is not None and (
        isinstance(settings.duration, bool) or not isinstance(settings.duration, (int, float))
        or not math.isfinite(settings.duration) or not 0 < settings.duration <= 15
    ):
        raise ValueError("Duration must be empty or between 0 and 15 seconds")
    for name in ("denoise", "preprocess_prompt", "postprocess_output"):
        if not isinstance(getattr(settings, name), bool):
            raise ValueError(f"{name} must be a checkbox value")
    for name, limit in (("language", 80), ("reference_text", 4000), ("instruct", 500)):
        value = getattr(settings, name)
        if not isinstance(value, str) or len(value) > limit:
            raise ValueError(f"{name} is too long or not text")
    return settings


class ResponseStudio:
    """One local writer; jobs survive page reloads without ever auto-replaying."""

    def __init__(self, config: AppConfig, directory: Path, assets_dir: Path, *, cloner_factory=None):
        self.config = config
        self.directory = directory.resolve()
        self.assets_dir = assets_dir.resolve()
        self.lock = threading.RLock()
        self.cloner_factory = cloner_factory or (lambda: HuggingFaceSpaceVoiceCloner(max_attempts=1))
        self.thread = None
        self.closed = False
        self.path = self.directory / "workspace.json"
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
            if self.data.get("version") != 1:
                raise ValueError("Unsupported response workspace version; existing data was not changed")
            for job in self.data["jobs"]:
                if job["status"] in {"queued", "running"}:
                    job.update(status="interrupted", error="Studio restarted. Completed takes are saved; retry remaining takes explicitly.")
        else:
            examples = json.loads(Path(__file__).with_name("response_examples.json").read_text(encoding="utf-8"))
            self.data = {"version": 1, "phrases": [], "candidates": [], "jobs": [],
                         "settings": asdict(CloneSettings()), "reference": None}
            for group in config.responses:
                for text in examples.get(group, []):
                    self.data["phrases"].append({"id": uuid4().hex, "group": group, "text": text, "archived": False})
        self._save()

    def _save(self):
        encoded = json.dumps(self.data, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
        with tempfile.NamedTemporaryFile(dir=self.directory, suffix=".json", delete=False) as output:
            temporary = Path(output.name)
        try:
            temporary.write_bytes(encoded)
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)

    @contextmanager
    def _edit(self):
        with self.lock:
            before = deepcopy(self.data)
            try:
                yield
                self._save()
            except Exception:
                self.data = before
                raise

    def _find(self, collection, item_id):
        _identifier(item_id)
        for item in self.data[collection]:
            if item["id"] == item_id:
                return item
        raise ValueError("Item not found")

    def _group(self, name):
        if not isinstance(name, str) or name not in self.config.responses:
            raise ValueError("Choose a configured response group")
        return self.config.responses[name]

    def snapshot(self):
        with self.lock:
            result = deepcopy(self.data)
            result["groups"] = [
                {"name": name, "prefix": prefix,
                 "commands": [{"name": c.name, "utterance": c.utterance}
                              for c in self.config.commands.values() if c.response == name]}
                for name, prefix in self.config.responses.items()
            ]
            for item in result["candidates"]:
                item["audio_url"] = f"/api/responses/audio/{item['id']}"
            if result["reference"]:
                result["reference"]["audio_url"] = "/api/responses/reference"
            result["busy"] = any(j["status"] in {"queued", "running"} for j in result["jobs"])
            return result

    def save_phrase(self, payload):
        group = payload.get("group")
        self._group(group)
        text = payload.get("text")
        if not isinstance(text, str) or not text.strip() or len(text.strip()) > 300:
            raise ValueError("Enter a phrase of 1–300 characters")
        with self._edit():
            if payload.get("id"):
                phrase = self._find("phrases", payload["id"])
                if phrase["group"] != group:
                    raise ValueError("A phrase cannot be moved between groups")
                phrase["text"] = text.strip()
            else:
                if len(self.data["phrases"]) >= 1000:
                    raise ValueError("The workspace already has 1000 phrases")
                phrase = {"id": uuid4().hex, "group": group, "text": text.strip(), "archived": False}
                self.data["phrases"].append(phrase)
            return deepcopy(phrase)

    def archive_phrase(self, item_id, archived):
        if not isinstance(archived, bool):
            raise ValueError("Invalid archive choice")
        with self._edit():
            self._find("phrases", item_id)["archived"] = archived

    def save_settings(self, raw):
        settings = validate_settings(raw)
        with self._edit():
            self.data["settings"] = asdict(settings)

    def upload_reference(self, body):
        body = normalize_reference_audio(body)
        with wave.open(BytesIO(body), "rb") as audio:
            rate, count = audio.getframerate(), audio.getnframes()
        ref = {"id": uuid4().hex, "seconds": count / rate, "sample_rate": rate, "created_at": _now()}
        path = self.directory / "references" / f"{ref['id']}.wav"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(body)
        try:
            with self._edit():
                self.data["reference"] = ref
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return ref

    def audio_path(self, item_id=None):
        with self.lock:
            if item_id is None:
                ref = self.data["reference"]
                if not ref:
                    raise ValueError("Upload a reference first")
                return self.directory / "references" / f"{_identifier(ref['id'])}.wav"
            self._find("candidates", item_id)
            return self.directory / "candidates" / f"{_identifier(item_id)}.wav"

    def _ids(self, payload, key, maximum=64):
        ids = payload.get(key)
        if not isinstance(ids, list) or not 1 <= len(ids) <= maximum:
            raise ValueError(f"Select 1–{maximum} items")
        for item_id in ids:
            _identifier(item_id)
        return list(dict.fromkeys(ids))

    def generate(self, payload):
        if payload.get("consent_to_upload") is not True:
            raise ValueError("Confirm permission to send the reference and text to Hugging Face")
        request_id = _identifier(payload.get("request_id"))
        with self.lock:
            previous = next((j for j in self.data["jobs"] if j["request_id"] == request_id), None)
            if previous:
                return deepcopy(previous)
            settings = validate_settings(payload.get("settings", self.data["settings"]))
            size = _integer(payload.get("batch_size", 4), "GPU batch size", 1, 8)
            if payload.get("retry_job_id"):
                old = self._find("jobs", payload["retry_job_id"])
                if old["status"] not in {"failed", "interrupted", "cancelled"}:
                    raise ValueError("Only unfinished jobs can be retried")
                if old.get("retried_by"):
                    raise ValueError("This job was already retried; use its latest retry")
                entries = deepcopy(old["entries"][old["completed"]:])
                settings = validate_settings(old["settings"])
                reference = deepcopy(old["reference"])
                size = old["batch_size"]
            elif payload.get("candidate_ids") is not None:
                entries = []
                for item_id in self._ids(payload, "candidate_ids"):
                    candidate = self._find("candidates", item_id)
                    if candidate["status"] != "rejected" or candidate.get("replacement_id"):
                        raise ValueError("Select rejected takes that have not been regenerated")
                    self._group(candidate["group"])
                    entries.append({"phrase_id": candidate["phrase_id"], "group": candidate["group"],
                                    "prefix": self._group(candidate["group"]), "text": candidate["text"], "replaces": item_id})
                reference = deepcopy(self.data["reference"])
            else:
                takes = _integer(payload.get("takes", 1), "Takes per phrase", 1, 8)
                entries = []
                for item_id in self._ids(payload, "phrase_ids", 32):
                    phrase = self._find("phrases", item_id)
                    if phrase["archived"]:
                        raise ValueError("Restore the phrase before generating it")
                    prefix = self._group(phrase["group"])
                    entries.extend({"phrase_id": item_id, "group": phrase["group"], "prefix": prefix,
                                    "text": phrase["text"], "replaces": None} for _ in range(takes))
                reference = deepcopy(self.data["reference"])
            if not entries or len(entries) > 64:
                raise ValueError("Generate 1–64 takes per job")
            if not reference:
                raise ValueError("Upload a reference voice first")
            if self.closed or any(j["status"] in {"queued", "running"} for j in self.data["jobs"]):
                raise ValueError("A generation job is already running; wait or stop it first")
            job = {"id": uuid4().hex, "request_id": request_id, "entries": entries,
                   "settings": asdict(settings), "reference": reference, "batch_size": size,
                   "status": "queued", "completed": 0, "total": len(entries),
                   "created_at": _now(), "error": "", "cancel_requested": False}
            with self._edit():
                self.data["settings"] = asdict(settings)
                self.data["jobs"].append(job)
                if payload.get("retry_job_id"):
                    self._find("jobs", payload["retry_job_id"])["retried_by"] = job["id"]
            self.thread = threading.Thread(target=self._run, args=(job["id"],), daemon=True)
            self.thread.start()
            return deepcopy(job)

    def _run(self, job_id):
        try:
            with self._edit():
                job = self._find("jobs", job_id)
                job["status"] = "running"
            cloner = self.cloner_factory()
            while True:
                with self._edit():
                    job = self._find("jobs", job_id)
                    if job["completed"] == job["total"]:
                        job.update(status="completed", finished_at=_now())
                        return
                    if job["cancel_requested"] or self.closed:
                        job.update(status="cancelled", finished_at=_now())
                        return
                    chunk = []
                    for entry in job["entries"][job["completed"]:]:
                        if len(chunk) == 8 or sum(len(e["text"]) for e in chunk) + len(entry["text"]) > 1200:
                            break
                        chunk.append(deepcopy(entry))
                    settings = validate_settings(job["settings"])
                    reference = self.directory / "references" / f"{_identifier(job['reference']['id'])}.wav"
                    size = job["batch_size"]
                generated = cloner.generate_batch(texts=[e["text"] for e in chunk], reference_audio=reference,
                                                  settings=settings, batch_size=size, consent_to_upload=True)
                if len(generated.audio_paths) != len(chunk):
                    raise ValueError("The Space returned an incomplete batch")
                created = []
                try:
                    for entry, source in zip(chunk, generated.audio_paths):
                        candidate = {**entry, "id": uuid4().hex, "job_id": job_id, "status": "pending",
                                     "created_at": _now(), "settings": asdict(settings), "published": False,
                                     "reference_id": job["reference"]["id"]}
                        path = self.directory / "candidates" / f"{candidate['id']}.wav"
                        path.parent.mkdir(exist_ok=True)
                        created.append((candidate, path))
                        processed = process_wav(Path(source), path)
                        candidate["seconds"] = processed.output_seconds
                    with self._edit():
                        job = self._find("jobs", job_id)
                        for candidate, _ in created:
                            self.data["candidates"].append(candidate)
                            if candidate["replaces"]:
                                self._find("candidates", candidate["replaces"])["replacement_id"] = candidate["id"]
                        job["completed"] += len(created)
                except Exception:
                    for _, path in created:
                        path.unlink(missing_ok=True)
                    raise
        except Exception as exc:
            with self._edit():
                self._find("jobs", job_id).update(status="failed", error=" ".join(str(exc).split())[:500], finished_at=_now())

    def cancel(self, job_id):
        with self._edit():
            self._find("jobs", job_id)["cancel_requested"] = True

    def review(self, payload):
        status = payload.get("status")
        if status not in {"pending", "kept", "rejected"}:
            raise ValueError("Choose pending, kept, or rejected")
        with self._edit():
            candidates = [self._find("candidates", item_id) for item_id in self._ids(payload, "candidate_ids")]
            if any(c["published"] for c in candidates):
                raise ValueError("Remove the take from playback before changing its review")
            for candidate in candidates:
                candidate["status"] = status

    def publish(self, payload):
        created = []
        try:
            with self._edit():
                candidates = [self._find("candidates", item_id) for item_id in self._ids(payload, "candidate_ids")]
                for candidate in candidates:
                    if candidate["status"] != "kept":
                        raise ValueError("Only kept takes can be published")
                    if candidate["published"]:
                        continue
                    prefix = self._group(candidate["group"])
                    if prefix != candidate["prefix"]:
                        raise ValueError("The response configuration changed; generate a new take")
                    filename = f"{prefix}_studio_{_identifier(candidate['id'])}.wav"
                    destination = self.assets_dir / filename
                    if destination.exists():
                        raise ValueError("A playback file already exists; it was not overwritten")
                    _atomic_copy(self.audio_path(candidate["id"]), destination)
                    created.append(destination)
                    candidate.update(published=True, published_file=filename,
                                     published_sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
        except Exception:
            for path in created:
                path.unlink(missing_ok=True)
            raise

    def unpublish(self, item_id):
        with self.lock:
            candidate = self._find("candidates", item_id)
            if not candidate["published"]:
                return
            filename = candidate["published_file"]
            if Path(filename).name != filename:
                raise ValueError("Invalid published filename")
            path = self.assets_dir / filename
            if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() != candidate["published_sha256"]:
                raise ValueError("The playback file changed outside Studio; it was not removed")
            backup = self.directory / "unpublished" / filename
            if path.exists():
                backup.parent.mkdir(exist_ok=True)
                _atomic_copy(path, backup)
                path.unlink()
            try:
                with self._edit():
                    self._find("candidates", item_id)["published"] = False
            except Exception:
                if backup.exists():
                    _atomic_copy(backup, path)
                raise

    def close(self):
        with self.lock:
            self.closed = True
