"""Command-line interface for recording and matching voice templates."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

import numpy as np

from .actions import CommandHandler, no_action
from .audio import Audio, list_devices, load_wav, record_audio, save_wav
from .chime import play_confirmation_chime
from .config import AppConfig, load_config
from .continuous import ContinuousListener
from .control_server import VoiceControlServer
from .diagnostics import TriggerCaptureQueue
from .features import extract_command_features
from .matcher import (
    classify,
    leave_one_out_distances,
    load_start_phrase_templates,
    load_templates,
)
from .responses import VoiceResponsePlayer
from .services import SystemdUserServiceManager
from .start_phrase import StartPhraseGate


def _project_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ha-voice", description="Train and test local voice-command matching"
    )
    parser.add_argument("--config", type=_project_path, default=Path("commands.toml"))
    parser.add_argument(
        "--recordings", type=_project_path, default=Path("recordings")
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    subparsers.add_parser("devices", help="list audio input devices")

    record = subparsers.add_parser("record", help="record command templates")
    record.add_argument("command")
    record.add_argument("--count", type=int, default=10)
    record.add_argument("--seconds", type=float, default=2.5)
    record.add_argument("--device", default=None)
    record.add_argument("--output-device", default=None)
    record.add_argument(
        "--cued",
        action="store_true",
        help="play a chime before each take instead of waiting for Enter",
    )

    listen = subparsers.add_parser("listen", help="record and classify one command")
    listen.add_argument("--seconds", type=float, default=2.5)
    listen.add_argument("--device", default=None)

    match = subparsers.add_parser("match", help="classify an existing WAV file")
    match.add_argument("wav", type=_project_path)

    subparsers.add_parser("templates", help="show template counts")
    subparsers.add_parser("calibrate", help="measure genuine/impostor separation")

    studio = subparsers.add_parser("studio", help="open the local recording studio")
    studio.add_argument("--host", default="127.0.0.1")
    studio.add_argument("--port", type=int, default=8765)
    studio.add_argument(
        "--managed-service",
        default=None,
        help="optional user service restarted after library changes",
    )

    run = subparsers.add_parser(
        "run", help="listen continuously with the trained start-phrase gate"
    )
    run.add_argument("--input-device", default=None)
    run.add_argument("--output-device", default=None)
    run.add_argument("--min-rms", type=float, default=0.004)
    run.add_argument("--noise-multiplier", type=float, default=3.0)
    run.add_argument("--assets", type=_project_path, default=Path("assets"))
    run.add_argument(
        "--control-port",
        type=int,
        default=0,
        help="loopback HTTP port for local welcome events; zero disables it",
    )
    run.add_argument(
        "--direct",
        action="store_true",
        help="classify every utterance as a command without a start phrase",
    )
    run.add_argument(
        "--capture-command",
        default=None,
        help="save direct-test utterances as training examples for this command",
    )
    run.add_argument(
        "--capture-count",
        type=int,
        default=5,
        help="stop after this many direct-test training captures",
    )
    return parser


def _device(value: str | None) -> int | str | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return value


def _load_app_config(path: Path) -> AppConfig:
    try:
        return load_config(path)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Configuration error: {exc}") from exc


def _recording_utterance(name: str, config: AppConfig) -> str:
    if name == config.start_phrase.name:
        return config.start_phrase.utterance
    if name == config.calibration.name:
        return config.calibration.utterance
    command = config.commands.get(name)
    if command is None:
        choices = ", ".join(
            sorted({*config.commands, config.start_phrase.name, config.calibration.name})
        )
        raise SystemExit(f"Unknown recording set '{name}'. Choose: {choices}")
    return command.utterance


def _save_training_sample(
    samples: np.ndarray,
    *,
    sample_rate: int,
    command_name: str,
    recordings: Path,
) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = recordings / command_name / f"{stamp}.wav"
    save_wav(
        path,
        Audio(samples=np.asarray(samples, dtype=np.float32).copy(), sample_rate=sample_rate),
    )
    return path


def _print_result(result, config: AppConfig) -> int:
    ranking = sorted(result.per_command.items(), key=lambda item: item[1])
    for command, score in ranking:
        print(f"  {command:20} {score:.3f}")
    print(f"Margin: {result.margin:.1%}")
    if (
        not result.accepted
        or result.command is None
        or result.command not in config.commands
    ):
        print("REJECTED: no sufficiently confident command match")
        return 2
    command = config.commands.get(result.command)
    print(f"MATCH: {result.command} (distance {result.score:.3f})")
    print(f"INTENT: {command.intent_group if command else result.command}")
    return 0


def _classify_features(features: np.ndarray, config: AppConfig, recordings: Path) -> int:
    templates = load_templates(recordings, config.recognizer.sample_rate)
    if not templates:
        raise SystemExit("No recordings found. Use 'ha-voice record <command>' first.")
    result = classify(
        features,
        templates,
        max_distance=config.recognizer.max_distance,
        min_margin=config.recognizer.min_margin,
        top_k=config.recognizer.top_k,
    )
    return _print_result(result, config)


def _format_listener_result(result: dict[str, object], config: AppConfig) -> str | None:
    kind = result.get("kind")
    score = float(result.get("score", float("nan")))
    margin = float(result.get("margin", float("nan")))
    timing = result.get("match_seconds")
    timing_text = f" match={float(timing):.2f}s" if timing is not None else ""
    duration = result.get("audio_seconds")
    duration_text = f" audio={float(duration):.2f}s" if duration is not None else ""
    diagnostics = f"{timing_text}{duration_text}"
    captured_as = result.get("captured_as")
    capture_text = f" saved={captured_as}" if captured_as is not None else ""
    diagnostic_id = result.get("diagnostic_id")
    diagnostic_text = (
        f" diagnostic={diagnostic_id}" if diagnostic_id is not None else ""
    )
    if kind == "start_phrase":
        return (
            f"START: recognized score={score:.3f} margin={margin:.1%}"
            f"{diagnostics}{diagnostic_text}"
        )
    if kind == "ignored":
        if result.get("rejection_reason") == "duration_limit":
            return f"START: rejected reason=duration-limit{duration_text}"
        if result.get("rejection_reason") == "start_too_short":
            minimum = float(result.get("min_audio_seconds", float("nan")))
            return (
                f"START: rejected reason=too-short{duration_text} "
                f"min={minimum:.2f}s"
            )
        if result.get("rejection_reason") == "start_too_long":
            maximum = float(result.get("max_audio_seconds", float("nan")))
            return (
                f"START: rejected reason=too-long{duration_text} "
                f"max={maximum:.2f}s"
            )
        scores = result.get("scores")
        score_map = scores if isinstance(scores, dict) else {}
        start_score = float(score_map.get(config.start_phrase.name, float("nan")))
        rejection_score = float(score_map.get("_not_start_phrase", float("nan")))
        return (
            f"START: rejected start={start_score:.3f} "
            f"rejection={rejection_score:.3f} margin={margin:.1%}{diagnostics}"
        )
    if kind == "command":
        if result.get("rejection_reason") == "command_too_short":
            minimum = float(result.get("min_audio_seconds", float("nan")))
            return (
                f"COMMAND: rejected reason=too-short{duration_text} "
                f"min={minimum:.2f}s{diagnostic_text} (no action)"
            )
        if result.get("accepted"):
            action_status = result.get("action_status")
            if action_status == "sent":
                action_text = " action=sent"
            elif action_status == "failed":
                action_text = f" action=failed error={result.get('action_error')}"
            else:
                action_text = " (no action)"
            return (
                f"COMMAND: {result.get('utterance')} score={score:.3f} "
                f"margin={margin:.1%}{diagnostics}{capture_text}"
                f"{diagnostic_text}{action_text}"
            )
        scores = result.get("scores")
        score_map = scores if isinstance(scores, dict) else {}
        ranking = sorted(score_map.items(), key=lambda item: item[1])
        best_command = result.get("best_command")
        if not isinstance(best_command, str) and ranking:
            best_command = ranking[0][0]
        runner_command = result.get("runner_command")
        runner_score = result.get("runner_score")
        if not isinstance(runner_command, str) and len(ranking) > 1:
            runner_command, runner_score = ranking[1]
        best_text = f" best={best_command}" if isinstance(best_command, str) else ""
        runner_text = (
            f" runner={runner_command} runner_score={float(runner_score):.3f}"
            if isinstance(runner_command, str) and runner_score is not None
            else ""
        )
        return (
            f"COMMAND: rejected{best_text} score={score:.3f}{runner_text} "
            f"margin={margin:.1%}{diagnostics}{capture_text}"
            f"{diagnostic_text} (no action)"
        )
    return None


def _run_continuously(
    args: argparse.Namespace,
    config: AppConfig,
    recordings: Path,
    *,
    command_handler: CommandHandler | None = None,
) -> None:
    from .studio_server import match_features, match_samples

    if args.min_rms <= 0 or args.noise_multiplier <= 0:
        raise SystemExit("Voice detection values must be positive.")
    if not 0 <= args.control_port <= 65535:
        raise SystemExit("--control-port must be between 0 and 65535")
    if args.capture_command is not None:
        if not args.direct:
            raise SystemExit("--capture-command requires --direct")
        if args.capture_command not in config.commands:
            raise SystemExit(f"Unknown capture command '{args.capture_command}'")
        if args.capture_count < 1:
            raise SystemExit("--capture-count must be at least 1")
    command_templates = load_templates(recordings, config.recognizer.sample_rate)
    if not command_templates:
        raise SystemExit("No command recordings found.")
    listener = ContinuousListener(
        config.recognizer.sample_rate,
        heartbeat_timeout_seconds=None,
        min_rms=args.min_rms,
        noise_multiplier=args.noise_multiplier,
    )
    responses = (
        VoiceResponsePlayer(
            Path(args.assets).resolve(),
            config.responses,
            device=_device(args.output_device),
        )
        if config.responses
        else None
    )
    handler = command_handler or no_action

    def play_command(command_name: str) -> None:
        command = config.commands[command_name]
        if responses is not None and command.response:
            responses.play(command.response)
    captures_completed = 0
    if args.direct:
        def direct_matcher(samples: np.ndarray) -> dict[str, object]:
            nonlocal captures_completed
            result = match_samples(
                samples=samples,
                config=config,
                templates=command_templates,
            )
            result["kind"] = "command"
            if args.capture_command is not None:
                _save_training_sample(
                    samples,
                    sample_rate=config.recognizer.sample_rate,
                    command_name=args.capture_command,
                    recordings=recordings,
                )
                captures_completed += 1
                result["captured_as"] = args.capture_command
            return result

        listener.start(
            device=_device(args.input_device),
            matcher=direct_matcher,
            command_feedback=play_command if responses is not None else None,
        )
        print(
            "Listening directly for commands. Start phrase and actions are disabled.",
            flush=True,
        )
    else:
        if not config.start_phrase.enabled:
            raise SystemExit(
                "Enable [start_phrase] in commands.toml before continuous use."
            )
        start_count = len(
            list((recordings / config.start_phrase.name).glob("*.wav"))
        )
        if start_count < config.start_phrase.target_samples:
            raise SystemExit(
                f"Start phrase needs {config.start_phrase.target_samples} recordings; "
                f"found {start_count}."
            )
        start_templates = load_start_phrase_templates(
            recordings,
            start_phrase_name=config.start_phrase.name,
            negative_names={*config.commands, config.calibration.name},
            sample_rate=config.recognizer.sample_rate,
        )

        def command_matcher(features: np.ndarray) -> dict[str, object]:
            result = match_features(
                features=features,
                config=config,
                templates=command_templates,
            )
            command_name = result.get("command")
            if result.get("accepted") and isinstance(command_name, str):
                try:
                    outcome = handler(command_name)
                except RuntimeError as exc:
                    result["action_status"] = "failed"
                    result["action_error"] = str(exc)
                    result["suppress_feedback"] = True
                else:
                    result["action_status"] = outcome.status
                    if outcome.error:
                        result["action_error"] = outcome.error
                    command = config.commands[command_name]
                    if outcome.suppress_feedback or (
                        outcome.status == "unassigned" and not command.response
                    ):
                        result["suppress_feedback"] = True
            return result

        gate = StartPhraseGate(
            sample_rate=config.recognizer.sample_rate,
            start_label=config.start_phrase.name,
            display_name=config.start_phrase.utterance,
            templates=start_templates,
            max_distance=config.start_phrase.max_distance,
            min_margin=config.start_phrase.min_margin,
            top_k=config.start_phrase.top_k,
            command_timeout_seconds=config.start_phrase.command_timeout_seconds,
            min_audio_seconds=config.start_phrase.min_audio_seconds,
            max_audio_seconds=config.start_phrase.max_audio_seconds,
            min_command_audio_seconds=(
                config.start_phrase.min_command_audio_seconds
            ),
            command_matcher=command_matcher,
        )
        diagnostic_queue = TriggerCaptureQueue(
            recordings,
            sample_rate=config.recognizer.sample_rate,
        )
        def start_feedback() -> None:
            if responses is not None and config.start_phrase.response:
                responses.play(config.start_phrase.response)
            gate.arm_command_window()

        listener.start(
            device=_device(args.input_device),
            matcher=gate,
            phase_provider=lambda: gate.phase,
            start_phrase_feedback=start_feedback,
            command_feedback=play_command if responses is not None else None,
            capture_callback=diagnostic_queue.capture,
        )
        print(
            f"Listening for {config.start_phrase.utterance}. "
            "Configured command handler is active.",
            flush=True,
        )
    control_server: VoiceControlServer | None = None
    last_event_id = 0
    try:
        if args.control_port:
            callbacks = {}
            for route, response_group in config.control_events.items():
                def queue_response(group: str = response_group) -> None:
                    if responses is None:
                        raise RuntimeError("No response groups are configured")
                    listener.enqueue_feedback(lambda: responses.play(group))

                callbacks[route] = queue_response
            control_server = VoiceControlServer(
                "127.0.0.1",
                args.control_port,
                callbacks,
            )
            control_server.start()
            print(
                f"Local voice control listening on 127.0.0.1:{args.control_port}.",
                flush=True,
            )
        while True:
            snapshot = listener.snapshot()
            if not snapshot["running"]:
                raise RuntimeError(snapshot["error"] or "Listener stopped unexpectedly")
            if snapshot["event_id"] > last_event_id:
                last_event_id = snapshot["event_id"]
                result = snapshot["last_result"] or {}
                message = _format_listener_result(result, config)
                if message:
                    print(message, flush=True)
                if (
                    args.capture_command is not None
                    and captures_completed >= args.capture_count
                ):
                    print(
                        f"Captured {captures_completed} examples for "
                        f"{args.capture_command}; stopping.",
                        flush=True,
                    )
                    break
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if control_server is not None:
            control_server.stop()
        listener.stop()


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values), percentile))


def main(
    argv: list[str] | None = None,
    *,
    command_handler: CommandHandler | None = None,
) -> None:
    args = _parser().parse_args(argv)
    config_path = Path(args.config).resolve()
    recordings = Path(args.recordings).resolve()

    if args.subcommand == "devices":
        for index, device in enumerate(list_devices()):
            inputs = int(device.get("max_input_channels", 0))
            if inputs:
                print(f"{index:3}: {device.get('name')} ({inputs} input channels)")
        return

    if args.subcommand == "studio":
        from .studio_server import serve_studio

        try:
            listener_manager = (
                SystemdUserServiceManager(args.managed_service)
                if args.managed_service
                else None
            )
        except (ValueError, RuntimeError) as exc:
            raise SystemExit(str(exc)) from exc
        serve_studio(
            config_path=config_path,
            recordings_dir=recordings,
            host=args.host,
            port=args.port,
            listener_manager=listener_manager,
        )
        return

    config = _load_app_config(config_path)

    if args.subcommand == "run":
        _run_continuously(
            args,
            config,
            recordings,
            command_handler=command_handler,
        )
        return

    if args.subcommand == "record":
        utterance = _recording_utterance(args.command, config)
        if args.count < 1:
            raise SystemExit("--count must be at least 1")
        command_dir = recordings / args.command
        for number in range(1, args.count + 1):
            print(f"\nTemplate {number}/{args.count}: say “{utterance}”")
            if args.cued:
                play_confirmation_chime(
                    sample_rate=config.recognizer.sample_rate,
                    device=_device(args.output_device),
                )
            else:
                input("Press Enter when ready...")
            audio = record_audio(
                args.seconds,
                config.recognizer.sample_rate,
                _device(args.device),
                countdown=0,
            )
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            path = command_dir / f"{stamp}.wav"
            save_wav(path, audio)
            print(f"Saved {path}")
        return

    if args.subcommand == "listen":
        audio = record_audio(
            args.seconds,
            config.recognizer.sample_rate,
            _device(args.device),
        )
        features = extract_command_features(audio.samples, audio.sample_rate)
        raise SystemExit(_classify_features(features, config, recordings))

    if args.subcommand == "match":
        audio = load_wav(args.wav, config.recognizer.sample_rate)
        features = extract_command_features(audio.samples, audio.sample_rate)
        raise SystemExit(_classify_features(features, config, recordings))

    templates = load_templates(recordings, config.recognizer.sample_rate)
    if args.subcommand == "templates":
        counts: dict[str, int] = {}
        for template in templates:
            counts[template.command] = counts.get(template.command, 0) + 1
        for command in sorted(config.commands):
            print(f"{command:20} {counts.get(command, 0):3}")
        return

    if args.subcommand == "calibrate":
        if len({template.command for template in templates}) < 2:
            raise SystemExit("Calibration needs at least two trained commands.")
        genuine, impostor = leave_one_out_distances(templates)
        if not genuine or not impostor:
            raise SystemExit("Record at least two examples for every trained command.")
        genuine_p95 = _percentile(genuine, 95)
        impostor_p05 = _percentile(impostor, 5)
        suggested = (genuine_p95 + impostor_p05) / 2.0
        print(f"Same-command distance p50/p95: {_percentile(genuine, 50):.3f} / {genuine_p95:.3f}")
        print(f"Other-command distance p05/p50: {impostor_p05:.3f} / {_percentile(impostor, 50):.3f}")
        print(f"Suggested max_distance midpoint: {suggested:.3f}")
        if genuine_p95 >= impostor_p05:
            print("WARNING: command classes overlap; add varied recordings or distinct phrases.")
        else:
            separation = (impostor_p05 - genuine_p95) / impostor_p05
            print(f"Conservative separation: {separation:.1%}")
        return

    raise AssertionError(f"Unhandled command: {args.subcommand}")


if __name__ == "__main__":
    main(sys.argv[1:])
