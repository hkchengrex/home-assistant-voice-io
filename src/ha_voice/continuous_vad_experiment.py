"""Offline continuous-audio replay through the frozen wake and command gate.

No microphone, playback, private adapter, or action handler is used. The optional
mute intervals are measured playback intervals from the original capture, not
hypothetical candidate playback. Results are paired replay evidence, not a live
end-to-end benchmark.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .audio import load_wav
from .config import load_config
from .continuous import VoiceSegmenter
from .matcher import load_templates, load_start_phrase_templates
from .silero import SileroDetector, MODEL_SHA256, MODEL_COMMIT
from .start_phrase import StartPhraseGate
from .studio_server import match_features

class ReplayGate(StartPhraseGate):
    """Use audio time so an accelerated replay cannot extend command windows."""
    now = 0.0
    @property
    def phase(self):
        if self._command_deadline > self.now:
            return "awaiting_command"
        self._command_deadline = 0.0
        return "waiting_for_start"
    def arm_command_window(self):
        self._command_deadline = self.now + self.command_timeout_seconds


def make_gate(config, templates, wake_templates):
    wake = config.start_phrase
    return ReplayGate(
        sample_rate=config.recognizer.sample_rate, start_label=wake.name,
        display_name=wake.utterance, templates=wake_templates,
        max_distance=wake.max_distance, min_margin=wake.min_margin, top_k=wake.top_k,
        command_timeout_seconds=wake.command_timeout_seconds,
        min_audio_seconds=wake.min_audio_seconds, max_audio_seconds=wake.max_audio_seconds,
        min_command_audio_seconds=wake.min_command_audio_seconds,
        command_matcher=lambda features: match_features(features=features, config=config, templates=templates),
    )


def clean(result):
    keys = ("kind", "accepted", "command", "best_command", "score", "margin", "rejection_reason")
    return {key: result[key] for key in keys if key in result}


def evaluate(samples, backend, model, config, templates, wake_templates, mute, min_rms, multiplier):
    detector = SileroDetector(model) if backend == "silero" else None
    def segmenter(calibration=0):
        return VoiceSegmenter(min_rms=min_rms, noise_multiplier=multiplier,
                              calibration_ms=calibration, speech_detector=detector)
    segment = segmenter(1000)
    gate = make_gate(config, templates, wake_templates)
    wake_gate = make_gate(config, templates, wake_templates)
    events, frame_ms, probabilities = [], [], []
    was_muted = False
    pending_wake_refresh = False
    cpu_detector = 0.0
    for index in range(0, len(samples), 320):
        block = samples[index:index+320]
        if len(block) < 320:
            block = np.pad(block, (0, 320-len(block)))
        start, now = index/16000, (index+320)/16000
        gate.now = wake_gate.now = now
        muted = next((m for m in mute if start < m['end'] and now > m['start']), None)
        if muted is not None:
            if muted.get('kind') == 'wake' and gate.phase == 'awaiting_command':
                pending_wake_refresh = True
            was_muted = True
            continue
        if was_muted:
            old_noise = segment.noise_rms
            if detector:
                detector.reset()
            segment = segmenter()
            segment.noise_rms = old_noise
            if pending_wake_refresh:
                gate.arm_command_window()
            was_muted, pending_wake_refresh = False, False
        began, cpu_start = time.perf_counter(), time.process_time()
        utterances = segment.process(block)
        cpu_detector += time.process_time()-cpu_start
        frame_ms.append((time.perf_counter()-began)*1000)
        if detector:
            probabilities.append([round(now, 3), round(detector.probability, 5)])
        for utterance in utterances:
            began = time.perf_counter()
            wake_gate._command_deadline = 0.0
            if utterance.hit_duration_limit:
                wake = {"kind":"ignored", "accepted":False, "rejection_reason":"duration_limit"}
            else:
                try:
                    wake = clean(wake_gate(utterance.samples))
                except ValueError as exc:
                    wake = {"accepted":False, "rejection_reason":str(exc)}
            if utterance.hit_duration_limit and gate.phase == 'waiting_for_start':
                decision = {"kind":"ignored", "accepted":False, "rejection_reason":"duration_limit"}
            else:
                try:
                    decision = clean(gate(utterance.samples))
                except ValueError as exc:
                    decision = {"accepted":False, "rejection_reason":str(exc)}
            command = decision.get('command')
            if command in config.commands:
                decision['intent_group'] = config.commands[command].intent_group
            events.append({"emitted_at": round(now,3), "retained_seconds":len(utterance.samples)/16000,
                           "duration_limit":utterance.hit_duration_limit, "wake_only":wake,
                           "pipeline":decision, "matching_ms":(time.perf_counter()-began)*1000})
    return {"backend":backend, "segments":len(events),
            "duration_caps":sum(e['duration_limit'] for e in events),
            "wake_only_accepts":sum(bool(e['wake_only'].get('accepted')) for e in events),
            "pipeline_wakes":sum(e['pipeline'].get('kind')=='start_phrase' and e['pipeline'].get('accepted',False) for e in events),
            "pipeline_commands":sum(e['pipeline'].get('kind')=='command' and e['pipeline'].get('accepted',False) for e in events),
            "wake_rejection_reasons":dict(Counter(e['wake_only'].get('rejection_reason','distance_or_class') for e in events if not e['wake_only'].get('accepted'))),
            "detector_frame_ms_p50":float(np.median(frame_ms)) if frame_ms else None,
            "detector_frame_ms_p95":float(np.percentile(frame_ms,95)) if frame_ms else None,
            "detector_cpu_seconds_per_audio_second":cpu_detector/(len(samples)/16000),
            "events":events, "probabilities":probabilities}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audio',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--recordings',type=Path,required=True)
    p.add_argument('--mute-intervals',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--min-rms',type=float,default=0.004)
    p.add_argument('--noise-multiplier',type=float,default=3.0)
    args=p.parse_args()
    config=load_config(args.config)
    if config.recognizer.sample_rate != 16000:
        p.error('This experiment requires 16 kHz')
    audio=load_wav(args.audio,16000)
    if len(audio.samples)<16000:
        p.error('Use a continuous recording longer than one second')
    mute=json.loads(args.mute_intervals.read_text()) if args.mute_intervals else []
    for interval in mute:
        if not 0 <= interval['start'] < interval['end'] <= len(audio.samples)/16000:
            p.error('Mute interval outside recording')
    templates=load_templates(args.recordings)
    wake_templates=load_start_phrase_templates(args.recordings,start_phrase_name=config.start_phrase.name,
        negative_names={*config.commands,config.calibration.name})
    results=[evaluate(audio.samples,b,args.model,config,templates,wake_templates,mute,args.min_rms,args.noise_multiplier)
             for b in ('energy','silero')]
    import onnxruntime
    report={"audio_seconds":len(audio.samples)/16000,
            "audio_sha256":hashlib.sha256(args.audio.read_bytes()).hexdigest(),
            "config_sha256":hashlib.sha256(args.config.read_bytes()).hexdigest(),
            "template_hashes":{str(t.path.relative_to(args.recordings)):hashlib.sha256(t.path.read_bytes()).hexdigest() for t in templates+wake_templates},
            "onnxruntime":onnxruntime.__version__,"model_sha256":MODEL_SHA256,"model_commit":MODEL_COMMIT,
            "silero_threshold":0.5,"silero_release":0.35,"mute_intervals":mute,
            "min_rms":args.min_rms,"noise_multiplier":args.noise_multiplier,
            "limitations":["Replay uses audio time; no real actions or playback",
                "Captured speaker responses may contaminate results unless measured mute intervals are provided",
                "Measured playback masking follows the original listener, not hypothetical candidate timing",
                "Matching runs synchronously without modeling its queue delay",
                "No human speech-boundary annotations; processing time is not response latency",
                "A small prompted session is not a held-out general accuracy benchmark"],
            "results":results}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    for result in results:
        print(json.dumps({k:v for k,v in result.items() if k not in ('events','probabilities')},indent=2))
        for e in result['events']:
            if e['wake_only'].get('accepted') or e['pipeline'].get('accepted'):
                print(json.dumps(e))

if __name__=='__main__': main()
