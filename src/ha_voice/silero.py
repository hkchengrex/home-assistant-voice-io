"""CPU-only Silero ONNX streaming adapter; no torch or network access."""
from pathlib import Path
import hashlib
import numpy as np

MODEL_SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
MODEL_COMMIT = "867c2aa692646a1f1de3e94a15c9dd9f614c0acb"

class SileroDetector:
    """Consume 20 ms frames causally using complete 32 ms model windows.

    A held probability is used until another complete window is available.
    This adds up to 32 ms of model update delay, without reading future audio.
    """
    def __init__(self, model: Path, threshold: float = 0.5, release: float = 0.35):
        if not 0 <= release <= threshold <= 1:
            raise ValueError("Require 0 <= release <= threshold <= 1")
        if hashlib.sha256(Path(model).read_bytes()).hexdigest() != MODEL_SHA256:
            raise ValueError("Silero model hash does not match the pinned model")
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])
        self.threshold, self.release = threshold, release
        self.reset()

    def reset(self):
        self.state = np.zeros((2, 1, 128), np.float32)
        self.context = np.zeros((1, 64), np.float32)
        self.pending = np.empty(0, np.float32)
        self.probability = 0.0
        self.voiced = False

    def __call__(self, samples):
        samples = np.asarray(samples, dtype=np.float32)
        if samples.shape != (320,) or not np.isfinite(samples).all():
            raise ValueError("Expected one finite mono 20 ms frame at 16 kHz")
        self.pending = np.concatenate((self.pending, samples))
        while self.pending.size >= 512:
            chunk = self.pending[:512].reshape(1, 512)
            self.pending = self.pending[512:]
            inputs = np.concatenate((self.context, chunk), axis=1)
            output, self.state = self.session.run(None, {
                "input": inputs, "state": self.state, "sr": np.array(16000, dtype=np.int64)
            })
            self.context = chunk[:, -64:].copy()
            self.probability = float(output.item())
            self.voiced = self.probability >= (self.release if self.voiced else self.threshold)
        return self.voiced
