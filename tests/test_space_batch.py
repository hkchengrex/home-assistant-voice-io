"""Test the deployed batch core without loading a GPU model."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "space_batch_api", ROOT / "deploy/huggingface/omnivoice/batch_api.py"
)
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


class FakeModel:
    sampling_rate = 24000

    def __init__(self):
        self.prompts = []
        self.calls = []
        self.prompt = object()

    def create_voice_clone_prompt(self, **kwargs):
        self.prompts.append(kwargs)
        return self.prompt

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return [np.full(24000, float(text) / 10) for text in kwargs["text"]]


def run(model, texts, batch_size=4, **kwargs):
    return batch.generate_batch(
        model, SimpleNamespace, texts, "English", "reference.wav", " Transcript. ",
        "", 32, 2.0, True, 1.0, None, False, True, batch_size, **kwargs,
    )


def test_model_batches_share_one_prompt_and_preserve_order():
    model = FakeModel()
    audios, report = run(model, ["1", "2", "3", "4", "5"], batch_size=2)
    assert len(model.prompts) == 1
    assert model.prompts[0]["preprocess_prompt"] is False
    assert model.prompts[0]["ref_text"] == "Transcript."
    assert [c["text"] for c in model.calls] == [["1", "2"], ["3", "4"], ["5"]]
    assert all(c["voice_clone_prompt"] is model.prompt for c in model.calls)
    assert [round(float(a[0]) / 32767, 1) for a in audios] == [.1, .2, .3, .4, .5]
    assert report["status"] == "done"
    assert report["count"] == 5
    assert report["items"] == [{"index": i, "audio_seconds": 1.0} for i in range(5)]
    assert report["gpu_task_seconds"] >= report["generation_seconds"] >= 0


@pytest.mark.parametrize("texts", [[], "one", [None], [" "], ["x"] * 9, ["x" * 301], ["x" * 300] * 5])
def test_invalid_batch_never_touches_model(texts):
    model = FakeModel()
    with pytest.raises(ValueError):
        run(model, texts)
    assert model.prompts == model.calls == []


@pytest.mark.parametrize("size", [0, 9, 1.5, True, float("nan")])
def test_invalid_microbatch_size(size):
    with pytest.raises(ValueError, match="batch_size"):
        run(FakeModel(), ["1"], batch_size=size)


def test_audio_is_clipped_before_pcm_conversion():
    audios, _ = run(FakeModel(), ["20"])
    assert np.all(audios[0] == 32767)


def test_incomplete_model_result_fails():
    model = FakeModel()
    model.generate = lambda **kwargs: []
    with pytest.raises(RuntimeError, match="incomplete"):
        run(model, ["1"])


def test_gradio_batch_round_trip(tmp_path):
    gr = pytest.importorskip("gradio")
    sf = pytest.importorskip("soundfile")
    from gradio_client import Client, handle_file

    model = FakeModel()
    with gr.Blocks() as demo:
        text = gr.Textbox()
        gr.Button().click(lambda value: value, text, text, api_name="existing")

    def generate(*args):
        return batch.generate_batch(model, SimpleNamespace, *args)

    batch.attach_batch_api(demo, model, generate)
    assert all(fn.concurrency_id == "omnivoice-gpu" for fn in demo.fns.values())
    reference = tmp_path / "reference.wav"
    sf.write(reference, np.full(24000, .1), 24000, subtype="PCM_16")
    try:
        _, url, _ = demo.launch(prevent_thread_lock=True, quiet=True, server_name="127.0.0.1")
        client = Client(url, verbose=False)
        assert client.predict("still works", api_name="/existing") == "still works"
        paths, report = client.predict(
            ["1", "2"], "English", handle_file(str(reference)), "Reference.", "",
            32, 2.0, True, 1.0, None, True, True, 2, api_name="/clone_batch",
        )
        assert len(paths) == 2
        assert report["count"] == 2
        for path in paths:
            assert sf.info(path).subtype == "PCM_16"
        assert len(model.prompts) == 1
        assert model.calls[0]["text"] == ["1", "2"]
    finally:
        demo.close()
