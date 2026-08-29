import numpy as np

from ha_voice.asset_processing import publish_asset_library, trim_and_normalize
from ha_voice.audio import Audio, load_wav, save_wav


def test_trim_and_normalize_preserves_internal_audio_end_and_channels() -> None:
    sample_rate = 24000
    time = np.arange(sample_rate // 2, dtype=np.float32) / sample_rate
    speech = 0.05 * np.sin(2 * np.pi * 180 * time)
    stereo_speech = np.column_stack((speech, speech * 0.8))
    # Low room noise must not keep otherwise silent edges in the output.
    noise_time = np.arange(sample_rate // 2, dtype=np.float32) / sample_rate
    noise = 0.004 * np.sin(2 * np.pi * 73 * noise_time)
    silence = np.column_stack((noise, noise * 0.8))
    samples = np.concatenate((silence, stereo_speech, silence))
    samples[round(sample_rate * 0.1)] = 0.2

    result = trim_and_normalize(samples, sample_rate)

    assert result.samples.ndim == 2
    assert result.samples.shape[1] == 2
    assert 0.99 <= result.output_seconds <= 1.01
    assert result.trim_start_seconds >= 0.49
    assert result.trim_end_seconds == 0.0
    assert result.samples.shape[0] == samples.shape[0] - round(
        result.trim_start_seconds * sample_rate
    )
    assert np.isclose(result.speech_rms_dbfs, -18.0, atol=0.1)
    assert result.output_peak_dbfs <= -1.0


def test_trim_and_normalize_respects_peak_ceiling() -> None:
    samples = np.full(8000, 0.001, dtype=np.float32)
    samples[2000:6000] = 0.02
    samples[4000] = 0.9

    result = trim_and_normalize(samples, 16000)

    assert result.output_peak_dbfs <= -1.0
    assert result.gain_db < 0.0


def test_publish_asset_library_keeps_sources_and_replaces_flat_group(tmp_path) -> None:
    sample_rate = 24000
    source_dir = tmp_path / "bye"
    time = np.arange(sample_rate // 2, dtype=np.float32) / sample_rate
    speech = 0.05 * np.sin(2 * np.pi * 180 * time)
    trailing = np.full(sample_rate // 4, 0.0005, dtype=np.float32)
    samples = np.concatenate((np.zeros(sample_rate // 4), speech, trailing))
    first = source_dir / "audio.wav"
    second = source_dir / "audio (1).wav"
    save_wav(first, Audio(samples, sample_rate))
    save_wav(second, Audio(samples, sample_rate))
    stale = tmp_path / "bye_3.wav"
    save_wav(stale, Audio(samples, sample_rate))
    reference_dir = tmp_path / "reference"
    reference = reference_dir / "reference_1.wav"
    save_wav(reference, Audio(samples, sample_rate))

    published = publish_asset_library(tmp_path, prefixes={"bye"})

    assert [path.name for path, _ in published] == ["bye_1.wav", "bye_2.wav"]
    assert first.exists() and second.exists()
    assert reference.exists()
    assert not (tmp_path / "reference_1.wav").exists()
    assert not stale.exists()
    output = load_wav(tmp_path / "bye_1.wav", sample_rate)
    first_result = published[0][1]
    assert output.samples.size == samples.size - round(
        first_result.trim_start_seconds * sample_rate
    )
    assert np.any(output.samples[-trailing.size :])
