import numpy as np
import pytest

from ha_voice.audio import Audio, load_wav, save_wav, trim_silence
from ha_voice.config import RecognizerConfig, load_config
from ha_voice.features import extract_command_features
from ha_voice.matcher import load_templates


@pytest.mark.parametrize('field,value', [('pcen_smoothing', 0), ('pcen_smoothing', 2),
                                       ('pcen_alpha', -1), ('pcen_alpha', 2)])
def test_invalid_config(tmp_path, field, value):
    path = tmp_path / 'commands.toml'
    path.write_text(f'[recognizer]\n{field} = {value}\n')
    with pytest.raises(ValueError, match=field):
        load_config(path)


def test_configured_templates_match_query_features(tmp_path):
    options = RecognizerConfig(pcen_smoothing=.1, pcen_alpha=.95).pcen_options
    samples = np.random.default_rng(3).normal(0, .1, 8000).astype(np.float32)
    for label in ('example', '_not_command'):
        path = tmp_path / label / 'sample.wav'
        path.parent.mkdir()
        save_wav(path, Audio(samples, 16000))
    templates = load_templates(tmp_path, **options)
    assert len(templates) == 2
    for template in templates:
        audio = load_wav(template.path)
        trimmed = trim_silence(audio.samples, audio.sample_rate)
        np.testing.assert_array_equal(template.features, extract_command_features(trimmed, **options))
        assert not np.array_equal(template.features, extract_command_features(trimmed))
