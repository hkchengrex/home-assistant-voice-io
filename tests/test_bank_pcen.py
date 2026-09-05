import numpy as np
import pytest

from ha_voice.bank_pcen import BASELINE, candidates, extract, selection_key
from ha_voice.features import extract_command_features, extract_pcen_cepstra


def test_grid_is_unique_and_keeps_baseline_first():
    grid = candidates()
    assert grid[0] == ('baseline', BASELINE)
    assert len(grid) == len({tuple(p.items()) for _, p in grid}) == 23


@pytest.mark.parametrize('samples', [np.zeros(100, dtype=np.float32),
                                    np.random.default_rng(42).normal(0, .1, 8000).astype(np.float32)])
def test_all_candidates_are_finite_and_baseline_matches_runtime(samples):
    np.testing.assert_array_equal(extract(samples, BASELINE), extract_command_features(samples))
    for _, params in candidates():
        output = extract(samples, params)
        assert output.shape[1] == 26
        assert np.isfinite(output).all()


@pytest.mark.parametrize('params', [dict(alpha=-.1), dict(alpha=np.nan),
                                   dict(delta=-1), dict(delta=np.inf),
                                   dict(root=0), dict(root=1.1),
                                   dict(epsilon=0), dict(epsilon=np.nan)])
def test_invalid_pcen_parameters(params):
    with pytest.raises(ValueError):
        extract_pcen_cepstra(np.ones(400, dtype=np.float32), **params)


def test_selection_prioritizes_wrong_accepts_over_recall():
    assert selection_key(dict(wrong_accepts=0, correct_accepts=10)) > selection_key(
        dict(wrong_accepts=1, correct_accepts=100))
