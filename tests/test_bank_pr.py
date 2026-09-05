from ha_voice.bank_pr import frontier, operating_points
import numpy as np
import pytest


def prediction(distance, margin, correct=True, negative=False):
    return dict(distance=distance, margin=margin, target=None if negative else 'a',
                predicted='a' if correct else 'b', represented=True)


def test_envelope_matches_brute_force_with_ties():
    rng = np.random.default_rng(14)
    rows = [prediction(float(rng.integers(1, 8)), float(rng.integers(-2, 8)) / 10,
                       bool(rng.integers(2)), bool(rng.integers(2))) for _ in range(30)]
    expected = {0: 0}
    for distance in sorted({p['distance'] for p in rows}):
        for margin in sorted({0.} | {p['margin'] for p in rows if p['margin'] >= 0}):
            accepted = [p for p in rows if p['distance'] <= distance and p['margin'] >= margin]
            tp = sum(p['target'] is not None and p['target'] == p['predicted'] for p in accepted)
            fp = len(accepted) - tp
            expected[tp] = min(expected.get(tp, 1000), fp)
    assert {p['tp']: p['fp'] for p in operating_points(rows)} == expected


def test_wrong_intent_and_negative_both_count_as_false_accepts():
    rows = [prediction(1, .5), prediction(1, .5, correct=False), prediction(1, .5, negative=True)]
    point = operating_points(rows)[-1]
    assert (point['tp'], point['fp'], point['recall'], point['precision']) == (1, 2, .5, 1 / 3)


def test_unsupported_and_nonfinite_positives_remain_in_denominator():
    rows = [prediction(1, .5), prediction(float('inf'), .5), prediction(1, .5, correct=False) | {'represented': False}]
    point = operating_points(rows)[-1]
    assert point['recall'] == 1 / 3
    assert point['fp'] == 1  # Unsupported target cannot be an oracle rejection rule.


def test_repeated_observations_are_counted_consistently():
    rows = [prediction(1, .5), prediction(2, .1, correct=False)] * 3
    point = operating_points(rows)[-1]
    assert (point['tp'], point['fp'], point['recall']) == (3, 0, .5)


def test_frontier_removes_dominated_points():
    points = [dict(recall=0., precision=1.), dict(recall=.1, precision=1.),
              dict(recall=.2, precision=.8), dict(recall=.3, precision=.9), dict(recall=.4, precision=.7)]
    assert frontier(points) == [points[i] for i in (0, 1, 3, 4)]


def test_no_positives_is_rejected():
    with pytest.raises(ValueError, match='positive'):
        operating_points([prediction(1, .5, negative=True)])
