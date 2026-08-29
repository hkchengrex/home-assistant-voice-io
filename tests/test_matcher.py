from pathlib import Path

import numpy as np
import pytest

import ha_voice.matcher as matcher
from ha_voice.matcher import (
    Template,
    classify,
    dtw_distance,
    load_start_phrase_templates,
    load_templates,
    shortlist_templates,
)
from ha_voice.audio import Audio, save_wav


def _features(seed: int, frames: int = 30) -> np.ndarray:
    random = np.random.default_rng(seed)
    base = np.linspace(-1, 1, frames, dtype=np.float32)[:, None]
    return np.concatenate((base, base * 0.5), axis=1) + random.normal(
        0, 0.02, (frames, 2)
    )


def _reference_dtw_distance(
    first: np.ndarray, second: np.ndarray, band_ratio: float = 0.25
) -> float:
    """Straightforward reference used to guard optimized DTW behavior."""
    rows, columns = first.shape[0], second.shape[0]
    band = max(abs(rows - columns), int(max(rows, columns) * band_ratio), 2)
    previous_cost = np.full(columns + 1, np.inf, dtype=np.float64)
    previous_steps = np.zeros(columns + 1, dtype=np.int32)
    previous_cost[0] = 0.0
    for row in range(1, rows + 1):
        current_cost = np.full(columns + 1, np.inf, dtype=np.float64)
        current_steps = np.zeros(columns + 1, dtype=np.int32)
        for column in range(max(1, row - band), min(columns, row + band) + 1):
            candidates = (
                (previous_cost[column], previous_steps[column]),
                (current_cost[column - 1], current_steps[column - 1]),
                (previous_cost[column - 1], previous_steps[column - 1]),
            )
            best_cost, best_steps = min(candidates, key=lambda item: item[0])
            local = float(np.linalg.norm(first[row - 1] - second[column - 1]))
            current_cost[column] = best_cost + local
            current_steps[column] = best_steps + 1
        previous_cost, previous_steps = current_cost, current_steps
    return float(previous_cost[columns] / previous_steps[columns])


def test_dtw_identity_is_zero() -> None:
    features = _features(1)
    assert dtw_distance(features, features) == 0.0


def test_dtw_handles_different_speeds() -> None:
    slow = _features(2, 40)
    fast = slow[::2]
    assert dtw_distance(slow, fast) < 0.1


def test_optimized_dtw_matches_reference() -> None:
    first = _features(20, 37)
    second = _features(21, 29)

    assert np.isclose(
        dtw_distance(first, second),
        _reference_dtw_distance(first, second),
        rtol=1e-12,
        atol=1e-12,
    )


def test_vectorized_dtw_matches_reference_across_lengths() -> None:
    for first_frames, second_frames in ((5, 5), (9, 17), (31, 12), (80, 73)):
        first = _features(first_frames + 100, first_frames)
        second = _features(second_frames + 200, second_frames)

        assert np.isclose(
            dtw_distance(first, second),
            _reference_dtw_distance(first, second),
            rtol=1e-12,
            atol=1e-12,
        )


def test_native_dtw_backend_when_built() -> None:
    try:
        from ha_voice._dtw_native import accumulate_distance
    except ImportError:
        return

    first = _features(300, 43)
    second = _features(301, 36)
    rows, columns = first.shape[0], second.shape[0]
    band_ratio = 0.25
    band = max(abs(rows - columns), int(max(rows, columns) * band_ratio), 2)
    local_distances = np.ascontiguousarray(
        np.linalg.norm(
            first[:, np.newaxis, :] - second[np.newaxis, :, :], axis=2
        ),
        dtype=np.float32,
    )
    assert np.isclose(
        accumulate_distance(local_distances, band),
        _reference_dtw_distance(first, second, band_ratio),
        rtol=1e-12,
        atol=1e-12,
    )


def test_dtw_normalizes_native_input_to_contiguous_float32(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[np.ndarray] = []

    def fake_native(distances: np.ndarray, band: int) -> float:
        received.append(distances)
        return 1.25

    monkeypatch.setattr(matcher, "_native_accumulate_distance", fake_native)
    first = np.arange(24, dtype=np.float64).reshape(8, 3)[::2]
    second = np.arange(30, dtype=np.float64).reshape(10, 3)[::2]

    assert matcher.dtw_distance(first, second) == 1.25
    assert len(received) == 1
    assert received[0].dtype == np.float32
    assert received[0].flags.c_contiguous


def test_classify_selects_nearest_command() -> None:
    query = _features(3)
    templates = [
        Template("alpha", Path("alpha-1.wav"), query + 0.01),
        Template("alpha", Path("alpha-2.wav"), query - 0.01),
        Template("beta", Path("beta.wav"), query + 2.0),
    ]
    result = classify(
        query, templates, max_distance=0.5, min_margin=0.1, top_k=2
    )
    assert result.accepted
    assert result.command == "alpha"


def test_classify_rejects_ambiguous_command() -> None:
    query = _features(4)
    templates = [
        Template("alpha", Path("alpha.wav"), query + 0.10),
        Template("beta", Path("beta.wav"), query + 0.11),
    ]
    result = classify(
        query, templates, max_distance=1.0, min_margin=0.2, top_k=1
    )
    assert not result.accepted
    assert result.command is None


def test_shortlist_keeps_nearest_templates_per_source_group() -> None:
    query = _features(5)
    templates = [
        Template("start", Path("start/near.wav"), query + 0.01),
        Template("start", Path("start/far.wav"), query + 1.0),
        Template("negative", Path("first/near.wav"), query + 0.02),
        Template("negative", Path("first/far.wav"), query + 1.0),
        Template("negative", Path("second/near.wav"), query + 0.03),
        Template("negative", Path("second/far.wav"), query + 1.0),
    ]

    selected = shortlist_templates(
        query,
        templates,
        limits={"start": 1, "negative": 1},
    )

    assert [template.path.name for template in selected] == [
        "near.wav",
        "near.wav",
        "near.wav",
    ]


def test_classify_can_limit_templates_without_changing_clear_match() -> None:
    query = _features(6)
    templates = [
        Template("alpha", Path(f"alpha/{index}.wav"), query + index * 0.01)
        for index in range(1, 7)
    ] + [
        Template("beta", Path(f"beta/{index}.wav"), query + 2.0 + index * 0.01)
        for index in range(1, 7)
    ]

    result = classify(
        query,
        templates,
        max_distance=0.5,
        min_margin=0.1,
        top_k=3,
        default_template_limit=3,
    )

    assert result.accepted
    assert result.command == "alpha"


def test_load_templates_includes_explicit_command_rejections(tmp_path: Path) -> None:
    samples = np.sin(np.linspace(0, 20, 8000, dtype=np.float32)) * 0.2
    save_wav(tmp_path / "alpha" / "one.wav", Audio(samples, 16000))
    save_wav(tmp_path / "_not_command" / "one.wav", Audio(samples, 16000))

    templates = load_templates(tmp_path)

    assert [template.command for template in templates] == [
        "alpha",
        "_not_command",
    ]


def test_start_phrase_loader_reuses_existing_audio_as_negative(tmp_path: Path) -> None:
    samples = np.sin(np.linspace(0, 20, 8000, dtype=np.float32)) * 0.2
    save_wav(tmp_path / "_start_phrase" / "wake.wav", Audio(samples, 16000))
    save_wav(tmp_path / "alpha" / "command.wav", Audio(samples, 16000))
    save_wav(tmp_path / "_not_command" / "other.wav", Audio(samples, 16000))

    templates = load_start_phrase_templates(
        tmp_path,
        start_phrase_name="_start_phrase",
        negative_names={"alpha", "_not_command"},
    )

    assert [template.command for template in templates].count("_start_phrase") == 1
    assert [template.command for template in templates].count("_not_start_phrase") == 2


def test_start_phrase_loader_includes_reviewed_false_triggers(tmp_path: Path) -> None:
    samples = np.sin(np.linspace(0, 20, 8000, dtype=np.float32)) * 0.2
    save_wav(tmp_path / "_start_phrase" / "wake.wav", Audio(samples, 16000))
    save_wav(
        tmp_path / "_not_start_phrase" / "false.wav",
        Audio(samples, 16000),
    )

    templates = load_start_phrase_templates(
        tmp_path,
        start_phrase_name="_start_phrase",
        negative_names=set(),
    )

    assert [template.command for template in templates] == [
        "_start_phrase",
        "_not_start_phrase",
    ]
