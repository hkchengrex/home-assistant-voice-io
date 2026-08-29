"""Dynamic Time Warping and template classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import load_wav, trim_silence, trim_spoken_phrase
from .features import extract_command_features, extract_mfcc

try:
    from ._dtw_native import accumulate_distance as _native_accumulate_distance
except ImportError:  # Optional fallback for platforms without a C compiler.
    _native_accumulate_distance = None


@dataclass(frozen=True)
class Template:
    command: str
    path: Path
    features: np.ndarray


@dataclass(frozen=True)
class MatchResult:
    command: str | None
    score: float
    runner_up_score: float
    margin: float
    accepted: bool
    per_command: dict[str, float]


def _temporal_signature(features: np.ndarray, frame_count: int = 24) -> np.ndarray:
    """Resample a feature sequence into a cheap fixed-size comparison signature."""
    positions = np.linspace(0.0, features.shape[0] - 1, frame_count)
    left = np.floor(positions).astype(np.int32)
    right = np.minimum(left + 1, features.shape[0] - 1)
    weight = (positions - left)[:, np.newaxis]
    resampled = features[left] * (1.0 - weight) + features[right] * weight
    return resampled.reshape(-1)


def shortlist_templates(
    query: np.ndarray,
    templates: list[Template],
    *,
    limits: dict[str, int],
    default_limit: int | None = None,
) -> list[Template]:
    """Choose query-relevant representatives while retaining every saved template."""
    query_signature = _temporal_signature(query)
    groups: dict[tuple[str, str], list[tuple[float, Template]]] = {}
    for template in templates:
        group = (template.command, template.path.parent.name)
        distance = float(
            np.linalg.norm(query_signature - _temporal_signature(template.features))
        )
        groups.setdefault(group, []).append((distance, template))

    selected: list[Template] = []
    for (command, _), candidates in groups.items():
        limit = limits.get(command, default_limit)
        if limit is None:
            selected.extend(template for _, template in candidates)
            continue
        if limit < 1:
            raise ValueError("template shortlist limits must be positive")
        candidates.sort(key=lambda item: item[0])
        selected.extend(template for _, template in candidates[:limit])
    return selected


def dtw_distance(first: np.ndarray, second: np.ndarray, band_ratio: float = 0.25) -> float:
    """Compute path-length-normalized DTW with a Sakoe-Chiba band."""
    if first.ndim != 2 or second.ndim != 2 or first.shape[1] != second.shape[1]:
        raise ValueError("feature matrices must be 2-D with equal feature counts")
    if first.shape[0] == 0 or second.shape[0] == 0:
        raise ValueError("feature matrices may not be empty")

    rows, columns = first.shape[0], second.shape[0]
    band = max(abs(rows - columns), int(max(rows, columns) * band_ratio), 2)
    # Compute all frame-to-frame distances in one NumPy operation. Calling
    # np.linalg.norm for every DTW cell is especially expensive on low-power
    # low-power target machines.
    local_distances = np.linalg.norm(
        first[:, np.newaxis, :] - second[np.newaxis, :, :], axis=2
    )
    if _native_accumulate_distance is not None:
        # The optional extension deliberately has a narrow float32 ABI. NumPy
        # promotes mixed and float64 feature inputs, and sliced arrays are not
        # guaranteed to be contiguous, so normalize at the extension boundary.
        native_distances = np.ascontiguousarray(local_distances, dtype=np.float32)
        return float(_native_accumulate_distance(native_distances, band))

    previous_cost = np.full(columns + 1, np.inf, dtype=np.float64)
    previous_steps = np.zeros(columns + 1, dtype=np.int32)
    previous_cost[0] = 0.0

    for row in range(1, rows + 1):
        current_cost = np.full(columns + 1, np.inf, dtype=np.float64)
        current_steps = np.zeros(columns + 1, dtype=np.int32)
        start = max(1, row - band)
        end = min(columns, row + band)
        local = local_distances[row - 1, start - 1 : end].astype(
            np.float64, copy=False
        )

        # A path enters this row from above or diagonally, then may continue
        # horizontally. Expressing every possible entry as a prefix-sum offset
        # turns the left-dependent inner loop into a cumulative minimum.
        up_cost = previous_cost[start : end + 1]
        diagonal_cost = previous_cost[start - 1 : end]
        use_up = up_cost <= diagonal_cost
        entry_cost = np.where(use_up, up_cost, diagonal_cost)
        entry_steps = np.where(
            use_up,
            previous_steps[start : end + 1],
            previous_steps[start - 1 : end],
        )

        prefix = np.cumsum(local, dtype=np.float64)
        prefix_before = np.concatenate((np.zeros(1), prefix[:-1]))
        keys = entry_cost - prefix_before
        running_min = np.minimum.accumulate(keys)

        # Track which entry supplied each running minimum so path-length
        # normalization and the original up/left/diagonal tie order are kept.
        prior_min = np.concatenate((np.full(1, np.inf), running_min[:-1]))
        update = (keys < prior_min) | ((keys == prior_min) & use_up)
        offsets = np.arange(end - start + 1, dtype=np.int32)
        selected = np.maximum.accumulate(np.where(update, offsets, -1))

        current_cost[start : end + 1] = prefix + running_min
        current_steps[start : end + 1] = (
            entry_steps[selected] + offsets - selected + 1
        )
        previous_cost, previous_steps = current_cost, current_steps

    if not np.isfinite(previous_cost[columns]) or previous_steps[columns] == 0:
        return float("inf")
    return float(previous_cost[columns] / previous_steps[columns])


def load_templates(recordings_dir: Path, sample_rate: int = 16_000) -> list[Template]:
    templates: list[Template] = []
    if not recordings_dir.exists():
        return templates
    for path in sorted(recordings_dir.glob("*/*.wav")):
        # Underscore-prefixed sets are calibration data, never command classes.
        if path.parent.name.startswith("_"):
            continue
        audio = load_wav(path, sample_rate)
        samples = trim_silence(audio.samples, audio.sample_rate)
        if samples.size == 0:
            continue
        templates.append(
            Template(
                command=path.parent.name,
                path=path,
                features=extract_command_features(samples, audio.sample_rate),
            )
        )
    for path in sorted((recordings_dir / "_not_command").glob("*.wav")):
        audio = load_wav(path, sample_rate)
        samples = trim_silence(audio.samples, audio.sample_rate)
        if samples.size == 0:
            continue
        templates.append(
            Template(
                command="_not_command",
                path=path,
                features=extract_command_features(samples, audio.sample_rate),
            )
        )
    return templates


def load_start_phrase_templates(
    recordings_dir: Path,
    *,
    start_phrase_name: str,
    negative_names: set[str],
    sample_rate: int = 16_000,
) -> list[Template]:
    """Load a binary start-phrase set using existing speech as rejection data."""
    templates: list[Template] = []
    labels = {start_phrase_name: start_phrase_name}
    labels.update({name: "_not_start_phrase" for name in negative_names})
    labels["_not_start_phrase"] = "_not_start_phrase"
    for directory_name, label in labels.items():
        for path in sorted((recordings_dir / directory_name).glob("*.wav")):
            audio = load_wav(path, sample_rate)
            samples = (
                trim_spoken_phrase(audio.samples, audio.sample_rate)
                if directory_name == start_phrase_name
                else trim_silence(audio.samples, audio.sample_rate)
            )
            if samples.size == 0:
                continue
            templates.append(
                Template(
                    command=label,
                    path=path,
                    features=extract_mfcc(samples, audio.sample_rate),
                )
            )
    return templates


def classify(
    query: np.ndarray,
    templates: list[Template],
    *,
    max_distance: float,
    min_margin: float,
    top_k: int = 3,
    template_limits: dict[str, int] | None = None,
    default_template_limit: int | None = None,
) -> MatchResult:
    if not templates:
        raise ValueError("no templates are available")
    if template_limits is not None or default_template_limit is not None:
        templates = shortlist_templates(
            query,
            templates,
            limits=template_limits or {},
            default_limit=default_template_limit,
        )
    distances: dict[str, list[float]] = {}
    for template in templates:
        distances.setdefault(template.command, []).append(
            dtw_distance(query, template.features)
        )

    per_command: dict[str, float] = {}
    for command, command_distances in distances.items():
        nearest = sorted(command_distances)[: min(top_k, len(command_distances))]
        per_command[command] = float(np.mean(nearest))

    ranking = sorted(per_command.items(), key=lambda item: item[1])
    best_command, best_score = ranking[0]
    runner_up_score = ranking[1][1] if len(ranking) > 1 else float("inf")
    margin = (
        1.0
        if not np.isfinite(runner_up_score)
        else max(0.0, (runner_up_score - best_score) / max(runner_up_score, 1e-9))
    )
    accepted = best_score <= max_distance and margin >= min_margin
    return MatchResult(
        command=best_command if accepted else None,
        score=best_score,
        runner_up_score=runner_up_score,
        margin=margin,
        accepted=accepted,
        per_command=per_command,
    )


def leave_one_out_distances(templates: list[Template]) -> tuple[list[float], list[float]]:
    """Return nearest same-command and different-command distances per template."""
    genuine: list[float] = []
    impostor: list[float] = []
    for index, query in enumerate(templates):
        same: list[float] = []
        different: list[float] = []
        for other_index, candidate in enumerate(templates):
            if index == other_index:
                continue
            distance = dtw_distance(query.features, candidate.features)
            if query.command == candidate.command:
                same.append(distance)
            else:
                different.append(distance)
        if same:
            genuine.append(min(same))
        if different:
            impostor.append(min(different))
    return genuine, impostor
