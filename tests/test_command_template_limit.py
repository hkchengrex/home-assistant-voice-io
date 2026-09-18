from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from ha_voice.config import load_config
from ha_voice.matcher import Template
from ha_voice.studio_server import match_features


@pytest.mark.parametrize("limit", [0, 8, 16])
def test_load_command_template_limit(tmp_path, limit):
    path = tmp_path / "commands.toml"
    path.write_text(f"[recognizer]\ncommand_template_limit = {limit}\n[commands.lights_on]\n")
    assert load_config(path).recognizer.command_template_limit == limit


def test_reject_negative_limit(tmp_path):
    path = tmp_path / "commands.toml"
    path.write_text("[recognizer]\ncommand_template_limit = -1\n[commands.lights_on]\n")
    with pytest.raises(ValueError, match="command_template_limit"):
        load_config(path)


def test_full_bank_recovers_time_warped_matches_omitted_by_shortlist():
    config = load_config(Path(__file__).parents[1] / "commands.toml")
    config = replace(config, recognizer=replace(config.recognizer, max_distance=.05, min_margin=.06, top_k=3))
    query = np.repeat([0., 1.], 10).astype(np.float32)[:, None]
    templates = [Template("lights_on", Path(f"lights_on/distractor{i}.wav"), query + .1 + i * .01) for i in range(8)]
    for count in (13, 14, 15):
        warped = np.array([0.] * count + [1.] * (20 - count), dtype=np.float32)[:, None]
        templates.append(Template("lights_on", Path(f"lights_on/warped{count}.wav"), warped))
    templates.append(Template("_not_command", Path("_not_command/negative.wav"), query + .8))
    shortlisted = match_features(features=query, config=config, templates=templates)
    full_config = replace(config, recognizer=replace(config.recognizer, command_template_limit=0))
    full = match_features(features=query, config=full_config, templates=templates)
    assert not shortlisted["accepted"]
    assert full["accepted"]
    assert full["command"] == "lights_on"
    assert full["score"] == pytest.approx(0.)
