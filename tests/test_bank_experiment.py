from types import SimpleNamespace
import numpy as np
from ha_voice.bank_experiment import features, scores, threshold_fit, prediction


def configuration():
    return SimpleNamespace(start_phrase=SimpleNamespace(name='_wake'), commands={
        'on': SimpleNamespace(intent_group='lights'),
        'on_alternate': SimpleNamespace(intent_group='lights'),
        'off': SimpleNamespace(intent_group='dark')})


def test_intent_alias_is_not_runner_up():
    rows = [{'label': label, 'id': str(i), 'day': '2026-01-01'} for i, label in enumerate(['on','on_alternate','off','_not_command'])]
    matrix = np.array([[0,1.1,3,4],[1.1,0,3,4],[3,3,0,4],[4,4,4,0]])
    predicted, distance, margin = scores(matrix,0,[1,2,3],rows,'command',configuration(),1)
    assert predicted == 'lights'
    assert distance == 1.1
    assert margin > .6


def test_missing_heldout_class_is_reported_as_unsupported():
    rows = [{'label': label, 'id': str(i), 'day': '2026-01-01'} for i, label in enumerate(['on','off','_not_command'])]
    matrix = np.array([[0,1,2],[1,0,2],[2,2,0]])
    result = prediction(matrix,0,[1,2],rows,'command',configuration(),1)
    assert result is not None and not result['represented']


def test_calibration_does_not_accept_training_impostor():
    predictions = [dict(distance=1.,margin=.4,predicted='lights',target='lights'),
                   dict(distance=1.1,margin=.1,predicted='lights',target=None)]
    distance, margin = threshold_fit(predictions)
    assert predictions[0]['distance'] <= distance and predictions[0]['margin'] >= margin
    assert not (predictions[1]['distance'] <= distance and predictions[1]['margin'] >= margin)


def test_variance_floor_profiles_handle_low_energy_and_short_audio():
    for profile in ('mfcc_floor','pcen_floor','pcen24_floor'):
        for samples in (np.zeros(100,dtype=np.float32), np.sin(np.arange(16000)*.11).astype(np.float32)*1e-7):
            output = features(samples,profile)
            assert output.shape[1] == (48 if profile == 'pcen24_floor' else 26)
            assert np.isfinite(output).all()
