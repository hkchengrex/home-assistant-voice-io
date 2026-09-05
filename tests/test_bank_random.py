from collections import Counter
from types import SimpleNamespace
import numpy as np
import pytest
from ha_voice.bank_random import random_splits, batch_predictions
from ha_voice.bank_experiment import prediction


def test_random_splits_cover_every_clip_and_ignore_dates():
    rows=[{'id':str(i),'label':str(i//10),'day':'same'} for i in range(30)]
    first=random_splits(rows)
    second=random_splits([dict(r,day=str(i)) for i,r in enumerate(rows)])
    assert first==second
    for repeat in range(3):
        folds=[s for s in first if s['repeat']==repeat]
        counts=Counter(i for s in folds for i in s['test'])
        assert counts==Counter(range(30))
        for split in folds:
            assert not set(split['train']) & set(split['test'])
            assert set(split['train']) | set(split['test']) == set(range(30))
            assert Counter(rows[i]['label'] for i in split['test']) == {'0':2,'1':2,'2':2}


def test_duplicate_audio_identity_cannot_cross_splits():
    with pytest.raises(ValueError):
        random_splits([{'id':'same','label':'a'},{'id':'same','label':'a'}])


@pytest.mark.parametrize('head',['wake','command'])
@pytest.mark.parametrize('k',[1,2,3])
def test_vectorized_scores_match_existing_scorer_with_self_excluded(head,k):
    config=SimpleNamespace(start_phrase=SimpleNamespace(name='_wake'),calibration=SimpleNamespace(name='_calibration'),
        commands={'on':SimpleNamespace(intent_group='light'),'alternate':SimpleNamespace(intent_group='light'),
                  'off':SimpleNamespace(intent_group='dark')})
    labels=['_wake','_wake','on','on','alternate','off','off','_not_command','_not_start_phrase']
    rows=[{'id':str(i),'label':label,'day':'unused'} for i,label in enumerate(labels)]
    matrix=np.random.default_rng(17).uniform(.1,4,(len(rows),len(rows))).astype(np.float64)
    np.fill_diagonal(matrix,np.inf)
    train=list(range(8))
    actual=batch_predictions(matrix,list(range(len(rows))),train,rows,config,head,k)
    for result in actual:
        i=int(result['id'])
        expected=prediction(matrix,i,[j for j in train if j!=i],rows,head,config,k)
        for key in ('target','predicted','represented'):
            assert result[key]==expected[key]
        assert result['distance']==pytest.approx(expected['distance'],abs=1e-12)
        assert result['margin']==pytest.approx(expected['margin'],abs=1e-12)
