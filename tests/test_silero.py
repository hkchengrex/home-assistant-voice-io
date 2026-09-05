from pathlib import Path
import numpy as np
import pytest
from ha_voice.silero import SileroDetector
from ha_voice.continuous_vad_experiment import ReplayGate

class FakeSession:
    def __init__(self): self.inputs=[]
    def run(self, names, inputs):
        self.inputs.append({k:v.copy() for k,v in inputs.items()})
        return np.array([[0.9]],np.float32), inputs['state']+1

def test_silero_bridge_preserves_order_and_does_not_read_ahead():
    detector=SileroDetector.__new__(SileroDetector)
    detector.session=FakeSession()
    detector.threshold,detector.release=0.5,0.35
    detector.reset()
    assert not detector(np.arange(320,dtype=np.float32)/10000)
    assert detector(np.arange(320,640,dtype=np.float32)/10000)
    assert len(detector.session.inputs)==1
    first=detector.session.inputs[0]['input']
    np.testing.assert_array_equal(first[0,:64],np.zeros(64))
    np.testing.assert_array_equal(first[0,64:],np.arange(512,dtype=np.float32)/10000)
    detector(np.arange(640,960,dtype=np.float32)/10000)
    assert len(detector.session.inputs)==1
    detector(np.arange(960,1280,dtype=np.float32)/10000)
    second=detector.session.inputs[1]
    np.testing.assert_array_equal(second['input'][0,:64],np.arange(448,512,dtype=np.float32)/10000)
    np.testing.assert_array_equal(second['input'][0,64:],np.arange(512,1024,dtype=np.float32)/10000)
    assert np.all(second['state']==1)
    detector.reset()
    assert not detector.voiced and detector.pending.size==0
    assert np.all(detector.state==0)

def test_silero_rejects_wrong_model_before_loading_runtime(tmp_path):
    path=tmp_path/'bad.onnx'
    path.write_bytes(b'not a model')
    with pytest.raises(ValueError,match='hash'):
        SileroDetector(path)

def test_replay_gate_expires_in_audio_time():
    gate=ReplayGate.__new__(ReplayGate)
    gate._command_deadline=0
    gate.command_timeout_seconds=3
    gate.now=10
    gate.arm_command_window()
    gate.now=12.99
    assert gate.phase=='awaiting_command'
    gate.now=13
    assert gate.phase=='waiting_for_start'
