import unittest
import numpy as np
from ha_voice.bank_metric import cosine_dtw
from ha_voice.matcher import _native_accumulate_distance


class CosineDtwTests(unittest.TestCase):
    @unittest.skipIf(_native_accumulate_distance is None, 'native DTW required')
    def test_positive_frame_rescaling_preserves_distance(self):
        rng=np.random.default_rng(3)
        first=rng.normal(size=(20,8)).astype(np.float32)
        second=rng.normal(size=(18,8)).astype(np.float32)
        expected=cosine_dtw(first,second)
        actual=cosine_dtw(first*np.linspace(.1,4,20)[:,None],second*np.linspace(.2,3,18)[:,None])
        self.assertAlmostEqual(expected,actual,places=6)

    @unittest.skipIf(_native_accumulate_distance is None, 'native DTW required')
    def test_repeated_matching_frames_align_at_zero_cost(self):
        first=np.eye(3,dtype=np.float32)
        second=np.repeat(first,2,axis=0)
        self.assertAlmostEqual(cosine_dtw(first,second),0.,places=6)

    def test_feature_width_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            cosine_dtw(np.ones((3,2)),np.ones((3,4)))
