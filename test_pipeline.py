import unittest

import cv2 as cv
import numpy as np

from glare import Config, GlareSuppressor, coordinates, flow_reliability, fusion_weights, glare_masks, pyramid_fuse, render_highlights, valid_bilinear
from inputs import StereoInput, build_maps
from make_demo import scene


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv.setNumThreads(1)
        cv.ocl.setUseOpenCL(False)

    def test_geometry_and_consistency_are_independent(self):
        gray = (np.indices((48, 56)).sum(axis=0) % 7).astype(np.float32) * 12
        support = np.ones_like(gray, dtype=bool)
        support[21, 21] = False
        x, y = coordinates(gray.shape)
        valid = valid_bilinear(support, x + 0.5, y + 0.5)
        self.assertFalse(valid[20:22, 20:22].any())
        forward = np.zeros((*gray.shape, 2), dtype=np.float32)
        backward = forward.copy()
        backward[18:23, 29:33, 0] = 3
        _, _, valid, occluded, confidence, error = flow_reliability(gray, gray, support, support, forward, backward, Config())
        self.assertTrue(valid[20, 30] and occluded[20, 30])
        self.assertGreater(confidence[20, 30], 0.99)
        self.assertAlmostEqual(error[20, 30], 3.0)
        self.assertFalse(occluded[21, 21])
        self.assertTrue(np.isnan(error[-1, -1]))

    def test_gate_gain_and_weights(self):
        normal = np.full((80, 80), 76, np.float32)
        dark = normal / 2
        normal[36:43, 36:43] = 255
        normal[35, 40] = 150
        normal[1, 1] = 180
        valid = np.ones_like(normal, dtype=bool)
        core, candidate, halo, gate, gain, _, _ = glare_masks(normal, dark, valid, valid, Config())
        self.assertEqual(gain, 2)
        self.assertTrue(core[40, 40] and candidate[35, 40] and halo[35, 40])
        self.assertFalse(halo[1, 1])
        self.assertGreater(gate[40, 40], gate[1, 1])
        self.assertIsNone(glare_masks(normal, dark, valid, ~valid, Config())[4])
        config = Config()
        weights = fusion_weights(np.full((1, 4), 0.85), np.array([[1, .4, 1, 1]]),
                                 np.array([[True, True, False, True]]),
                                 np.array([[False, False, False, True]]), True, config)
        np.testing.assert_allclose(weights.sum(axis=-1), 1)
        np.testing.assert_array_equal(weights[0, 2:], [[1, 0, 0], [1, 0, 0]])
        self.assertGreater(weights[0, 0, 1:].sum(), weights[0, 1, 1:].sum())
        self.assertTrue(np.all(weights[..., 2] <= weights[..., 1:].sum(axis=-1) * config.auxiliary_max))

    def test_odd_pyramid_reconstruction(self):
        frame = np.random.default_rng(0).integers(0, 256, (73, 91, 3), dtype=np.uint8)
        weights = np.zeros_like(frame, dtype=np.float32)
        weights[..., 0] = 1
        np.testing.assert_array_equal(pyramid_fuse((frame, 255 - frame, frame), weights, 5), frame)

    def test_stereo_calibration_and_view_order(self):
        k = [[50, 0, 24], [0, 50, 18], [0, 0, 1]]
        maps = build_maps(dict(image_size=[48, 36], left_camera_matrix=k, right_camera_matrix=k,
                               left_distortion=[0] * 5, right_distortion=[0] * 5,
                               R=np.eye(3).tolist(), T=[-0.12, 0, 0]))
        self.assertTrue(all(m.shape == (36, 48) for m in maps))
        normal, dark = scene()
        stereo = StereoInput(normal_side="right")
        a, b, _ = stereo.prepare(*stereo.split(np.hstack((dark, normal))))
        np.testing.assert_array_equal(a, normal)
        np.testing.assert_array_equal(b, dark)

    def test_end_to_end_translation_and_fallback(self):
        normal, dark = scene()
        result = GlareSuppressor(Config()).process(normal, dark)
        region = np.s_[30:90, 25:160]
        np.testing.assert_allclose(np.median(result['forward'][region], axis=(0, 1)), [6, 2], atol=1)
        self.assertGreater(result['reliable'].mean(), .65)
        self.assertAlmostEqual(result['gain'], 1 / .43, delta=.15)
        halo = result['halo'] & result['reliable']
        self.assertGreater(halo.sum(), 100)
        self.assertGreater(normal[halo].mean() - result['fused'][halo].mean(), 10)
        fallback = ~result['valid'] | result['occluded'] | (result['confidence'] <= 0)
        np.testing.assert_array_equal(result['raw_fused'][fallback], normal[fallback])

    def test_highlight_render_keeps_shadow_color_and_softens_fallback(self):
        normal = np.full((160, 224, 3), [2, 0, 0], dtype=np.uint8)
        normal[25:120, 70:170] = 255
        fused = normal.copy()
        fused[70:80, 110:120] = 45
        rendered = render_highlights(normal, fused, .38, 32)
        np.testing.assert_array_equal(rendered[0:20], normal[0:20])
        self.assertLess(int(rendered[60, 110, 0]), 225)
        self.assertGreater(int(rendered[75, 115, 0]), 175)
        self.assertLess(abs(int(rendered[60, 110, 0]) - int(rendered[75, 115, 0])), 30)
        np.testing.assert_array_equal(render_highlights(normal, fused, 0, 32), fused)


if __name__ == "__main__":
    unittest.main()
