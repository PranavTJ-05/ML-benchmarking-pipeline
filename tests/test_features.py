import numpy as np

from kws_benchmark.data.features import mfcc, normalize


def test_mfcc_shape_and_determinism():
    audio = np.sin(np.linspace(0, 20, 16_000)).astype(np.float32)
    first = mfcc(audio)
    assert first.shape == (101, 40)
    np.testing.assert_array_equal(first, mfcc(audio))


def test_normalization_uses_training_statistics():
    features = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    normalized, mean, std = normalize(features)
    np.testing.assert_allclose(normalized.mean(axis=(0, 1)), 0, atol=1e-6)
    repeated, _, _ = normalize(features, mean, std)
    np.testing.assert_array_equal(normalized, repeated)
