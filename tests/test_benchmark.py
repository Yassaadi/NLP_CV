import numpy as np
from sklearn.datasets import make_blobs

from benchmark import benchmark, combine


def test_informative_features_beat_noise():
    X, y = make_blobs(n_samples=300, centers=4, n_features=20, random_state=0)
    noise = np.random.default_rng(0).normal(size=(300, 20))
    result = benchmark({"informative": X, "noise": noise}, y)
    assert result.iloc[0]["representation"] == "informative"
    assert result.iloc[0]["ari"] > 0.9 and result.iloc[1]["ari"] < 0.1


def test_combine_keeps_small_modality_visible():
    X, y = make_blobs(n_samples=300, centers=4, n_features=10, random_state=1)
    big_noise = np.random.default_rng(1).normal(scale=100, size=(300, 2000))
    naive = benchmark({"naive": np.hstack([X, big_noise])}, y, n_components=10).iloc[0]["ari"]
    balanced = benchmark({"balanced": combine(X, big_noise, n_components=10)}, y, n_components=20).iloc[0]["ari"]
    assert balanced > naive
