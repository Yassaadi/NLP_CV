"""Benchmark feature representations for unsupervised product categorisation.

Each representation is clustered with k-means (k = number of true categories) and
scored against the real categories with the Adjusted Rand Index (ARI): 0 means no
better than random grouping, 1 means the clusters match the categories exactly.

    from benchmark import benchmark, combine
    results = benchmark({"tfidf": X_tfidf, "resnet": X_img, "text+image": combine(X_tfidf, X_img)}, y)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


def reduce(X: np.ndarray, n_components: int = 50, seed: int = 0) -> np.ndarray:
    """Standardise and project on principal components (keeps k-means fast and stable)."""
    X = StandardScaler().fit_transform(np.asarray(X, dtype=float))
    n = min(n_components, X.shape[0] - 1, X.shape[1])
    return PCA(n_components=n, random_state=seed).fit_transform(X)


def combine(*blocks: np.ndarray, n_components: int = 50) -> np.ndarray:
    """Concatenate modalities after reducing each one to the same size.

    Without this, the largest block (e.g. 100,000 image dimensions vs 512 text
    dimensions) dominates the distance and the other modalities are ignored.
    """
    return np.hstack([reduce(b, n_components) for b in blocks])


def score(X: np.ndarray, y: np.ndarray, seed: int = 0) -> dict[str, float]:
    k = len(np.unique(y))
    labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
    return {"ari": adjusted_rand_score(y, labels), "silhouette": silhouette_score(X, labels)}


def benchmark(features: dict[str, np.ndarray], y, n_components: int = 50) -> pd.DataFrame:
    """ARI and silhouette for each representation, best first."""
    y = np.asarray(y)
    rows = [{"representation": name, **score(reduce(X, n_components), y)} for name, X in features.items()]
    return pd.DataFrame(rows).sort_values("ari", ascending=False, ignore_index=True).round(3)
