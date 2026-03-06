"""Tests for model parameter quantization."""

import numpy as np
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
from sklearn.datasets import load_wine, make_blobs
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from smart_inference_ai_fusion.models.knn_model import KNNModel
from smart_inference_ai_fusion.quantization.model.weight_quantizer import WeightQuantizer


def test_knn_quantization_keeps_original_unchanged() -> None:
    """quantize_model must return a copy, not mutate original estimator."""
    x, y = load_wine(return_X_y=True)
    model = KNeighborsClassifier(n_neighbors=5).fit(x[:120], y[:120])
    original_fit_x = model._fit_X.copy()  # pylint: disable=protected-access

    quantized = WeightQuantizer(num_bits=16).quantize_model(model)

    assert np.array_equal(model._fit_X, original_fit_x)  # pylint: disable=protected-access
    assert quantized is not model
    assert quantized._fit_X.shape == model._fit_X.shape  # pylint: disable=protected-access


def test_knn_quantization_preserves_reasonable_accuracy() -> None:
    """16-bit quantization should not collapse KNN performance."""
    x, y = load_wine(return_X_y=True)
    model = KNeighborsClassifier(n_neighbors=5).fit(x[:120], y[:120])
    baseline = model.score(x[120:], y[120:])
    quantized = WeightQuantizer(num_bits=16).quantize_model(model)
    quantized_score = quantized.score(x[120:], y[120:])
    assert abs(baseline - quantized_score) < 0.10


def test_tree_quantization_preserves_leaf_sentinel() -> None:
    """Leaf threshold sentinel (-2.0) must remain unchanged."""
    x, y = load_wine(return_X_y=True)
    model = DecisionTreeClassifier(max_depth=4, random_state=42).fit(x, y)
    quantized = WeightQuantizer(num_bits=16).quantize_model(model)
    assert np.any(model.tree_.threshold == -2.0)
    assert np.all(quantized.tree_.threshold[model.tree_.threshold == -2.0] == -2.0)


def test_mlp_quantization_keeps_weight_shapes() -> None:
    """MLP tensors should keep shape after quantize->dequantize."""
    x, y = load_wine(return_X_y=True)
    mlp = MLPClassifier(hidden_layer_sizes=(8,), max_iter=120, random_state=42).fit(x, y)
    quantized = WeightQuantizer(num_bits=16).quantize_model(mlp)
    assert [w.shape for w in quantized.coefs_] == [w.shape for w in mlp.coefs_]
    assert [b.shape for b in quantized.intercepts_] == [b.shape for b in mlp.intercepts_]


def test_kmeans_quantization_keeps_predict_usable() -> None:
    """MiniBatchKMeans quantized copy should still predict labels."""
    x, _ = make_blobs(n_samples=250, centers=3, random_state=42)
    model = MiniBatchKMeans(n_clusters=3, random_state=42).fit(x)
    quantized = WeightQuantizer(num_bits=16).quantize_model(model)
    labels = quantized.predict(x[:20])
    assert labels.shape == (20,)


def test_gmm_quantization_keeps_predict_usable() -> None:
    """GaussianMixture quantized copy should still predict components."""
    x, _ = make_blobs(n_samples=200, centers=3, random_state=42)
    model = GaussianMixture(n_components=3, random_state=42).fit(x)
    quantized = WeightQuantizer(num_bits=16).quantize_model(model)
    labels = quantized.predict(x[:20])
    assert labels.shape == (20,)


def test_agglomerative_has_no_weight_effect_but_copy_is_returned() -> None:
    """Models with no internal quantized params should still be copied safely."""
    x, _ = make_blobs(n_samples=120, centers=3, random_state=42)
    model = AgglomerativeClustering(n_clusters=3).fit(x)
    quantized = WeightQuantizer(num_bits=16).quantize_model(model)
    assert quantized is not model
    assert np.array_equal(model.labels_, quantized.labels_)


def test_wrapper_input_returns_wrapper_output() -> None:
    """Framework BaseModel wrappers should preserve wrapper type on output."""
    x, y = load_wine(return_X_y=True)
    wrapper = KNNModel(params={"n_neighbors": 3})
    wrapper.train(x[:120], y[:120])

    wrapper_quantized = WeightQuantizer(num_bits=16).quantize_model(wrapper)
    assert isinstance(wrapper_quantized, KNNModel)
    assert wrapper_quantized is not wrapper
