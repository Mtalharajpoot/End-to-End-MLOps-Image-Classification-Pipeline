"""Torch-free test of the ONNX serving path, using a hand-built 3-op ONNX graph."""

import json

import numpy as np
import pytest
from PIL import Image

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from visionops.inference import Predictor, meta_path, softmax  # noqa: E402

SIZE, N_CLASSES = 8, 3
MEAN, STD = [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]


@pytest.fixture()
def tiny_model(tmp_path):
    rng = np.random.default_rng(0)
    weights = rng.normal(size=(3, N_CLASSES)).astype(np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("GlobalAveragePool", ["image"], ["pooled"]),
            helper.make_node("Flatten", ["pooled"], ["flat"], axis=1),
            helper.make_node("Gemm", ["flat", "w", "b"], ["logits"]),
        ],
        "tiny",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, ["batch", 3, SIZE, SIZE])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", N_CLASSES])],
        initializer=[
            numpy_helper.from_array(weights, "w"),
            numpy_helper.from_array(np.zeros(N_CLASSES, dtype=np.float32), "b"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    path = tmp_path / "model.onnx"
    onnx.save(model, path)
    meta_path(path).write_text(
        json.dumps({"classes": ["a", "b", "c"], "image_size": SIZE, "mean": MEAN, "std": STD})
    )
    return path, weights


def test_softmax_sums_to_one():
    p = softmax(np.array([1.0, 2.0, 3.0]))
    assert p.sum() == pytest.approx(1.0)
    assert p.argmax() == 2


def test_predict_matches_manual_computation(tiny_model):
    path, weights = tiny_model
    predictor = Predictor(path)
    image = Image.new("RGB", (40, 30), color=(200, 100, 50))  # arbitrary size gets resized

    expected_in = (np.array([200, 100, 50], dtype=np.float32) / 255.0 - MEAN) / STD
    expected = softmax(expected_in @ weights)
    preds = predictor.predict(image, top_k=3)

    assert [c for c, _ in preds] == [["a", "b", "c"][i] for i in np.argsort(-expected)]
    assert sum(p for _, p in preds) == pytest.approx(1.0, abs=1e-5)
    assert preds[0][1] == pytest.approx(float(expected.max()), abs=1e-4)


def test_preprocess_shape_and_dtype(tiny_model):
    path, _ = tiny_model
    x = Predictor(path).preprocess(Image.new("L", (100, 100)))  # grayscale input is converted
    assert x.shape == (1, 3, SIZE, SIZE)
    assert x.dtype == np.float32
