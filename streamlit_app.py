"""Streamlit demo: upload an image, get top-5 class probabilities from the ONNX model."""

import os
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from visionops.inference import Predictor

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.onnx")

st.set_page_config(page_title="VisionOps", page_icon="🧠")
st.title("🧠 VisionOps — Image Classifier")
st.caption("ONNX Runtime inference • trained with PyTorch Lightning • tracked with W&B")


@st.cache_resource
def load_predictor(path: str) -> Predictor:
    return Predictor(path)


if not Path(MODEL_PATH).exists():
    st.error(
        f"Model not found at `{MODEL_PATH}`. Train first (`make train`) and copy "
        "`outputs/model.onnx` and `outputs/model_meta.json` into `models/`."
    )
    st.stop()

predictor = load_predictor(MODEL_PATH)
upload = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if upload is not None:
    image = Image.open(upload)
    st.image(image, caption="Input", width=256)
    preds = predictor.predict(image, top_k=5)
    st.subheader(f"Prediction: **{preds[0][0]}** ({preds[0][1]:.1%})")
    st.bar_chart(pd.DataFrame({"probability": [p for _, p in preds]}, index=[c for c, _ in preds]))
