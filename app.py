"""Streamlit demo: upload a chest X-ray -> pneumonia prediction + Grad-CAM heatmap.

Run:  streamlit run app.py
"""
import json
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

from src.data import CLASSES, IMG_SIZE, ROOT, eval_transforms
from src.gradcam import GradCAM, overlay
from src.model import CHECKPOINT, gradcam_target_layer, load_trained

SAMPLES_DIR = ROOT / "samples"
METRICS = ROOT / "models" / "metrics.json"

st.set_page_config(page_title="Pneumonia X-Ray Classifier", page_icon="🫁", layout="wide")


@st.cache_resource
def get_model():
    return load_trained(CHECKPOINT, device="cpu")


def analyse(img: Image.Image, model, threshold: float):
    gray = img.convert("L")
    x = eval_transforms()(gray).unsqueeze(0)
    with torch.no_grad():
        p_pneu = float(model(x).softmax(1)[0, 1])
    label_idx = int(p_pneu >= threshold)
    # Explain the class we actually report; explaining a near-zero score gives a blank map.
    cam_fn = GradCAM(model, gradcam_target_layer(model))
    try:
        cam, _, _ = cam_fn(x, class_idx=label_idx)
    finally:
        cam_fn.remove()
    base = np.asarray(gray.resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32) / 255.0
    base = np.repeat(base[..., None], 3, axis=2)
    return p_pneu, label_idx, base, overlay(base, cam)


st.title("🫁 Pneumonia Detection from Chest X-Rays")
st.caption("EfficientNet-B0 fine-tuned on the Kermany et al. pediatric chest X-ray dataset, "
           "with Grad-CAM explanations. Academic project - not a medical device.")

if not CHECKPOINT.exists():
    st.error(f"Model file not found at `{CHECKPOINT.relative_to(ROOT)}`. "
             "Copy `best_model.pt` from the Kaggle outputs into the `models/` folder.")
    st.stop()

model, meta = get_model()
threshold = float(meta["threshold"])

with st.sidebar:
    st.header("Model")
    st.write(f"Best epoch: **{meta.get('epoch', '?')}**")
    if "val_auc" in meta:
        st.write(f"Validation AUC: **{meta['val_auc']:.3f}**")
    st.write(f"Decision threshold: **{threshold:.2f}**")
    if METRICS.exists():
        m = json.loads(METRICS.read_text())
        t = m["at_tuned"]
        st.header("Test-set results")
        st.metric("ROC-AUC", f"{m['test_auc']:.3f}")
        c1, c2 = st.columns(2)
        c1.metric("Accuracy", f"{t['accuracy']:.1%}")
        c2.metric("F1", f"{t['f1']:.3f}")
        c1.metric("Sensitivity", f"{t['recall_sensitivity']:.1%}")
        c2.metric("Specificity", f"{t['specificity']:.1%}")

samples = sorted(SAMPLES_DIR.glob("*.jp*g")) if SAMPLES_DIR.exists() else []
left, right = st.columns([1, 1])
with left:
    uploaded = st.file_uploader("Upload a chest X-ray (JPEG/PNG)", type=["jpg", "jpeg", "png"])
with right:
    choice = st.selectbox("...or pick a sample from the test set",
                          ["-"] + [p.name for p in samples]) if samples else "-"

image = None
if uploaded is not None:
    image = Image.open(uploaded)
elif choice != "-":
    image = Image.open(SAMPLES_DIR / choice)

if image is None:
    st.info("Upload an X-ray or choose a sample to get a prediction.")
    st.stop()

with st.spinner("Analysing..."):
    p_pneu, label_idx, base, heat = analyse(image, model, threshold)

label = CLASSES[label_idx]
confidence = p_pneu if label == "PNEUMONIA" else 1 - p_pneu
if label == "PNEUMONIA":
    st.error(f"### Prediction: PNEUMONIA  ({confidence:.1%} confidence)")
else:
    st.success(f"### Prediction: NORMAL  ({confidence:.1%} confidence)")
st.progress(p_pneu, text=f"P(pneumonia) = {p_pneu:.3f}  (threshold {threshold:.2f})")

c1, c2 = st.columns(2)
c1.image(base, caption="Input X-ray (224×224)", width="stretch")
c2.image(heat, caption=f"Grad-CAM: regions driving the {label} prediction (red = high)", width="stretch")
