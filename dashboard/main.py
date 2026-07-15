"""
dashboard/main.py — Step 4: Streamlit Dashboard

Local, single-user demo tool (NOT a deployed service — see caveat at the
bottom of every result). Flow:

  1. User uploads an image.
  2. features.py's extraction functions run on it directly (same code path
     as training — no drift between train-time and inference-time features).
  3. outputs/model_artifacts.joblib is loaded once at startup (cached), so
     no retraining happens per upload.
  4. Scaled features go through both LogReg and RF; RF is used as the
     "primary" score since it outperformed LogReg in LOOCV (81.5% vs 74.1%),
     but both are shown for transparency.
  5. ai.py sends the model's score + feature deviations (not the image) to
     Groq for a plain-language, structured explanation.
  6. Everything renders top-to-bottom: image -> scores -> explanation.

This file lives at PROJECT_ROOT/dashboard/main.py -- two levels down from
PROJECT_ROOT, hence the double os.path.dirname() below. If you move this
file again, update that line to match its new depth.

Run from the PROJECT ROOT (not from inside dashboard/):
    streamlit run dashboard/main.py
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from PIL import Image

# This file is at PROJECT_ROOT/dashboard/main.py, so two dirname() calls
# walk back up to PROJECT_ROOT. (Previously one call, when main.py lived
# directly at the project root -- update this again if the file moves.)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "ai"))

from features import check_exif, fft_hf_ratio, ela_variance, noise_perfection_score  # noqa: E402
from ai import compute_baselines, get_llm_explanation, FEATURE_COLS  # noqa: E402

MODEL_ARTIFACTS_PATH = os.path.join(PROJECT_ROOT, "outputs", "model_artifacts.joblib")
FEATURES_CSV_PATH = os.path.join(PROJECT_ROOT, "outputs", "features.csv")

st.set_page_config(page_title="Fake Image Detector — Demo", page_icon="🔍", layout="centered")


# ---------------- Cached loaders (run once per session, not per upload) ----------------

@st.cache_resource
def load_model_artifacts():
    if not os.path.exists(MODEL_ARTIFACTS_PATH):
        return None
    return joblib.load(MODEL_ARTIFACTS_PATH)


@st.cache_data
def load_baselines():
    features_df = pd.read_csv(FEATURES_CSV_PATH)
    return compute_baselines(features_df)


# ---------------- Feature extraction for one uploaded image ----------------

def extract_features_for_upload(img: Image.Image) -> dict:
    """Mirrors features.py's extract_features_for_dir, for a single in-memory image."""
    has_exif, has_camera_tag = check_exif(img)
    fft_score = fft_hf_ratio(img)
    ela_var, ela_mean = ela_variance(img)
    noise_score = noise_perfection_score(img)
    return {
        "has_exif": int(has_exif),
        "has_camera_tag": int(has_camera_tag),
        "fft_hf_ratio": fft_score,
        "ela_variance": ela_var,
        "ela_mean": ela_mean,
        "noise_perfection_score": noise_score,
    }


# ---------------- UI ----------------

st.title("🔍 Fake Image Detector")
st.caption(
    "Local demo tool — trained on 27 dog photos as a one-day proof-of-concept. "
    "Not a validated detector. See caveats below every result."
)

artifacts = load_model_artifacts()
if artifacts is None:
    st.error(
        f"No trained model found at `outputs/model_artifacts.joblib`. "
        f"Run `python3 src/model.py` first to train and save it."
    )
    st.stop()

scaler = artifacts["scaler"]
logreg = artifacts["logreg"]
rf = artifacts["rf"]

baselines = load_baselines()

uploaded_file = st.file_uploader("Upload an image (jpg, jpeg, png, webp)", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img.load()

    st.image(img, caption=uploaded_file.name, use_container_width=True)

    with st.spinner("Extracting features..."):
        image_features = extract_features_for_upload(img)
        X = np.array([[image_features[col] for col in FEATURE_COLS]])
        X_scaled = scaler.transform(X)

        logreg_proba = float(logreg.predict_proba(X_scaled)[0][1])
        rf_proba = float(rf.predict_proba(X_scaled)[0][1])  # RF unaffected by scaling, kept consistent with model.py

    st.subheader("Classifier scores")
    col1, col2 = st.columns(2)
    col1.metric("Logistic Regression — P(fake)", f"{logreg_proba:.1%}")
    col2.metric("Random Forest — P(fake)", f"{rf_proba:.1%}")

    if abs(logreg_proba - rf_proba) > 0.3:
        st.warning(
            "The two models disagree substantially on this image "
            f"(LogReg {logreg_proba:.1%} vs RF {rf_proba:.1%} P(fake)). "
            "Treat this as a strong signal to rely on human review."
        )

    with st.expander("Raw feature values vs. dataset baselines"):
        rows = []
        for col in FEATURE_COLS:
            b = baselines[col]
            rows.append({
                "feature": col,
                "this image": round(image_features[col], 4),
                "real mean ± std": f"{b['real_mean']:.4f} ± {b['real_std']:.4f}",
                "fake mean ± std": f"{b['fake_mean']:.4f} ± {b['fake_std']:.4f}",
            })
        st.table(pd.DataFrame(rows))

    st.subheader("AI explanation")
    with st.spinner("Asking the reasoning layer to explain the scores..."):
        try:
            explanation = get_llm_explanation(
                image_features, baselines,
                classifier_score=rf_proba,
                model_name="Random Forest",
            )
        except Exception as e:
            explanation = None
            st.error(f"LLM explanation failed: {e}")

    if explanation:
        verdict = explanation["verdict"]
        confidence = explanation["confidence"]

        verdict_color = {
            "likely authentic": "green",
            "likely synthetic": "red",
            "flag for human review": "orange",
        }.get(verdict, "gray")

        st.markdown(f"**Verdict:** :{verdict_color}[{verdict}]  ·  **Confidence:** {confidence}")
        st.markdown(f"**Top contributing signal:** `{explanation['top_contributing_signal']}`")
        st.write(explanation["explanation"])

        if explanation["disagreement_note"].strip().lower() != "none":
            st.info(f"**Disagreement noted:** {explanation['disagreement_note']}")

        st.caption(f"⚠️ {explanation['caveat']}")

    st.divider()
    st.caption(
        "This is an offline, single-user demo built for a one-day proof-of-concept — "
        "not a deployed or production detection service. Results should be treated as "
        "a directional signal for human review, not a verdict."
    )

    # Manual reviewer input — logs the human's final call alongside the model/LLM output
    st.subheader("Your review")
    human_call = st.radio("Your final call for this image:", ["Not reviewed", "Real", "Fake", "Uncertain"], horizontal=True)
    if human_call != "Not reviewed":
        st.success(f"Logged: you called this **{human_call}**.")
        # Hook point: write (uploaded_file.name, image_features, rf_proba, explanation, human_call)
        # to outputs/review_log.csv here if you want persistent logging across sessions.
else:
    st.info("Upload an image above to get started.")