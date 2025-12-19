import streamlit as st
import numpy as np
import json
from PIL import Image
import tensorflow as tf

# -----------------------
# Load model & metadata
# -----------------------
MODEL_PATH = "models/brain_tumor_multiclass.h5"
META_PATH = "models/meta.json"

model = tf.keras.models.load_model(MODEL_PATH)
with open(META_PATH) as f:
    class_indices = json.load(f)["class_indices"]

labels = {v: k for k, v in class_indices.items()}

# -----------------------
# Thresholds (class-specific)
# -----------------------
THRESHOLDS = {
    "no_tumor": 0.85,   # temiz demek için daha yüksek eşik
    "meningioma": 0.60,
    "glioma": 0.55,
    "pituitary": 0.55
}

# -----------------------
# UI
# -----------------------
st.set_page_config(page_title="Brain Tumor MRI", layout="centered")
st.title("🧠 Brain Tumor MRI Classification")

patient = st.text_input("Patient Name")
date = st.date_input("Date of Scan")
uploaded = st.file_uploader("Upload MRI Scan", type=["jpg", "png", "jpeg"])

# -----------------------
# Helper
# -----------------------
def preprocess(img: Image.Image):
    img = img.convert("RGB").resize((224, 224))
    x = np.array(img) / 255.0
    return np.expand_dims(x, axis=0)

# -----------------------
# Prediction
# -----------------------
if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded MRI", use_column_width=True)

    x = preprocess(image)
    preds = model.predict(x, verbose=0)[0]

    # --- AŞAMA 1: Tümör var mı? ---
    no_idx = class_indices["no_tumor"]
    tumor_prob = 1.0 - preds[no_idx]

    st.subheader("Results:")

    if preds[no_idx] >= THRESHOLDS["no_tumor"]:
        # Güvenli şekilde temiz
        st.success(f"🟢 No Tumor (%{preds[no_idx]*100:.1f})")
    else:
        # --- AŞAMA 2: Tümör tipi ---
        # no_tumor'u devre dışı bırak
        tumor_preds = preds.copy()
        tumor_preds[no_idx] = 0.0

        tumor_idx = int(np.argmax(tumor_preds))
        label = labels[tumor_idx]
        confidence = float(tumor_preds[tumor_idx])

        if confidence >= THRESHOLDS[label]:
            st.error(f"🔴 {label.upper()} Tumor Risk (%{confidence*100:.1f})")
        else:
            st.warning(
                "⚠️ Undecisive\n"
                f"Guess: {label} (%{confidence*100:.1f})\n"
                "Consult a doctor."
            )

    # --- Debug (isteğe bağlı) ---
    with st.expander("Detail Probabilities"):
        for i, p in enumerate(preds):
            st.write(f"{labels[i]}: %{p*100:.2f}")
