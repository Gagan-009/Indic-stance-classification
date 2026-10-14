"""Streamlit demonstration interface for Indic Stance Classification (Milestone 1)."""

import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from src.pipeline import load_pipeline
from src.text_cleaner import clean_indic_text
from src.train import train_and_evaluate

MODEL_PATH = PROJECT_ROOT / "models" / "stance_pipeline.joblib"
METRICS_PATH = PROJECT_ROOT / "models" / "evaluation_metrics.json"
DATASET_PATH = PROJECT_ROOT / "data" / "illustrative_dataset.csv"

# Predefined illustrative examples for quick testing
SAMPLE_EXAMPLES = [
    {
        "label": "Hindi - Favour (Supports concern)",
        "text": "ग्लोबल वॉर्मिंग सचमुच एक बहुत बड़ा संकट है, हमें तुरंत कड़े कदम उठाने चाहिए।",
    },
    {
        "label": "Hindi - Against (Skeptical / Denies emergency)",
        "text": "जलवायु परिवर्तन सिर्फ पश्चिमी देशों का एक झूठा प्रोपेगैंडा और प्रचार है।",
    },
    {
        "label": "Hindi - None (Unrelated comment)",
        "text": "यह वीडियो बहुत अच्छी जानकारी देता है, शेयर करने के लिए धन्यवाद।",
    },
    {
        "label": "Bengali - Favour (Supports concern)",
        "text": "জলবায়ু পরিবর্তন সত্যি একটি মারাত্মক সমস্যা, অবিলম্বে পদক্ষেপ নেওয়া দরকার।",
    },
    {
        "label": "Bengali - Against (Skeptical / Denies emergency)",
        "text": "জলবায়ু পরিবর্তন হলো উন্নত দেশগুলোর তৈরি করা একটি মিথ্যা প্রচার ও নাটক।",
    },
    {
        "label": "Bengali - None (Unrelated comment)",
        "text": "এই ভিডিওটির উপস্থাপনা খুব সুন্দর হয়েছে, তথ্য দেওয়ার জন্য ধন্যবাদ।",
    },
]


def load_cached_pipeline():
    """Load model pipeline or train on demand if artifact is missing or incompatible."""
    if MODEL_PATH.is_file():
        try:
            return load_pipeline(MODEL_PATH)
        except Exception:
            # Handle potential joblib / scikit-learn version mismatch on target machine
            pass

    with st.spinner("Preparing baseline pipeline..."):
        train_and_evaluate(data_path=DATASET_PATH, model_save_path=MODEL_PATH, metrics_save_path=METRICS_PATH)
    return load_pipeline(MODEL_PATH)


def load_metrics():
    """Load saved evaluation metrics JSON."""
    if METRICS_PATH.is_file():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def main():
    st.set_page_config(
        page_title="Indic Stance Classification - Milestone 1",
        page_icon="🌍",
        layout="wide",
    )

    st.title("PJ01 — Indic Stance Classification")
    st.subheader("Milestone 1 Demonstration: Hindi & Bengali Stance Detection")

    # Target Claim Box
    st.info(
        "🎯 **Target Claim:**  \n"
        "> *\"Climate change and global warming is a serious concern.\"*  \n\n"
        "**Classes:** `Favour` (agrees claim is serious) | `Against` (opposes/denies claim) | `None` (neutral/unrelated)"
    )

    # Mandatory Synthetic Provenance Warning
    st.warning(
        "⚠️ **NOTICE — DEMONSTRATION ON SYNTHETIC DATA ONLY**  \n"
        "This model is trained strictly on an illustrative synthetic dataset (`data/illustrative_dataset.csv`) "
        "created for Milestone 1 pipeline verification. No official competition or real-world social media "
        "dataset was available. **These predictions and metrics must not be represented as real-world academic research performance.**"
    )

    # Sidebar: Metrics & System Info
    with st.sidebar:
        st.header("Pipeline Status & Metrics")
        metrics_data = load_metrics()

        if metrics_data:
            ov = metrics_data.get("overall_validation_metrics", {})
            st.metric("Primary Metric (Macro-F1)", f"{ov.get('macro_f1', 0.0):.4f}")
            st.metric("Validation Accuracy", f"{ov.get('accuracy', 0.0):.4f}")
            st.metric("Total Validation Samples", ov.get("total_samples", 0))

            st.write("---")
            st.write("**Per-Class Validation F1:**")
            per_cls = ov.get("per_class", {})
            f1_table = {cls: f"{data.get('f1-score', 0):.2f}" for cls, data in per_cls.items()}
            st.json(f1_table)

            st.write("---")
            st.write("**Language Breakdown (Macro-F1):**")
            lang_metrics = metrics_data.get("language_validation_metrics", {})
            for lang, l_data in lang_metrics.items():
                st.write(f"- `{lang.upper()}`: {l_data.get('macro_f1', 0):.4f}")
        else:
            st.info("Metrics not found. Run training to generate evaluation scores.")

        if st.button("Retrain Pipeline Now"):
            with st.spinner("Retraining pipeline..."):
                train_and_evaluate(data_path=DATASET_PATH, model_save_path=MODEL_PATH, metrics_save_path=METRICS_PATH)
            st.success("Pipeline retrained!")
            st.rerun()

        st.write("---")
        st.caption("Architecture: Unicode Normalization + Word & Char TF-IDF + Balanced Logistic Regression.")

    # Main Area: Live Prediction
    pipeline = load_cached_pipeline()

    st.markdown("### Test a Comment")

    # Example picker
    selected_example_label = st.selectbox(
        "Select an illustrative sample comment or enter custom text below:",
        options=["(Custom Input)"] + [ex["label"] for ex in SAMPLE_EXAMPLES],
    )

    default_text = ""
    if selected_example_label != "(Custom Input)":
        for ex in SAMPLE_EXAMPLES:
            if ex["label"] == selected_example_label:
                default_text = ex["text"]
                break

    input_text = st.text_area(
        "Comment Text (Hindi Devanagari or Bengali script):",
        value=default_text,
        height=100,
        placeholder="यहाँ टिप्पणी दर्ज करें / এখানে মন্তব্য লিখুন...",
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        classify_button = st.button("Classify Stance", type="primary")

    if classify_button or (selected_example_label != "(Custom Input)" and input_text):
        cleaned = clean_indic_text(input_text)
        if not cleaned:
            st.error("Please enter a non-empty text comment.")
        else:
            prediction = pipeline.predict([cleaned])[0]
            probabilities = pipeline.predict_proba([cleaned])[0]
            classes = pipeline.classes_

            st.markdown("#### Classification Result")

            badge_color = {
                "Favour": "🟢",
                "Against": "🔴",
                "None": "⚪",
            }.get(prediction, "🔹")

            st.markdown(f"### {badge_color} Predicted Stance: **{prediction}**")

            # Probability Distribution
            st.write("**Model Probabilities:**")
            prob_dict = {cls: float(prob) for cls, prob in zip(classes, probabilities)}
            prob_df = pd.DataFrame(
                list(prob_dict.items()),
                columns=["Stance Class", "Probability"],
            ).sort_values("Probability", ascending=False)

            st.dataframe(
                prob_df.style.format({"Probability": "{:.2%}"}),
                use_container_width=True,
                hide_index=True,
            )

            st.caption(f"Normalized input: `{cleaned}`")


if __name__ == "__main__":
    main()
