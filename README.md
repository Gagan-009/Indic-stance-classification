# Indic Stance Classification

NLP project for detecting stance in Hindi and Bengali YouTube comments toward the claim:

> "Climate change and global warming is a serious concern."

## Classes

- Favour
- Against
- None

---

## Milestone 1 Demonstration (October 2026)

### 1. Data Provenance & Ethical Disclosure

> [!WARNING]
> **No official competition or human-annotated dataset has been released yet.**  
> For this Milestone 1 demonstration, an **illustrative synthetic dataset** (`data/illustrative_dataset.csv`) is provided strictly to test end-to-end pipeline functionality, data validation, stratified splitting, feature extraction, evaluation metrics (Macro-F1), and live UI testing.  
> **Do not cite or represent performance scores on this synthetic dataset as real-world academic research findings.**

- **Target Claim**: *"Climate change and global warming is a serious concern."*
- **Languages**: Hindi (`hi`, Devanagari script) and Bengali (`bn`, Bengali script).
- **Illustrative Samples**: 72 balanced examples (24 Favour, 24 Against, 24 None; 36 Hindi, 36 Bengali).
- **Metadata**: Provenance flags and constraints documented in `data/dataset_metadata.json`.

---

### 2. Pipeline Architecture

1. **Text Normalization**: Unicode NFKC normalization and whitespace standardization (`src/text_cleaner.py`).
2. **Data Integrity Checks**: Explicit verification of schema, missing values, duplicates, and class distribution (`src/data_loader.py`). Handles edge cases such as pandas parsing class label `"None"` as NaN.
3. **Leakage Prevention**: Stratified train/validation splitting is strictly performed **before** fitting feature extractors.
4. **Feature Extraction**: Combined word n-grams (1–2) and character n-grams (3–5 `char_wb`) via scikit-learn `FeatureUnion` to capture sub-word morphemes and roots in Indic scripts.
5. **Classification**: Multinomial Logistic Regression with balanced class weights (`src/pipeline.py`).
6. **Primary Evaluation Metric**: **Macro-F1** (along with Accuracy, per-class Precision/Recall/F1, and Confusion Matrix in `src/evaluate.py`).

---

### 3. Quick Start & Reproduction Commands

#### A. Environment Setup
Dependencies are listed in `requirements.txt`. Install via:
```powershell
pip install -r requirements.txt
```

#### B. Run Training & Evaluation Pipeline
Trains on the stratified training split, evaluates on the held-out validation split, prints the evaluation summary, and saves artifacts to `models/`:
```powershell
python src/train.py
```

#### C. Run Test Suite
Executes unit tests verifying data validation constraints, label rules, leakage prevention, and pipeline mechanics:
```powershell
python -m unittest discover -s tests -v
```

#### D. Launch Interactive Streamlit Demo
Launches a live web interface to classify Hindi and Bengali comments with real-time class probabilities and illustrative presets:
```powershell
streamlit run app.py
```

---

### 4. Repository Structure

```text
indistance/
├── data/
│   ├── illustrative_dataset.csv     # Illustrative synthetic benchmark dataset
│   └── dataset_metadata.json        # Provenance, schema, and ethical disclosure metadata
├── src/
│   ├── __init__.py
│   ├── text_cleaner.py              # Unicode-safe text cleaning & normalization
│   ├── data_loader.py               # Dataset loading & schema validation checks
│   ├── pipeline.py                  # Scikit-learn Pipeline (TF-IDF + Logistic Regression)
│   ├── evaluate.py                  # Macro-F1, per-class metrics, confusion matrix
│   └── train.py                     # Stratified split, training, evaluation & artifact persistence
├── tests/
│   ├── __init__.py
│   ├── test_data_validation.py      # Tests for schema, labels, empty fields, and duplicates
│   └── test_pipeline.py             # Tests for feature extraction, prediction, and metrics
├── models/
│   ├── stance_pipeline.joblib       # Persisted fitted pipeline
│   └── evaluation_metrics.json      # Evaluation metrics log
├── app.py                           # Interactive Streamlit UI
├── requirements.txt                 # Project dependencies
├── .gitignore                       # Git ignore rules
└── README.md                        # Project documentation
```