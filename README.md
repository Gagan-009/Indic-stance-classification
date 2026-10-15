# Indic Stance Classification: Hindi & Bengali YouTube Comments

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests: 23 Passed](https://img.shields.io/badge/tests-23%20passed-brightgreen.svg)]()

A natural language processing repository for stance classification in low-resource Indic languages (Hindi and Bengali). The project evaluates whether viewer comments on YouTube support, oppose, or express no stance toward the claim:

> **Target Claim:** *"Climate change and global warming is a serious concern."*

---

## 1. Problem Formulation & Classes

Stance classification determines the author's point of view toward a specific target claim, which is distinct from generic sentiment (e.g., whether a commenter likes a video or dislikes an anchor). Each comment is categorized into one of three stance classes:

| Class | Definition | Typical Examples |
| :--- | :--- | :--- |
| **`Favour`** | Expresses concern about global warming, highlights ecological crises, pollution, rising temperatures, deforestation, or calls for climate action. | *"पेड़ लगाओ, पर्यावरण और भविष्य की पीढ़ियों को बचाओ।"* / *"জলবায়ু পরিবর্তন সত্যিই একটি মারাত্মক বিপদ।"* |
| **`Against`** | Rejects, denies, mocks, or minimizes climate change concerns (e.g., claiming it is natural, a hoax, or media hype). | *"मौसम तो हर साल बदलता है, ये सब मीडिया का ढोंग है।"* / *"এটি সম্পূর্ণ মিথ্যা এবং একটি ব্যবসায়িক প্রচার।"* |
| **`None`** | Off-topic comments, channel compliments, neutral factual queries, political remarks without climate stance, or ambiguous text. | *"দাদা আপনার ভিডিও এডিটিং দারুণ।"* / *"आज तापमान कितना है?"* |

---

## 2. Language Coverage & Indic Script Handling

The project focuses on two major Indo-Aryan languages:
1. **Hindi (`hi`)**: Written in the **Devanagari** script (`U+0900`–`U+097F`).
2. **Bengali (`bn`)**: Written in the **Bengali** script (`U+0980`–`U+09FF`).

### Script Identification & Conservative Preprocessing
- **Language Detection:** Implemented in [`src/prepare_dataset.py`](src/prepare_dataset.py) via Unicode block character frequency. Comments are classified as `hi` or `bn` only when native script characters dominate. Romanized comments (Hinglish/Banglish) and other scripts are explicitly categorized and separated, not silently conflated with native text.
- **Text Normalization:** Unicode NFKC normalization, zero-width space removal, and whitespace standardization ([`src/text_cleaner.py`](src/text_cleaner.py)). Aggressive transformations (stemming, stopword removal, or ASCII stripping) are avoided to preserve Indic negation markers and inflectional suffixes.

---

## 3. Dataset Description & Provenance

### Evolution: Synthetic Pilot vs. Authentic YouTube Dataset

| Dimension | Milestone 1 (Synthetic Pilot) | Milestone 2 (Current Real-Data Baseline) |
| :--- | :--- | :--- |
| **Data Source** | Hand-crafted synthetic examples | Authentic YouTube comments harvested via YouTube Data API v3 |
| **Total Rows** | 72 comments | **250 comments** |
| **Languages** | 36 Hindi, 36 Bengali | **125 Hindi (`hi`), 125 Bengali (`bn`)** |
| **Source Videos** | Synthetic mock IDs | **23 unique authentic YouTube videos** (13 Hindi, 10 Bengali) |
| **Class Distribution** | 24 Favour, 24 Against, 24 None (Balanced) | **130 None (52.0%), 94 Favour (37.6%), 26 Against (10.4%)** |
| **Purpose** | Pipeline mechanics, schema testing, Streamlit UI | **Leakage-safe classical ML benchmark evaluation** |
| **Storage Location** | `data/illustrative_dataset.csv` | Local `data/processed/stance_dataset.csv` (Split IDs tracked) |

### Annotation Methodology & Ethical Disclosure
- **AI-Assisted Annotation:** Initial draft labels and confidence scores were inferred using an LLM prompt grounded in the three-class definition.
- **Spot Human Review:** Low-confidence comments and borderline sarcastic cases were manually inspected, and verified corrections were applied to produce the final labels.
- **Ethical Caveat:** These labels are **AI-assisted with spot human review**, not fully independent double-annotated consensus ground truth. Performance metrics represent an initial technical baseline.
- **Privacy & Responsible Data Sharing:** To comply with YouTube API Terms of Service and protect author privacy, raw unmoderated comments and author display names (`@username`) are kept local. The repository tracks reproducible split assignments and metadata ([`data/processed/split_assignments.csv`](data/processed/split_assignments.csv)).

---

## 4. Leakage-Safe Video-Grouped Methodology

A critical failure mode in social media NLP is **topic and video leakage**. Comments under the same YouTube video frequently share video-specific vocabulary (e.g., discussing a specific anchor, regional river, or local weather event). Random row-level splitting allows comments from the same video into both training and test sets, inflating accuracy through memorization.

To prevent leakage:
1. **Group by `video_id`:** Comments are grouped strictly by their source video ID. Comments from any given video appear in **either** the training fold or the test fold, never both.
2. **Reproducible Split (`random_state=21`):** Selected purely based on distributional fidelity to the full dataset:
   - **Training Set:** **198 comments** across **18 videos** (79.2% of data).
   - **Test Set:** **52 comments** across **5 unseen videos** (20.8% of data).
   - **Leakage Check:** $\text{Train Videos} \cap \text{Test Videos} = \emptyset$ (strictly 0 overlapping videos).
3. **Class Representation:** Both folds maintain all three stance classes:
   - Train: 103 `None`, 75 `Favour`, 20 `Against`.
   - Test: 27 `None`, 19 `Favour`, 6 `Against`.
   - Test Videos Held Out: `BcSH8_EnqLI` (bn), `EXP1XkAN528` (bn), `MlbKTUkifLI` (hi), `ZBpD-ocIi0g` (hi), `nH6gFgTpnYU` (hi).

---

## 5. Classical ML Baseline & Results

The baseline uses a simple, interpretable scikit-learn pipeline:
- **Feature Extraction:** Word-level `TfidfVectorizer(analyzer="word", ngram_range=(1, 1), min_df=1)` fitted **strictly on training text only**. Vocabulary size: 1,054 words.
- **Classifier:** Multinomial `LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)`. Balanced class weighting prevents the model from ignoring the minority `Against` class (~10%).
- **Convergence:** The solver converged cleanly in **16 iterations** without warnings.

### Why Macro-F1 is the Primary Metric
Because the dataset is naturally imbalanced (52% `None`, 38% `Favour`, 10% `Against`), standard classification accuracy is misleading. A naive majority-class classifier would achieve 52% accuracy while completely failing on `Against`. **Macro-F1** assigns equal importance to all three classes:
$$\text{Macro-F1} = \frac{\text{F1}_{\text{Favour}} + \text{F1}_{\text{Against}} + \text{F1}_{\text{None}}}{3}$$

### Evaluation Performance Summary

| Metric | Overall Test (52 samples) | Hindi Slice (21 samples) | Bengali Slice (31 samples) |
| :--- | :---: | :---: | :---: |
| **Primary: Macro-F1** | **`0.3979`** | **`0.4370`** | **`0.3294`** |
| **Accuracy** | **`0.4808`** | **`0.4762`** | **`0.4839`** |
| **Micro-F1** | **`0.4808`** | **`0.4762`** | **`0.4839`** |

#### Per-Class Performance (Overall Held-Out Test)
| Stance Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **`Favour`** | 0.4286 | 0.4737 | **0.4500** | 19 |
| **`Against`** | 0.1667 | 0.1667 | **0.1667** | 6 |
| **`None`** | 0.6000 | 0.5556 | **0.5769** | 27 |

#### Confusion Matrix (Overall Test)
```text
                 Pred: Favour    Pred: Against    Pred: None
True: Favour          9               1               9
True: Against         4               1               1
True: None            8               4              15
```

---

## 6. Key Baseline Findings & Limitations

1. **Minority Class Sparsity (`Against`):** Climate skepticism constitutes only 10.4% of comments (26 total, 6 in test). Without balanced weights, recall was 0.0. With balanced weights, the model detects `Against` (F1 = 0.1667), but simple word unigrams struggle with nuanced rhetorical skepticism.
2. **Cross-Video Generalization:** Because test comments originate from completely unseen videos, video-specific channel vocabulary does not transfer, presenting a realistic and challenging hidden-test setting.
3. **Language Asymmetry:** Hindi achieved higher Macro-F1 (0.4370) than Bengali (0.3294), where the single `Against` test comment was misclassified into `Favour`.
4. **Sample Size:** 250 comments provide a functional software and methodology baseline, but a larger human-annotated corpus is needed for definitive research claims.

---

## 7. Repository Structure

```text
indistance/
├── data/
│   ├── illustrative_dataset.csv     # Milestone 1 synthetic dataset (72 rows)
│   ├── dataset_metadata.json        # Milestone 1 metadata & ethical disclosures
│   └── processed/
│       ├── split_assignments.csv    # Reproducible train/test split assignments (250 rows)
│       └── split_metadata.json      # Video grouping and class balance metadata
├── models/
│   ├── stance_pipeline.joblib       # Milestone 1 demo model artifact
│   └── evaluation_metrics.json      # Milestone 1 synthetic evaluation metrics
├── reports/
│   ├── baseline_report.md           # Classical baseline report
│   ├── pilot_authenticity_audit.md  # Raw data provenance audit
│   ├── data_quality_report.md       # Corpus quality analysis
│   ├── dataset_decision.md          # Dataset roadmap and decisions
│   └── transformer_readiness_report.md # Multi-GPU / CPU transformer readiness
├── results/
│   ├── baseline_metrics.json        # Real baseline test metrics
│   ├── baseline_predictions.csv    # Predictions with class probabilities (52 test rows)
│   ├── baseline_report.md           # Markdown baseline evaluation report
│   └── confusion_matrix.json        # Test confusion matrix breakdown
├── src/
│   ├── __init__.py
│   ├── collect_youtube_comments.py  # YouTube Data API v3 harvester
│   ├── data_loader.py               # Dataset loading & schema validation
│   ├── evaluate.py                  # Macro-F1, per-class metrics, confusion matrix
│   ├── pipeline.py                  # Scikit-learn Pipeline builders
│   ├── prepare_dataset.py           # Indic language identification & candidate sampling
│   ├── text_cleaner.py              # Unicode NFKC text normalization
│   ├── train.py                     # Milestone 1 synthetic training script
│   └── train_real_baseline.py       # Milestone 2 real-data baseline trainer
├── tests/
│   ├── __init__.py
│   ├── test_data_validation.py      # Schema, null, and label integrity tests
│   ├── test_pipeline.py             # Feature union and pipeline tests
│   └── test_real_data_pipeline.py   # Real data leakage, split, and results tests
├── app.py                           # Interactive Streamlit demo
├── requirements.txt                 # Project dependencies
├── .gitignore                       # Git ignore rules protecting raw data & keys
└── README.md                        # Project documentation
```

---

## 8. Environment Setup & Reproduction

### A. Environment Installation
Clone the repository and install dependencies in a Python 3.10+ virtual environment:
```powershell
git clone https://github.com/Gagan-009/Indic-stance-classification.git
cd Indic-stance-classification
pip install -r requirements.txt
```

### B. Run the Test Suite
Run the 23 automated unit tests verifying schema integrity, zero-leakage assertions, Indic language detection, and results consistency:
```powershell
python -m unittest discover tests
```

### C. Reproduce Real-Data Classical Baseline
To train the Word TF-IDF + Logistic Regression baseline on the video-grouped split and generate evaluation metrics under `results/`:
```powershell
python -m src.train_real_baseline
```
*(Note: Requires local `data/processed/stance_dataset.csv`. Split metadata and evaluation predictions are precomputed under `results/` for inspection).*

### D. Launch Interactive Streamlit Demo (Synthetic Benchmark)
To test live stance classification with pre-loaded Hindi and Bengali presets:
```powershell
streamlit run app.py
```

---

## 9. Current Status & Next Milestones

- [x] **Milestone 1:** End-to-end classification pipeline, synthetic demonstration, and interactive Streamlit web application.
- [x] **Milestone 2:** Authentic YouTube data collection, Indic script filtering, AI-assisted annotation with human review, zero-leakage video-grouped split, and classical ML baseline.
- [ ] **Milestone 3 (Upcoming):** Fine-tuning multilingual Indic pretrained transformers (e.g., Google MuRIL, IndicBERT, XLM-RoBERTa).
- [ ] **Milestone 4 (Upcoming):** Systematic comparative evaluation (Classical vs. Transformer), cross-lingual zero-shot analysis, error taxonomy, and final report.