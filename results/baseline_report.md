# PJ01 — Real-Data Classical Baseline Evaluation Report

> [!IMPORTANT]
> **Dataset Status:** Labels in `data/processed/stance_dataset.csv` are AI-assisted with spot human review.
> These results represent an initial technical and pipeline baseline, not definitive research ground truth.

## 1. Experimental Setup & Leakage Prevention

- **Dataset:** `data/processed/stance_dataset.csv` (250 comments across 23 authentic YouTube videos)
- **Target Claim:** *Climate change and global warming is a serious concern.*
- **Split Strategy:** Video-grouped (`GroupShuffleSplit`, `test_size=0.20`, `seed=21`)
- **Train Fold:** 198 comments (18 videos)
- **Test Fold:** 52 comments (5 videos)
- **Leakage Check:** `train_videos ∩ test_videos = ∅` (Zero overlap strictly verified)
- **Test Videos Held Out:** `['BcSH8_EnqLI', 'EXP1XkAN528', 'MlbKTUkifLI', 'ZBpD-ocIi0g', 'nH6gFgTpnYU']`

## 2. Model Architecture

- **Pipeline:** `scikit-learn` Pipeline
- **Feature Extraction:** Word-level TF-IDF (unigrams, `analyzer='word'`, fitted strictly on training text)
- **Vocabulary Size:** 1054 words
- **Classifier:** Multinomial Logistic Regression (`class_weight='balanced'`, `max_iter=1000`)
- **Solver Iterations:** 16 (Converged cleanly without warnings)

## 3. Evaluation Metrics

| Metric | Overall Test (52 samples) | Hindi Slice (21 samples) | Bengali Slice (31 samples) |
| :--- | :---: | :---: | :---: |
| **Primary: Macro-F1** | **0.3979** | **0.4370** | **0.3294** |
| Accuracy | 0.4808 | 0.4762 | 0.4839 |
| Micro-F1 | 0.4808 | 0.4762 | 0.4839 |

### Per-Class Performance (Overall Test)

| Stance Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| `Favour` | 0.4286 | 0.4737 | 0.4500 | 19 |
| `Against` | 0.1667 | 0.1667 | 0.1667 | 6 |
| `None` | 0.6000 | 0.5556 | 0.5769 | 27 |

### Confusion Matrix

Label Order: `['Favour', 'Against', 'None']`

```text
                 Pred: Favour       Pred: Against      Pred: None
True: Favour     9                 1                 9
True: Against    4                 1                 1
True: None       8                 4                 15
```

## 4. Key Limitations & Analysis

1. **Minority Class Sparsity (`Against`):** `Against` represents ~10% of the dataset (6 in test). While balanced class weighting allows non-zero recall, true skepticism is expressed with subtle rhetorical devices that simple unigrams struggle to capture.
2. **Cross-Video Lexical Shift:** Because test comments originate from completely unseen YouTube videos, vocabulary that is specific to one video's narrator or setting does not appear in the test set. This proves zero-leakage evaluation is genuinely rigorous.
3. **Language Difference:** Hindi comments achieved higher Macro-F1 (0.4370) than Bengali (0.3294), where the single `Against` test comment was misclassified into `Favour`.
