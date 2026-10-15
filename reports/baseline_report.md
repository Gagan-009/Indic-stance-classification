# PJ01 — Real-Data Classical Baseline Evaluation Report

**Model Architecture:** Word TF-IDF (1–2 n-grams) + Character TF-IDF (3–5 char_wb n-grams) + Balanced Multinomial Logistic Regression

## 1. Experimental Setup & Leakage Prevention

- **Dataset Source:** `data/processed/cleaned_stance_dataset.csv` (Real YouTube comments)
- **Total Samples:** 72
- **Split Strategy:** Group-by-Video ID (ensures comments from test videos never appear in training fold)
- **Train Videos (4):** `['BBC_BN_SUND01', 'BBC_HI_CLIM01', 'DW_HI_WARM02', 'SOMOY_BN_CLIM02']` (48 samples)
- **Test Videos (2):** `['NDTV_HI_ENV03', 'PROTHOM_BN_ENV03']` (24 samples)
- **Leakage Check:** `train_videos ∩ test_videos = ∅` (Zero overlap verified)

## 2. Test Performance Summary

| Metric | Overall Test | Hindi (`hi`) Slice | Bengali (`bn`) Slice |
| :--- | :---: | :---: | :---: |
| **Primary: Macro-F1** | **0.5724** | **0.6278** | **0.5172** |
| Accuracy | 0.5833 | 0.6667 | 0.5000 |
| Micro-F1 | 0.5833 | 0.6667 | 0.5000 |
| Evaluated Samples | 24 | 12 | 12 |

## 3. Per-Class Test Performance (Overall)

| Stance Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| `Favour` | 0.5455 | 0.7500 | 0.6316 | 8 |
| `Against` | 0.3333 | 0.2500 | 0.2857 | 8 |
| `None` | 0.8571 | 0.7500 | 0.8000 | 8 |

## 4. Confusion Matrix

Labels Order: `['Favour', 'Against', 'None']`

```text
             Pred: Against  Pred: Favour  Pred: None
True: Against     6             2             0
True: Favour      5             2             1
True: None        0             2             6
```

## 5. Error Analysis & Findings

1. **`None` Class Detection:** High precision and recall (F1 = 0.80) because general viewer comments (compliments, greetings, channel feedback) contain distinct non-environmental vocabulary.
2. **`Favour` Class Detection:** Solid recall (0.75, F1 = 0.63) driven by frequent climate keywords (ग्लेशियर, प्रदूषण, জলবায়ু, বন্যা, উষ্ণায়ন).
3. **`Against` Class Detection:** Lowest F1 (0.29), reflecting the subtle rhetorical strategies of climate denial/skepticism (attributing changes to natural cycles, questioning motives, or accusing media of hype without using overt denial keywords).
4. **Language Differences:** Hindi achieved higher Macro-F1 (0.6278) than Bengali (0.5172), primarily due to higher lexical overlap across Hindi climate videos.
