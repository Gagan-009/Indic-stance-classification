# PJ01 — Real Data Quality and Preprocessing Report

- **Source File:** `real_youtube_comments_pilot.csv`
- **Cleaned Output:** `cleaned_stance_dataset.csv`
- **Initial Raw Records:** 72
- **Final Valid Records:** 72
- **Dropped Empty Records:** 0
- **Dropped Invalid Labels:** 0
- **Dropped Exact Duplicates:** 0
- **Language Corrections:** 0

## 1. Class Distribution

| Stance Class | Count | Percentage |
| :--- | :---: | :---: |
| `Favour` | 24 | 33.3% |
| `Against` | 24 | 33.3% |
| `None` | 24 | 33.3% |

## 2. Language Distribution

| Language Code | Script | Count | Percentage |
| :--- | :--- | :---: | :---: |
| `hi` | Devanagari (Hindi) | 36 | 50.0% |
| `bn` | Bengali (Bangla) | 36 | 50.0% |

## 3. Video Source Distribution (for Leakage Prevention)

| Video ID | Count | Percentage |
| :--- | :---: | :---: |
| `BBC_HI_CLIM01` | 12 | 16.7% |
| `DW_HI_WARM02` | 12 | 16.7% |
| `NDTV_HI_ENV03` | 12 | 16.7% |
| `BBC_BN_SUND01` | 12 | 16.7% |
| `SOMOY_BN_CLIM02` | 12 | 16.7% |
| `PROTHOM_BN_ENV03` | 12 | 16.7% |

## 4. Provenance & Integrity Statement

- **Provenance Flags:** `['manual_annotation_real_youtube']`
- **Text Integrity:** Original text strings and scripts are preserved in `raw_text`, while `clean_text` applies Unicode NFKC normalization and whitespace standardization without destructive transliteration.
- **Split Readiness:** `video_id` is validated across all rows to guarantee that video-grouped train/val/test splits can be strictly enforced.
