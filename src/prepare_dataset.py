import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.text_cleaner import clean_indic_text

VALID_LABELS = {"Favour", "Against", "None"}
VALID_LANGUAGES = {"hi", "bn"}
REQUIRED_COLUMNS = ["id", "video_id", "text", "stance"]

ANNOTATION_COLUMNS = [
    "comment_id",
    "video_id",
    "author_display_name",
    "published_at",
    "like_count",
    "detected_language",
    "raw_text",
    "clean_text",
    "stance",
    "provenance",
    "target_claim",
]

CANDIDATE_COLUMNS = [
    "comment_id",
    "video_id",
    "author_display_name",
    "published_at",
    "like_count",
    "detected_language",
    "raw_text",
    "clean_text",
    "stance",
    "annotation_notes",
    "provenance",
    "target_claim",
]

_HINGLISH_PATTERN = re.compile(
    r"\b(kya|hai|ye|yeh|ho|ki|ko|ka|ke|se|me|mein|par|aur|bhi|nahi|nahin|bhai|sab|karo|hota|hoti|hote|apne|kuch|hum|aap|unhe|inhe|kar|raha|rahe|rahi|modi|desh|sarkar|log|baat|woh)\b",
    re.IGNORECASE,
)
_BANGLISH_PATTERN = re.compile(
    r"\b(amar|amra|tumi|apni|koro|korche|ache|hobe|ei|oi|she|aar|bhalo|keno|kichu|shob|bangla|desh|manush|jani|kintu)\b",
    re.IGNORECASE,
)


def categorize_other_script_comment(text: str) -> str:
    """Categorize comments not dominated by Devanagari or Bengali script.

    Categories:
    - 'romanized_hinglish': Latin script with common Hindi phonetic tokens.
    - 'romanized_banglish': Latin script with common Bengali phonetic tokens.
    - 'noise_or_url': Empty, link-only, or punctuation/symbol only.
    - 'english_or_other_latin': English or general Latin-script text.
    """
    if not isinstance(text, str):
        return "noise_or_url"
    stripped = re.sub(r"https?://\S+", "", text).strip()
    if not stripped or len(stripped) < 3:
        return "noise_or_url"
    if _HINGLISH_PATTERN.search(text):
        return "romanized_hinglish"
    if _BANGLISH_PATTERN.search(text):
        return "romanized_banglish"
    return "english_or_other_latin"


def detect_indic_language(text: str, fallback_lang: str = "hi") -> str:
    """Identify Indic language based on Unicode character block distributions.

    Devanagari block: U+0900 to U+097F (Hindi)
    Bengali block:    U+0980 to U+09FF (Bengali / Bangla)

    Args:
        text: Input text string.
        fallback_lang: Language code to return if no Indic script characters are present.

    Returns:
        'hi' if Devanagari characters dominate, 'bn' if Bengali characters dominate.
    """
    if not isinstance(text, str):
        return fallback_lang

    hi_count = sum(1 for char in text if "\u0900" <= char <= "\u097F")
    bn_count = sum(1 for char in text if "\u0980" <= char <= "\u09FF")

    if hi_count > bn_count:
        return "hi"
    elif bn_count > hi_count:
        return "bn"
    return fallback_lang


def clean_and_prepare_dataset(
    raw_path: str | Path = "data/raw/real_youtube_comments_pilot.csv",
    processed_path: str | Path = "data/processed/cleaned_stance_dataset.csv",
    report_path: str | Path = "reports/data_quality_report.md",
) -> Tuple[pd.DataFrame, Dict]:
    """Execute reproducible data cleaning, normalization, and quality validation.

    Process:
    1. Ingest raw CSV with keep_default_na=False to protect 'None' stance label.
    2. Check required schema ('id', 'video_id', 'text', 'stance').
    3. Preserve original text while computing normalized clean_text (NFKC, whitespace).
    4. Detect and verify script language ('hi' vs 'bn') without destructive transliteration.
    5. Detect and remove invalid stance labels or empty records.
    6. Detect exact duplicate records without silently removing distinct comments.
    7. Track video IDs to support group-based train/val/test splitting.
    8. Export cleaned dataset to data/processed/.
    9. Write comprehensive data quality report to reports/.

    Returns:
        Tuple of (cleaned DataFrame, quality report dictionary).
    """
    raw_file = Path(raw_path)
    if not raw_file.is_file():
        raise FileNotFoundError(f"Raw dataset file not found at: {raw_file.resolve()}")

    df = pd.read_csv(raw_file, encoding="utf-8", keep_default_na=False)

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Raw dataset missing required columns: {missing_cols}")

    initial_count = len(df)
    report = {
        "raw_filepath": str(raw_file.resolve()),
        "processed_filepath": str(Path(processed_path).resolve()),
        "initial_rows": initial_count,
        "empty_text_dropped": 0,
        "invalid_label_dropped": 0,
        "duplicate_rows_dropped": 0,
        "language_mismatches_corrected": 0,
        "final_rows": 0,
        "class_distribution": {},
        "language_distribution": {},
        "video_distribution": {},
        "provenance_flags": list(df["provenance"].unique()) if "provenance" in df.columns else [],
    }

    # 1. Clean and normalize text
    df["raw_text"] = df["text"].astype(str)
    df["clean_text"] = df["raw_text"].apply(clean_indic_text)

    # 2. Drop empty / blank texts
    empty_mask = df["clean_text"].str.strip() == ""
    report["empty_text_dropped"] = int(empty_mask.sum())
    df = df[~empty_mask].copy()

    # 3. Handle invalid labels
    invalid_label_mask = ~df["stance"].isin(VALID_LABELS)
    report["invalid_label_dropped"] = int(invalid_label_mask.sum())
    df = df[~invalid_label_mask].copy()

    # 4. Check & verify language using Unicode script detection
    for idx, row in df.iterrows():
        declared_lang = str(row.get("language", "")).strip()
        detected_lang = detect_indic_language(row["clean_text"], fallback_lang=declared_lang)
        if declared_lang not in VALID_LANGUAGES:
            df.at[idx, "language"] = detected_lang
            report["language_mismatches_corrected"] += 1
        elif declared_lang != detected_lang and detected_lang in VALID_LANGUAGES:
            # Script dominance overrides incorrect tag if detected
            df.at[idx, "language"] = detected_lang
            report["language_mismatches_corrected"] += 1

    # 5. Handle exact duplicate records (duplicate ID or duplicate (video_id, clean_text))
    # Deduplicate exact spam or double-fetched rows while keeping distinct comments
    duplicate_mask = df.duplicated(subset=["id"], keep="first") | df.duplicated(
        subset=["video_id", "clean_text"], keep="first"
    )
    report["duplicate_rows_dropped"] = int(duplicate_mask.sum())
    df = df[~duplicate_mask].copy()

    # 6. Aggregate distributions
    report["final_rows"] = len(df)
    report["class_distribution"] = df["stance"].value_counts().to_dict()
    report["language_distribution"] = df["language"].value_counts().to_dict()
    report["video_distribution"] = df["video_id"].value_counts().to_dict()

    # 7. Export cleaned dataset
    dest_path = Path(processed_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest_path, index=False, encoding="utf-8")

    # 8. Write Markdown Data Quality Report
    rep_dest = Path(report_path)
    rep_dest.parent.mkdir(parents=True, exist_ok=True)
    with open(rep_dest, "w", encoding="utf-8") as f:
        f.write("# PJ01 — Real Data Quality and Preprocessing Report\n\n")
        f.write(f"- **Source File:** `{raw_file.name}`\n")
        f.write(f"- **Cleaned Output:** `{dest_path.name}`\n")
        f.write(f"- **Initial Raw Records:** {report['initial_rows']}\n")
        f.write(f"- **Final Valid Records:** {report['final_rows']}\n")
        f.write(f"- **Dropped Empty Records:** {report['empty_text_dropped']}\n")
        f.write(f"- **Dropped Invalid Labels:** {report['invalid_label_dropped']}\n")
        f.write(f"- **Dropped Exact Duplicates:** {report['duplicate_rows_dropped']}\n")
        f.write(f"- **Language Corrections:** {report['language_mismatches_corrected']}\n\n")

        f.write("## 1. Class Distribution\n\n")
        f.write("| Stance Class | Count | Percentage |\n| :--- | :---: | :---: |\n")
        for cls, cnt in report["class_distribution"].items():
            pct = (cnt / report["final_rows"]) * 100
            f.write(f"| `{cls}` | {cnt} | {pct:.1f}% |\n")

        f.write("\n## 2. Language Distribution\n\n")
        f.write("| Language Code | Script | Count | Percentage |\n| :--- | :--- | :---: | :---: |\n")
        for lang, cnt in report["language_distribution"].items():
            script_name = "Devanagari (Hindi)" if lang == "hi" else "Bengali (Bangla)"
            pct = (cnt / report["final_rows"]) * 100
            f.write(f"| `{lang}` | {script_name} | {cnt} | {pct:.1f}% |\n")

        f.write("\n## 3. Video Source Distribution (for Leakage Prevention)\n\n")
        f.write("| Video ID | Count | Percentage |\n| :--- | :---: | :---: |\n")
        for vid, cnt in report["video_distribution"].items():
            pct = (cnt / report["final_rows"]) * 100
            f.write(f"| `{vid}` | {cnt} | {pct:.1f}% |\n")

        f.write("\n## 4. Provenance & Integrity Statement\n\n")
        f.write(f"- **Provenance Flags:** `{report['provenance_flags']}`\n")
        f.write("- **Text Integrity:** Original text strings and scripts are preserved in `raw_text`, while `clean_text` applies Unicode NFKC normalization and whitespace standardization without destructive transliteration.\n")
        f.write("- **Split Readiness:** `video_id` is validated across all rows to guarantee that video-grouped train/val/test splits can be strictly enforced.\n")

    print(f"Data preparation complete. Saved {len(df)} cleaned records to {dest_path}")
    print(f"Data quality report saved to {rep_dest}")

    return df, report


def export_indic_annotation_pool(
    harvested_path: str | Path = PROJECT_ROOT / "data" / "raw" / "harvested_youtube_comments.csv",
    output_path: str | Path = PROJECT_ROOT / "data" / "raw" / "to_annotate_indic_comments.csv",
) -> Tuple[pd.DataFrame, Dict]:
    """Filter harvested YouTube comments to Indic script comments (Devanagari & Bengali) for annotation.

    Guarantees:
    - Preserves both 'raw_text' and 'clean_text'.
    - Preserves all provenance: comment_id, video_id, author_display_name, published_at, like_count, provenance, target_claim.
    - Does NOT alter or overwrite harvested_youtube_comments.csv.
    - Does NOT drop short, generic, or ambiguous comments.
    - Leaves 'stance' unpopulated (empty string) for human annotator input.
    - Separately analyzes and reports counts for Romanized/Hinglish/Banglish and other scripts.

    Returns:
        Tuple of (pool_dataframe, audit_summary_dict).
    """
    src_file = Path(harvested_path)
    if not src_file.is_file():
        raise FileNotFoundError(f"Harvested comments file not found at: {src_file.resolve()}")

    df = pd.read_csv(src_file, encoding="utf-8", keep_default_na=False)

    required_src_cols = ["comment_id", "video_id", "raw_text", "clean_text", "detected_language"]
    missing = [c for c in required_src_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Harvested dataset missing required columns: {missing}")

    total_harvested = len(df)

    # 1. Separate native Indic scripts from other scripts
    indic_mask = df["detected_language"].isin(["hi", "bn"])
    indic_df = df[indic_mask].copy()
    other_df = df[~indic_mask].copy()

    # 2. Detailed audit of other scripts (do not silently treat as Hindi or Bengali)
    other_categories = other_df["clean_text"].apply(categorize_other_script_comment).value_counts().to_dict()

    # 3. Ensure stance column exists and is empty string (no automated labelling)
    indic_df["stance"] = ""

    # Ensure all required annotation columns are present and well-ordered
    for col in ANNOTATION_COLUMNS:
        if col not in indic_df.columns:
            indic_df[col] = ""

    indic_df = indic_df[ANNOTATION_COLUMNS].copy()

    # 4. Save to destination
    dest_file = Path(output_path)
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    indic_df.to_csv(dest_file, index=False, encoding="utf-8")

    summary = {
        "harvested_source_file": str(src_file.resolve()),
        "output_file": str(dest_file.resolve()),
        "total_harvested_comments": total_harvested,
        "indic_script_pool_count": len(indic_df),
        "language_distribution": indic_df["detected_language"].value_counts().to_dict(),
        "video_distribution_hindi": indic_df[indic_df["detected_language"] == "hi"]["video_id"].value_counts().to_dict(),
        "video_distribution_bengali": indic_df[indic_df["detected_language"] == "bn"]["video_id"].value_counts().to_dict(),
        "other_script_total": len(other_df),
        "other_script_breakdown": other_categories,
    }

    return indic_df, summary


def sample_annotation_candidates(
    pool_input: Union[pd.DataFrame, str, Path] = PROJECT_ROOT / "data" / "raw" / "to_annotate_indic_comments.csv",
    output_path: str | Path = PROJECT_ROOT / "data" / "raw" / "annotation_candidate_250.csv",
    target_per_lang: int = 125,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a reproducible annotation candidate sample balanced across languages and distributed across videos.

    Guarantees:
    - Exactly target_per_lang comments for 'hi' and target_per_lang for 'bn' (default 125 + 125 = 250).
    - Proportional video stratification ensuring all represented videos get at least 1 sample if available.
    - Sampling does NOT use stance labels or model predictions.
    - Fully reproducible with fixed random_state.
    - Preserves both 'raw_text' and 'clean_text', all provenance fields and empty 'stance'.

    Returns:
        Sampled candidate DataFrame.
    """
    if isinstance(pool_input, (str, Path)):
        pool_file = Path(pool_input)
        if not pool_file.is_file():
            raise FileNotFoundError(f"Annotation pool file not found at: {pool_file.resolve()}")
        pool_df = pd.read_csv(pool_file, encoding="utf-8", keep_default_na=False)
    else:
        pool_df = pool_input.copy()

    samples = []
    for lang in ["hi", "bn"]:
        lang_df = pool_df[pool_df["detected_language"] == lang].copy()
        if len(lang_df) < target_per_lang:
            raise ValueError(
                f"Requested {target_per_lang} samples for language '{lang}', but only {len(lang_df)} available."
            )

        v_counts = lang_df["video_id"].value_counts()
        total_lang = len(lang_df)

        # Proportional allocation across source videos
        alloc = (v_counts / total_lang * target_per_lang).round().astype(int)

        # Ensure every video with comments gets at least 1 comment sampled for diversity
        for vid in v_counts.index:
            if alloc[vid] == 0 and v_counts[vid] > 0:
                alloc[vid] = 1

        # Adjust difference to hit exact target_per_lang
        diff = int(alloc.sum() - target_per_lang)
        if diff > 0:
            for vid in alloc.sort_values(ascending=False).index:
                deduct = min(diff, alloc[vid] - 1)
                alloc[vid] -= deduct
                diff -= deduct
                if diff == 0:
                    break
        elif diff < 0:
            for vid in alloc.sort_values(ascending=False).index:
                add = min(-diff, v_counts[vid] - alloc[vid])
                alloc[vid] += add
                diff += add
                if diff == 0:
                    break

        lang_samples = []
        for vid, n in alloc.items():
            if n <= 0:
                continue
            v_rows = lang_df[lang_df["video_id"] == vid]
            sampled = v_rows.sample(n=min(n, len(v_rows)), random_state=random_state)
            lang_samples.append(sampled)

        combined_lang = pd.concat(lang_samples, ignore_index=True)
        samples.append(combined_lang)

    final_df = (
        pd.concat(samples, ignore_index=True)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )

    # Ensure strict column ordering
    for col in CANDIDATE_COLUMNS:
        if col not in final_df.columns:
            final_df[col] = ""
    final_df = final_df[CANDIDATE_COLUMNS].copy()

    dest_file = Path(output_path)
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(dest_file, index=False, encoding="utf-8")

    return final_df


if __name__ == "__main__":
    indic_pool, rep = export_indic_annotation_pool()
    candidates = sample_annotation_candidates(indic_pool)
    print(f"Exported {len(indic_pool)} Indic comments to {rep['output_file']}")
    print(f"Sampled {len(candidates)} candidates to data/raw/annotation_candidate_250.csv")
