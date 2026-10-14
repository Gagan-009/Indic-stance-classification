import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.text_cleaner import clean_indic_text

VALID_LABELS: Set[str] = {"Favour", "Against", "None"}
VALID_LANGUAGES: Set[str] = {"hi", "bn"}
REQUIRED_COLUMNS: List[str] = ["id", "language", "text", "stance"]


def load_and_validate_dataset(
    filepath: str | Path,
    min_samples_per_class: int = 3,
    drop_invalid: bool = False,
) -> Tuple[pd.DataFrame, Dict]:
    """Load and validate the stance dataset.

    Checks performed:
    1. Existence of required columns ('id', 'language', 'text', 'stance').
    2. Missing values or empty / whitespace-only text entries.
    3. Invalid stance labels (must be 'Favour', 'Against', or 'None').
    4. Duplicated IDs or duplicate text entries.
    5. Minimum sample counts per class (critical for stratified splitting).
    6. Identification of synthetic data provenance flag.

    Args:
        filepath: Path to the dataset CSV file.
        min_samples_per_class: Minimum count of examples required per stance class.
        drop_invalid: If True, drops invalid rows; if False, raises ValueError on violations.

    Returns:
        Tuple of (validated DataFrame with cleaned text, validation report dict).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If schema or validation checks fail (when drop_invalid=False).
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found at: {path.resolve()}")

    # Note: keep_default_na=False is critical because 'None' is a valid stance class
    # and pandas default na_values would convert string "None" into NaN.
    df = pd.read_csv(path, encoding="utf-8", keep_default_na=False)

    # 1. Check required schema
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {missing_cols}")

    initial_count = len(df)
    report = {
        "filepath": str(path.resolve()),
        "initial_rows": initial_count,
        "empty_text_count": 0,
        "invalid_label_count": 0,
        "invalid_lang_count": 0,
        "duplicate_id_count": 0,
        "duplicate_text_count": 0,
        "provenance_flags": list(df["provenance"].unique()) if "provenance" in df.columns else [],
        "class_distribution": {},
        "language_distribution": {},
        "passed": False,
    }

    # Clean text column first
    df["clean_text"] = df["text"].astype(str).apply(clean_indic_text)

    # 2. Check empty / blank text
    empty_mask = df["clean_text"].str.strip() == ""
    report["empty_text_count"] = int(empty_mask.sum())

    # 3. Check invalid labels
    invalid_label_mask = ~df["stance"].isin(VALID_LABELS)
    report["invalid_label_count"] = int(invalid_label_mask.sum())

    # Check languages
    invalid_lang_mask = ~df["language"].isin(VALID_LANGUAGES)
    report["invalid_lang_count"] = int(invalid_lang_mask.sum())

    # 4. Check duplicates
    duplicate_ids = df["id"].duplicated()
    report["duplicate_id_count"] = int(duplicate_ids.sum())

    duplicate_texts = df["clean_text"].duplicated()
    report["duplicate_text_count"] = int(duplicate_texts.sum())

    has_violations = (
        report["empty_text_count"] > 0
        or report["invalid_label_count"] > 0
        or report["duplicate_id_count"] > 0
    )

    if has_violations and not drop_invalid:
        issues = []
        if report["empty_text_count"] > 0:
            issues.append(f"{report['empty_text_count']} empty text record(s)")
        if report["invalid_label_count"] > 0:
            issues.append(f"{report['invalid_label_count']} invalid label(s)")
        if report["duplicate_id_count"] > 0:
            issues.append(f"{report['duplicate_id_count']} duplicate ID(s)")
        raise ValueError(f"Dataset validation failed: {'; '.join(issues)}")

    if drop_invalid and has_violations:
        valid_mask = (~empty_mask) & (~invalid_label_mask) & (~duplicate_ids)
        df = df[valid_mask].copy()

    # 5. Check class counts for stratification feasibility
    class_counts = df["stance"].value_counts().to_dict()
    report["class_distribution"] = class_counts
    report["language_distribution"] = df["language"].value_counts().to_dict()

    for label in VALID_LABELS:
        count = class_counts.get(label, 0)
        if count < min_samples_per_class:
            raise ValueError(
                f"Class '{label}' has only {count} samples; minimum required is {min_samples_per_class} for stratified splitting."
            )

    report["final_rows"] = len(df)
    report["passed"] = True

    return df, report
