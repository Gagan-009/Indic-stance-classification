import json
import sys
from pathlib import Path
from typing import Dict, Tuple

# Ensure project root is in sys.path when running as a direct script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.model_selection import train_test_split

from src.data_loader import load_and_validate_dataset
from src.evaluate import compute_metrics, format_evaluation_summary
from src.pipeline import build_stance_pipeline, save_pipeline


def train_and_evaluate(
    data_path: str | Path = "data/illustrative_dataset.csv",
    model_save_path: str | Path = "models/stance_pipeline.joblib",
    metrics_save_path: str | Path = "models/evaluation_metrics.json",
    test_size: float = 0.25,
    random_state: int = 42,
) -> Tuple[Dict, Path]:
    """Train stance pipeline on stratified train split and evaluate on validation split.

    Process:
    1. Load and validate illustrative dataset.
    2. Split dataset with fixed random_state and stratification across (stance, language)
       BEFORE fitting feature extractors to prevent validation leakage.
    3. Fit Pipeline (Word & Char TF-IDF + Logistic Regression) on training fold only.
    4. Predict and compute Macro-F1 and per-class metrics on held-out validation fold.
    5. Persist model artifact and evaluation records.

    Returns:
        Tuple of (evaluation metrics dict, path to saved model).
    """
    print(f"[1/5] Loading and validating dataset from: {data_path}")
    df, validation_report = load_and_validate_dataset(data_path)
    print(f"      Loaded {len(df)} validated records.")
    print(f"      Classes: {validation_report['class_distribution']}")
    print(f"      Languages: {validation_report['language_distribution']}")

    # Stratify by a composite key (language + stance) to ensure balanced representation
    df["stratify_key"] = df["language"] + "_" + df["stance"]

    print(f"[2/5] Performing stratified train/validation split (test_size={test_size}, random_state={random_state})...")
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["stratify_key"],
    )

    X_train = train_df["clean_text"].values
    y_train = train_df["stance"].values
    X_val = val_df["clean_text"].values
    y_val = val_df["stance"].values

    print(f"      Train samples: {len(X_train)} | Validation samples: {len(X_val)}")

    print("[3/5] Fitting Pipeline (Word+Char TF-IDF + Balanced Logistic Regression) on TRAIN set only...")
    pipeline = build_stance_pipeline(random_state=random_state)
    pipeline.fit(X_train, y_train)

    print("[4/5] Evaluating on held-out validation set...")
    y_val_pred = pipeline.predict(X_val)
    overall_metrics = compute_metrics(y_val, y_val_pred)

    # Compute language-specific slices
    lang_metrics = {}
    for lang in ["hi", "bn"]:
        mask = (val_df["language"] == lang).values
        if mask.sum() > 0:
            lang_y_true = y_val[mask]
            lang_y_pred = y_val_pred[mask]
            lang_metrics[lang] = compute_metrics(lang_y_true, lang_y_pred)

    summary_text = format_evaluation_summary(overall_metrics)
    print("\n" + summary_text + "\n")

    # Save artifacts
    print(f"[5/5] Saving model pipeline and metrics...")
    model_path = save_pipeline(pipeline, model_save_path)

    full_output = {
        "metadata": {
            "purpose": "Milestone 1 Pipeline Demonstration",
            "is_synthetic_demonstration_data": True,
            "data_path": str(data_path),
            "random_state": random_state,
            "test_size": test_size,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "train_classes": train_df["stance"].value_counts().to_dict(),
            "val_classes": val_df["stance"].value_counts().to_dict(),
        },
        "overall_validation_metrics": overall_metrics,
        "language_validation_metrics": lang_metrics,
        "validation_report": validation_report,
    }

    metrics_dest = Path(metrics_save_path)
    metrics_dest.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_dest, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2, ensure_ascii=False)

    print(f"      Saved model artifact to: {model_path}")
    print(f"      Saved metrics JSON to: {metrics_dest.resolve()}")
    print("\n[NOTE] These metrics reflect pipeline functionality on illustrative demonstration data.")
    print("       Do not cite these scores as real-world benchmark performance.")

    return full_output, model_path


if __name__ == "__main__":
    train_and_evaluate()
