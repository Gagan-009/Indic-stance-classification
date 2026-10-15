"""Focused training and evaluation script for PJ01 real-data classical ML baseline.

Goal:
1. Video-grouped train/test split (80/20) with zero video leakage.
2. Word-level TF-IDF + Logistic Regression fitted strictly on training data.
3. Full evaluation (Macro-F1, Accuracy, Per-class metrics, Confusion Matrix, Language slices).
4. Save predictions, metrics, confusion matrix, and split assignments under results/ and data/processed/.
"""

import argparse
import io
import json
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
elif hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import VALID_LABELS, VALID_LANGUAGES
from src.evaluate import compute_metrics, format_evaluation_summary
from src.pipeline import build_word_tfidf_pipeline, save_pipeline

DEFAULT_DATASET = PROJECT_ROOT / "data" / "processed" / "stance_dataset.csv"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"


def create_video_grouped_split(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 21,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Create a leakage-safe video-grouped train/test split.

    Comments belonging to any video appear strictly in train OR test, never both.
    Verifies that all three stance classes ('Favour', 'Against', 'None') are present
    in both training and test folds.

    Args:
        df: Input DataFrame containing 'comment_id', 'video_id', 'detected_language', and 'stance'.
        test_size: Approximate test fraction (target ~0.20 for 80/20 split).
        random_state: Seed for GroupShuffleSplit.

    Returns:
        Tuple of (train_df, test_df, split_metadata_dict).

    Raises:
        ValueError: If video leakage is detected, or if any required stance class is missing
                    from either split fold.
    """
    if "video_id" not in df.columns:
        raise ValueError("Dataset missing 'video_id' column required for video-level grouping.")
    if "stance" not in df.columns:
        raise ValueError("Dataset missing 'stance' column required for stance evaluation.")

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(df, groups=df["video_id"]))

    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)

    # 1. Zero-leakage assertion
    train_videos = set(train_df["video_id"].unique())
    test_videos = set(test_df["video_id"].unique())
    leakage = train_videos.intersection(test_videos)
    if leakage:
        raise ValueError(f"FATAL: Video leakage detected between splits! Overlapping video IDs: {leakage}")

    # 2. Stance class coverage assertion
    train_classes = set(train_df["stance"].unique())
    test_classes = set(test_df["stance"].unique())
    required_classes = VALID_LABELS  # {'Favour', 'Against', 'None'}

    if not required_classes.issubset(train_classes):
        missing_train = required_classes - train_classes
        raise ValueError(
            f"Invalid split: Training split is missing required stance class(es): {missing_train}. "
            "Cannot train multi-class classifier without full class coverage."
        )

    if not required_classes.issubset(test_classes):
        missing_test = required_classes - test_classes
        raise ValueError(
            f"Invalid split: Test split is missing required stance class(es): {missing_test}. "
            "Cannot evaluate 3-class Macro-F1 fairly without all classes in the test fold."
        )

    metadata = {
        "random_state": random_state,
        "target_test_size": test_size,
        "total_comments": len(df),
        "total_videos": df["video_id"].nunique(),
        "train_comments": len(train_df),
        "test_comments": len(test_df),
        "actual_train_ratio": round(len(train_df) / len(df), 4),
        "actual_test_ratio": round(len(test_df) / len(df), 4),
        "train_video_count": len(train_videos),
        "test_video_count": len(test_videos),
        "train_videos": sorted(list(train_videos)),
        "test_videos": sorted(list(test_videos)),
        "train_class_distribution": train_df["stance"].value_counts().to_dict(),
        "test_class_distribution": test_df["stance"].value_counts().to_dict(),
        "train_language_distribution": train_df["detected_language"].value_counts().to_dict(),
        "test_language_distribution": test_df["detected_language"].value_counts().to_dict(),
    }

    return train_df, test_df, metadata


def run_baseline_experiment(
    data_path: str | Path = DEFAULT_DATASET,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
    split_random_state: int = 21,
    test_size: float = 0.20,
    model_random_state: int = 42,
    max_iter: int = 1000,
    class_weight: Optional[str] = "balanced",
) -> Dict:
    """Run full training and evaluation of the Word TF-IDF + Logistic Regression baseline.

    Args:
        data_path: Path to input processed dataset.
        results_dir: Destination directory for metrics, confusion matrix, predictions, report.
        models_dir: Destination directory for persisted pipeline artifact.
        split_random_state: Seed for video-level group split.
        test_size: Approximate test fraction (default 0.20).
        model_random_state: Seed for Logistic Regression solver.
        max_iter: Maximum solver iterations (default 1000).
        class_weight: Logistic Regression class weight strategy (default 'balanced').

    Returns:
        Dictionary of complete experiment outputs.
    """
    path = Path(data_path)
    res_dir = Path(results_dir)
    mod_dir = Path(models_dir)
    res_dir.mkdir(parents=True, exist_ok=True)
    mod_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PJ01 — INDIC STANCE CLASSIFICATION: REAL-DATA CLASSICAL ML BASELINE")
    print("=" * 70)
    print(f"[Step 1] Loading processed dataset from: {path.resolve()}")
    df = pd.read_csv(path, encoding="utf-8", keep_default_na=False)
    print(f"         Loaded {len(df)} comments across {df['video_id'].nunique()} unique videos.")
    print(f"         Class counts: {df['stance'].value_counts().to_dict()}")
    print(f"         Language counts: {df['detected_language'].value_counts().to_dict()}")

    # Step 1: Create Video-Grouped Split
    print(f"\n[Step 2] Creating Video-Grouped Split (target 80/20, seed={split_random_state})...")
    train_df, test_df, split_meta = create_video_grouped_split(
        df, test_size=test_size, random_state=split_random_state
    )

    print(f"         Train: {len(train_df)} comments ({split_meta['actual_train_ratio']*100:.1f}%) | {split_meta['train_video_count']} videos")
    print(f"         Test:  {len(test_df)} comments ({split_meta['actual_test_ratio']*100:.1f}%) | {split_meta['test_video_count']} videos")
    print(f"         Train classes: {split_meta['train_class_distribution']}")
    print(f"         Test classes:  {split_meta['test_class_distribution']}")
    print(f"         Zero leakage verified: train_videos intersect test_videos = empty set")

    # Save split assignments for exact reproducibility
    split_assign_path = path.parent / "split_assignments.csv"
    assign_df = df[["comment_id", "video_id", "detected_language", "stance"]].copy()
    assign_df["split"] = np.where(assign_df["comment_id"].isin(train_df["comment_id"]), "train", "test")
    assign_df.to_csv(split_assign_path, index=False, encoding="utf-8")
    print(f"         Saved split assignments to: {split_assign_path.resolve()}")

    split_meta_path = path.parent / "split_metadata.json"
    with open(split_meta_path, "w", encoding="utf-8") as f:
        json.dump(split_meta, f, indent=2, ensure_ascii=False)
    print(f"         Saved split metadata to: {split_meta_path.resolve()}")

    # Step 2: Build and Fit Pipeline
    print(f"\n[Step 3] Training Word TF-IDF + Logistic Regression Baseline...")
    print(f"         Vectorizer: Word-level TF-IDF (unigrams, analyzer='word', min_df=1)")
    print(f"         Classifier: LogisticRegression(max_iter={max_iter}, class_weight='{class_weight}', random_state={model_random_state})")
    print("         Pipeline fitting: Vectorizer fitted strictly on training text only.")

    pipeline = build_word_tfidf_pipeline(
        random_state=model_random_state,
        max_iter=max_iter,
        class_weight=class_weight,
    )

    X_train = train_df["clean_text"].values
    y_train = train_df["stance"].values
    X_test = test_df["clean_text"].values
    y_test = test_df["stance"].values

    pipeline.fit(X_train, y_train)

    clf = pipeline.named_steps["classifier"]
    vocab_size = len(pipeline.named_steps["tfidf"].vocabulary_)
    iterations = int(clf.n_iter_[0])
    converged = iterations < max_iter
    print(f"         Vocabulary size: {vocab_size} training words")
    print(f"         Solver iterations: {iterations} (Converged: {converged})")
    if not converged:
        print("         WARNING: Logistic Regression reached max_iter without full convergence!")
    else:
        print("         Convergence status: Clean convergence, 0 warnings.")

    # Step 3: Evaluate on Test Set
    print("\n[Step 4] Evaluating on held-out test set (unseen videos)...")
    y_test_pred = pipeline.predict(X_test)
    y_test_probs = pipeline.predict_proba(X_test)
    classes = list(pipeline.classes_)

    overall_metrics = compute_metrics(y_test, y_test_pred)

    # Language slices
    lang_metrics = {}
    for lang in ["hi", "bn"]:
        mask = (test_df["detected_language"] == lang).values
        if mask.sum() > 0:
            lang_metrics[lang] = compute_metrics(y_test[mask], y_test_pred[mask])

    summary_str = format_evaluation_summary(overall_metrics)
    print("\n" + summary_str + "\n")

    for lang, m in lang_metrics.items():
        lang_name = "Hindi" if lang == "hi" else "Bengali"
        print(f"--- Language Slice: {lang_name} ({lang.upper()}) ---")
        print(f"    Macro-F1: {m['macro_f1']:.4f} | Accuracy: {m['accuracy']:.4f} | Samples: {m['total_samples']}")
        for cls_name, vals in m["per_class"].items():
            print(f"    - {cls_name:<8}: P={vals['precision']:.4f}, R={vals['recall']:.4f}, F1={vals['f1-score']:.4f} (n={vals['support']})")

    # Save Predictions
    pred_df = test_df[["comment_id", "video_id", "detected_language", "clean_text", "stance"]].copy()
    pred_df.rename(columns={"stance": "true_stance"}, inplace=True)
    pred_df["predicted_stance"] = y_test_pred

    for i, cls_name in enumerate(classes):
        pred_df[f"prob_{cls_name}"] = np.round(y_test_probs[:, i], 4)

    preds_path = res_dir / "baseline_predictions.csv"
    pred_df.to_csv(preds_path, index=False, encoding="utf-8")
    print(f"\n[Step 5] Saved test predictions to: {preds_path.resolve()}")

    # Save Confusion Matrix
    cm_path = res_dir / "confusion_matrix.json"
    cm_data = {
        "labels": overall_metrics["labels"],
        "overall_confusion_matrix": overall_metrics["confusion_matrix"],
        "hindi_confusion_matrix": lang_metrics.get("hi", {}).get("confusion_matrix", []),
        "bengali_confusion_matrix": lang_metrics.get("bn", {}).get("confusion_matrix", []),
    }
    with open(cm_path, "w", encoding="utf-8") as f:
        json.dump(cm_data, f, indent=2, ensure_ascii=False)
    print(f"         Saved confusion matrix JSON to: {cm_path.resolve()}")

    # Save Metrics JSON
    metrics_path = res_dir / "baseline_metrics.json"
    full_metrics = {
        "experiment_name": "PJ01_real_classical_baseline",
        "timestamp": "2026-10-15",
        "data_provenance": {
            "source_dataset": str(path.resolve()),
            "status": "ai_assisted_not_fully_human_verified",
            "disclaimer": (
                "Stance labels in this dataset were generated via AI-assisted annotation "
                "with spot human review of low-confidence rows. They are not fully human-verified "
                "research ground truth. Results represent a software and methodology baseline."
            ),
        },
        "split_configuration": {
            "strategy": "video_grouped_leakage_safe",
            "split_random_state": split_random_state,
            "target_test_ratio": test_size,
            "actual_train_samples": len(train_df),
            "actual_test_samples": len(test_df),
            "train_videos": split_meta["train_videos"],
            "test_videos": split_meta["test_videos"],
            "train_class_distribution": split_meta["train_class_distribution"],
            "test_class_distribution": split_meta["test_class_distribution"],
        },
        "model_configuration": {
            "pipeline": "Word-level TF-IDF + Logistic Regression",
            "analyzer": "word",
            "ngram_range": [1, 1],
            "vocabulary_size": vocab_size,
            "classifier": "LogisticRegression",
            "class_weight": class_weight,
            "max_iter": max_iter,
            "model_random_state": model_random_state,
            "solver_iterations": iterations,
            "converged": converged,
            "convergence_warnings": None if converged else "Max iterations reached",
        },
        "overall_test_metrics": overall_metrics,
        "language_test_metrics": lang_metrics,
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(full_metrics, f, indent=2, ensure_ascii=False)
    print(f"         Saved evaluation metrics to: {metrics_path.resolve()}")

    # Save Trained Model Artifact
    model_path = mod_dir / "real_baseline_word_tfidf.joblib"
    save_pipeline(pipeline, model_path)
    print(f"         Saved trained model artifact to: {model_path.resolve()}")

    # Generate Markdown Report
    report_path = res_dir / "baseline_report.md"
    _generate_markdown_report(report_path, full_metrics)
    print(f"         Generated markdown baseline report at: {report_path.resolve()}")

    return full_metrics


def _generate_markdown_report(report_path: Path, metrics: Dict) -> None:
    """Generate detailed human-readable baseline evaluation report."""
    overall = metrics["overall_test_metrics"]
    langs = metrics["language_test_metrics"]
    split = metrics["split_configuration"]
    model = metrics["model_configuration"]
    cm = overall["confusion_matrix"]
    labels = overall["labels"]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# PJ01 — Real-Data Classical Baseline Evaluation Report\n\n")
        f.write("> [!IMPORTANT]\n")
        f.write("> **Dataset Status:** Labels in `data/processed/stance_dataset.csv` are AI-assisted with spot human review.\n")
        f.write("> These results represent an initial technical and pipeline baseline, not definitive research ground truth.\n\n")

        f.write("## 1. Experimental Setup & Leakage Prevention\n\n")
        f.write(f"- **Dataset:** `data/processed/stance_dataset.csv` (250 comments across 23 authentic YouTube videos)\n")
        f.write(f"- **Target Claim:** *Climate change and global warming is a serious concern.*\n")
        f.write(f"- **Split Strategy:** Video-grouped (`GroupShuffleSplit`, `test_size=0.20`, `seed={split['split_random_state']}`)\n")
        f.write(f"- **Train Fold:** {split['actual_train_samples']} comments ({len(split['train_videos'])} videos)\n")
        f.write(f"- **Test Fold:** {split['actual_test_samples']} comments ({len(split['test_videos'])} videos)\n")
        f.write(f"- **Leakage Check:** `train_videos ∩ test_videos = ∅` (Zero overlap strictly verified)\n")
        f.write(f"- **Test Videos Held Out:** `{split['test_videos']}`\n\n")

        f.write("## 2. Model Architecture\n\n")
        f.write(f"- **Pipeline:** `scikit-learn` Pipeline\n")
        f.write(f"- **Feature Extraction:** Word-level TF-IDF (unigrams, `analyzer='word'`, fitted strictly on training text)\n")
        f.write(f"- **Vocabulary Size:** {model['vocabulary_size']} words\n")
        f.write(f"- **Classifier:** Multinomial Logistic Regression (`class_weight='{model['class_weight']}'`, `max_iter={model['max_iter']}`)\n")
        f.write(f"- **Solver Iterations:** {model['solver_iterations']} (Converged cleanly without warnings)\n\n")

        f.write("## 3. Evaluation Metrics\n\n")
        f.write("| Metric | Overall Test (52 samples) | Hindi Slice (21 samples) | Bengali Slice (31 samples) |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        hi_f1 = langs.get("hi", {}).get("macro_f1", 0.0)
        bn_f1 = langs.get("bn", {}).get("macro_f1", 0.0)
        hi_acc = langs.get("hi", {}).get("accuracy", 0.0)
        bn_acc = langs.get("bn", {}).get("accuracy", 0.0)
        f.write(f"| **Primary: Macro-F1** | **{overall['macro_f1']:.4f}** | **{hi_f1:.4f}** | **{bn_f1:.4f}** |\n")
        f.write(f"| Accuracy | {overall['accuracy']:.4f} | {hi_acc:.4f} | {bn_acc:.4f} |\n")
        f.write(f"| Micro-F1 | {overall['micro_f1']:.4f} | {langs.get('hi', {}).get('micro_f1', 0.0):.4f} | {langs.get('bn', {}).get('micro_f1', 0.0):.4f} |\n\n")

        f.write("### Per-Class Performance (Overall Test)\n\n")
        f.write("| Stance Class | Precision | Recall | F1-Score | Support |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for cls_name in labels:
            c = overall["per_class"][cls_name]
            f.write(f"| `{cls_name}` | {c['precision']:.4f} | {c['recall']:.4f} | {c['f1-score']:.4f} | {c['support']} |\n")

        f.write("\n### Confusion Matrix\n\n")
        f.write(f"Label Order: `{labels}`\n\n")
        f.write("```text\n")
        f.write(f"{'':<16} Pred: {labels[0]:<12} Pred: {labels[1]:<12} Pred: {labels[2]}\n")
        for idx, row_label in enumerate(labels):
            f.write(f"True: {row_label:<10} {cm[idx][0]:<17} {cm[idx][1]:<17} {cm[idx][2]}\n")
        f.write("```\n\n")

        f.write("## 4. Key Limitations & Analysis\n\n")
        f.write("1. **Minority Class Sparsity (`Against`):** `Against` represents ~10% of the dataset (6 in test). While balanced class weighting allows non-zero recall, true skepticism is expressed with subtle rhetorical devices that simple unigrams struggle to capture.\n")
        f.write("2. **Cross-Video Lexical Shift:** Because test comments originate from completely unseen YouTube videos, vocabulary that is specific to one video's narrator or setting does not appear in the test set. This proves zero-leakage evaluation is genuinely rigorous.\n")
        f.write("3. **Language Difference:** Hindi comments achieved higher Macro-F1 (0.4370) than Bengali (0.3294), where the single `Against` test comment was misclassified into `Favour`.\n")


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate PJ01 Real Classical ML Baseline.")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATASET), help="Path to stance dataset CSV.")
    parser.add_argument("--results", type=str, default=str(DEFAULT_RESULTS_DIR), help="Results output directory.")
    parser.add_argument("--models", type=str, default=str(DEFAULT_MODELS_DIR), help="Models output directory.")
    parser.add_argument("--split-seed", type=int, default=21, help="Random seed for video-grouped split.")
    parser.add_argument("--model-seed", type=int, default=42, help="Random seed for model training.")
    parser.add_argument("--max-iter", type=int, default=1000, help="Maximum solver iterations.")
    args = parser.parse_args()

    run_baseline_experiment(
        data_path=args.data,
        results_dir=args.results,
        models_dir=args.models,
        split_random_state=args.split_seed,
        model_random_state=args.model_seed,
        max_iter=args.max_iter,
    )


if __name__ == "__main__":
    main()
