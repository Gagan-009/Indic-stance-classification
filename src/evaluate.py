"""Evaluation utilities for Indic Stance Classification."""

from typing import Dict, List, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

LABEL_ORDER = ["Favour", "Against", "None"]


def compute_metrics(
    y_true: List[str] | np.ndarray,
    y_pred: List[str] | np.ndarray,
    labels: Optional[List[str]] = None,
) -> Dict:
    """Compute comprehensive evaluation metrics for stance classification.

    Macro-F1 is the primary evaluation metric for this task, treating all three
    classes (Favour, Against, None) with equal importance regardless of class size.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        labels: Label order for reports and confusion matrix. Defaults to LABEL_ORDER.

    Returns:
        Dictionary containing Macro-F1, Accuracy, per-class metrics, and confusion matrix.
    """
    if labels is None:
        labels = LABEL_ORDER

    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    micro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="micro", zero_division=0))
    accuracy = float(accuracy_score(y_true, y_pred))

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    per_class = {}
    for label in labels:
        if label in report_dict:
            per_class[label] = {
                "precision": round(float(report_dict[label]["precision"]), 4),
                "recall": round(float(report_dict[label]["recall"]), 4),
                "f1-score": round(float(report_dict[label]["f1-score"]), 4),
                "support": int(report_dict[label]["support"]),
            }

    return {
        "macro_f1": round(macro_f1, 4),
        "micro_f1": round(micro_f1, 4),
        "accuracy": round(accuracy, 4),
        "labels": labels,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "total_samples": len(y_true),
    }


def format_evaluation_summary(metrics: Dict) -> str:
    """Format metrics dictionary into a readable terminal/markdown report."""
    lines = []
    lines.append("=" * 60)
    lines.append("INDIC STANCE CLASSIFICATION - EVALUATION SUMMARY")
    lines.append("=" * 60)
    lines.append(f"Primary Metric (Macro-F1) : {metrics['macro_f1']:.4f}")
    lines.append(f"Accuracy                  : {metrics['accuracy']:.4f}")
    lines.append(f"Micro-F1                  : {metrics['micro_f1']:.4f}")
    lines.append(f"Total Evaluated Samples   : {metrics['total_samples']}")
    lines.append("-" * 60)
    lines.append(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<8}")
    lines.append("-" * 60)
    for label in metrics["labels"]:
        cls_info = metrics["per_class"].get(label, {})
        p = cls_info.get("precision", 0.0)
        r = cls_info.get("recall", 0.0)
        f1 = cls_info.get("f1-score", 0.0)
        sup = cls_info.get("support", 0)
        lines.append(f"{label:<12} {p:<12.4f} {r:<12.4f} {f1:<12.4f} {sup:<8}")
    lines.append("-" * 60)
    lines.append("Confusion Matrix (rows: true, cols: pred):")
    lines.append(f"Labels: {metrics['labels']}")
    for row in metrics["confusion_matrix"]:
        lines.append(f"  {row}")
    lines.append("=" * 60)
    return "\n".join(lines)
