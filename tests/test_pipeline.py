"""Unit tests for ML pipeline architecture, training, and evaluation."""

import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate import compute_metrics
from src.pipeline import build_stance_pipeline, load_pipeline, save_pipeline


class TestPipeline(unittest.TestCase):
    """Test suite for pipeline creation, prediction, metrics, and persistence."""

    def setUp(self):
        self.sample_train_texts = [
            "ग्लोबल वॉर्मिंग सचमुच एक बहुत बड़ा संकट है",
            "पर्यावरण बचाना हमारा कर्तव्य है",
            "जलवायु परिवर्तन पश्चिमी देशों का प्रोपेगैंडा है",
            "मौसम में बदलाव सामान्य है कोई खतरा नहीं",
            "यह वीडियो बहुत अच्छा है धन्यवाद",
            "कृपया अगला वीडियो जल्द अपलोड करें",
            "জলবায়ু পরিবর্তন একটি মারাত্মক সমস্যা",
            "গাছ কাটা বন্ধ না করলে ভবিষ্যৎ বিপন্ন",
            "গ্লোবাল ওয়ার্মিং একটি ভুয়া প্রচার",
            "ঋতু পরিবর্তন স্বাভাবিক ঘটনা ভয়ের কিছু নেই",
            "এই ভিডিওটির উপস্থাপনা খুব সুন্দর",
            "চ্যানেলটি সাবস্ক্রাইব করে রাখলাম",
        ]
        self.sample_train_labels = [
            "Favour", "Favour",
            "Against", "Against",
            "None", "None",
            "Favour", "Favour",
            "Against", "Against",
            "None", "None",
        ]

    def test_pipeline_construction_and_components(self):
        """Verify pipeline contains FeatureUnion and LogisticRegression."""
        pipe = build_stance_pipeline()
        self.assertIn("features", pipe.named_steps)
        self.assertIn("classifier", pipe.named_steps)
        transformer_keys = [name for name, _ in pipe.named_steps["features"].transformer_list]
        self.assertIn("word_tfidf", transformer_keys)
        self.assertIn("char_tfidf", transformer_keys)

    def test_fit_and_predict(self):
        """Verify pipeline fits on Indic text and returns valid predictions."""
        pipe = build_stance_pipeline()
        pipe.fit(self.sample_train_texts, self.sample_train_labels)

        test_inputs = [
            "ग्लोबल वॉर्मिंग से खतरा बढ़ रहा है",
            "জলবায়ু সংকট একটি অসত্য প্রচার",
            "ভিডিওতে ভালো তথ্য দেওয়া হয়েছে",
        ]
        preds = pipe.predict(test_inputs)
        self.assertEqual(len(preds), 3)
        for p in preds:
            self.assertIn(p, ["Favour", "Against", "None"])

        probs = pipe.predict_proba(test_inputs)
        self.assertEqual(probs.shape, (3, 3))
        # Probabilities should sum to 1.0 per row
        for row in probs:
            self.assertAlmostEqual(float(np.sum(row)), 1.0, places=5)

    def test_save_and_load_roundtrip(self):
        """Verify fitted model can be saved and reloaded with identical predictions."""
        pipe = build_stance_pipeline()
        pipe.fit(self.sample_train_texts, self.sample_train_labels)
        original_preds = pipe.predict(self.sample_train_texts)

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            save_pipeline(pipe, tmp_path)
            loaded_pipe = load_pipeline(tmp_path)
            reloaded_preds = loaded_pipe.predict(self.sample_train_texts)
            np.testing.assert_array_equal(original_preds, reloaded_preds)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_compute_metrics_macro_f1(self):
        """Verify Macro-F1 computation behaves accurately."""
        y_true = ["Favour", "Against", "None", "Favour"]
        y_pred = ["Favour", "Against", "None", "Against"]

        metrics = compute_metrics(y_true, y_pred)
        self.assertIn("macro_f1", metrics)
        self.assertIn("accuracy", metrics)
        self.assertIn("per_class", metrics)
        self.assertIn("confusion_matrix", metrics)
        self.assertEqual(metrics["total_samples"], 4)
        self.assertEqual(metrics["accuracy"], 0.75)


if __name__ == "__main__":
    unittest.main()
