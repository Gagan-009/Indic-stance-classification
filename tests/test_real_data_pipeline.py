"""Unit tests for the real-data pipeline, schema integrity, and split leakage prevention."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import VALID_LABELS, VALID_LANGUAGES
from src.pipeline import build_stance_pipeline
from src.prepare_dataset import (
    ANNOTATION_COLUMNS,
    CANDIDATE_COLUMNS,
    clean_and_prepare_dataset,
    detect_indic_language,
    export_indic_annotation_pool,
    sample_annotation_candidates,
    categorize_other_script_comment,
)


class TestRealDataPipeline(unittest.TestCase):
    """Test suite for real YouTube stance data processing and leakage prevention."""

    def setUp(self):
        self.raw_path = PROJECT_ROOT / "data" / "raw" / "real_youtube_comments_pilot.csv"
        self.processed_path = PROJECT_ROOT / "data" / "processed" / "cleaned_stance_dataset.csv"
        self.metrics_path = PROJECT_ROOT / "models" / "real_baseline_metrics.json"
        self.harvested_path = PROJECT_ROOT / "data" / "raw" / "harvested_youtube_comments.csv"
        self.to_annotate_path = PROJECT_ROOT / "data" / "raw" / "to_annotate_indic_comments.csv"
        self.candidate_250_path = PROJECT_ROOT / "data" / "raw" / "annotation_candidate_250.csv"
        self.stance_dataset_path = PROJECT_ROOT / "data" / "processed" / "stance_dataset.csv"

    def test_raw_and_processed_files_exist(self):
        """Ensure both raw pilot data and processed real dataset files exist."""
        self.assertTrue(self.raw_path.is_file(), "Raw pilot dataset must exist in data/raw/")
        self.assertTrue(self.processed_path.is_file(), "Processed dataset must exist in data/processed/")

    def test_processed_dataset_schema_and_labels(self):
        """Verify cleaned dataset contains valid columns, labels, languages, and zero NaNs."""
        df = pd.read_csv(self.processed_path, encoding="utf-8", keep_default_na=False)

        required_cols = ["id", "video_id", "text", "clean_text", "stance", "language", "provenance"]
        for col in required_cols:
            self.assertIn(col, df.columns, f"Column '{col}' must be present in processed data.")

        # Check stance labels
        unique_labels = set(df["stance"].unique())
        self.assertTrue(unique_labels.issubset(VALID_LABELS), f"Invalid labels found: {unique_labels}")
        self.assertIn("None", unique_labels, "'None' class must be present")
        self.assertFalse(df["stance"].isna().any(), "Stance column must contain zero NaNs")

        # Check languages
        unique_langs = set(df["language"].unique())
        self.assertTrue(unique_langs.issubset(VALID_LANGUAGES), f"Invalid languages found: {unique_langs}")

        # Check no blank text
        self.assertFalse((df["clean_text"].str.strip() == "").any(), "Clean text must not have empty strings")

        # Check provenance is explicitly real YouTube manual annotation
        self.assertEqual(list(df["provenance"].unique()), ["manual_annotation_real_youtube"])

    def test_language_identification_logic(self):
        """Verify Unicode script counting correctly differentiates Hindi and Bengali without transliteration."""
        hindi_text = "ग्लोबल वॉर्मिंग सचमुच एक गंभीर संकट है।"
        bengali_text = "জলবায়ু পরিবর্তন সত্যি একটি মারাত্মক সমস্যা।"

        self.assertEqual(detect_indic_language(hindi_text), "hi")
        self.assertEqual(detect_indic_language(bengali_text), "bn")

    def test_duplicate_removal_in_preparation_pipeline(self):
        """Verify pipeline drops exact duplicate rows while preserving distinct comments."""
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as raw_tmp:
            raw_tmp.write("id,video_id,channel,video_title,text,language,stance,provenance,target_claim\n")
            raw_tmp.write("R1,V1,C1,T1,ग्लोबल वॉर्मिंग बहुत गंभीर समस्या है,hi,Favour,manual,claim\n")
            # Duplicate ID and text
            raw_tmp.write("R1,V1,C1,T1,ग्लोबल वॉर्मिंग बहुत गंभीर समस्या है,hi,Favour,manual,claim\n")
            # Distinct comment
            raw_tmp.write("R2,V1,C1,T1,मौसम में बदलाव सामान्य बात है,hi,Against,manual,claim\n")
            raw_tmp_path = Path(raw_tmp.name)

        out_tmp_path = raw_tmp_path.parent / "out_clean_test.csv"
        rep_tmp_path = raw_tmp_path.parent / "rep_test.md"

        try:
            cleaned_df, report = clean_and_prepare_dataset(
                raw_path=raw_tmp_path,
                processed_path=out_tmp_path,
                report_path=rep_tmp_path,
            )
            self.assertEqual(report["initial_rows"], 3)
            self.assertEqual(report["final_rows"], 2)
            self.assertEqual(report["duplicate_rows_dropped"], 1)
            self.assertEqual(len(cleaned_df), 2)
        finally:
            raw_tmp_path.unlink(missing_ok=True)
            out_tmp_path.unlink(missing_ok=True)
            rep_tmp_path.unlink(missing_ok=True)

    def test_split_leakage_prevention_across_videos(self):
        """Verify that video-grouped split strictly isolates test video IDs from train video IDs."""
        df = pd.read_csv(self.processed_path, encoding="utf-8", keep_default_na=False)
        test_video_ids = ["NDTV_HI_ENV03", "PROTHOM_BN_ENV03"]

        test_df = df[df["video_id"].isin(test_video_ids)]
        train_df = df[~df["video_id"].isin(test_video_ids)]

        train_vids = set(train_df["video_id"].unique())
        test_vids = set(test_df["video_id"].unique())

        # CRITICAL LEAKAGE ASSERTION: Intersection must be completely empty
        leakage = train_vids.intersection(test_vids)
        self.assertEqual(len(leakage), 0, f"Video leakage detected! Overlapping video IDs: {leakage}")
        self.assertTrue(train_vids.isdisjoint(test_vids), "Train and test video sets must be disjoint")
        self.assertGreater(len(train_df), 0, "Train split must contain data")
        self.assertGreater(len(test_df), 0, "Test split must contain data")

    def test_vectorizer_leakage_prevention(self):
        """Verify that TF-IDF vectorizers are fit strictly on training fold only."""
        train_texts = [
            "ग्लोबल वॉर्मिंग सच है",
            "पर्यावरण बचाओ",
            "জলবায়ু পরিবর্তন একটি অসত্য প্রচার",
        ]
        train_labels = ["Favour", "Favour", "Against"]

        # An unseen test word that should NOT appear in fitted feature vocabulary
        test_texts = ["अनदेखा शब्द बिल्कुल नया"]

        pipe = build_stance_pipeline()
        pipe.fit(train_texts, train_labels)

        word_vectorizer = pipe.named_steps["features"].transformer_list[0][1]
        feature_vocab = word_vectorizer.vocabulary_

        # Verify unseen test word is not in the vocabulary
        self.assertNotIn("अनदेखा", feature_vocab, "Unseen test token must not leak into fitted vectorizer")
        self.assertIn("सच", feature_vocab, "Training token must be present in fitted vectorizer")

    def test_real_baseline_metrics_file_integrity(self):
        """Verify that saved real baseline metrics match computed outputs."""
        self.assertTrue(self.metrics_path.is_file(), "Baseline metrics JSON must exist in models/")
        with open(self.metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("overall_test_metrics", data)
        self.assertIn("language_test_metrics", data)
        self.assertIn("metadata", data)
        self.assertFalse(data["metadata"]["is_synthetic_demonstration_data"])
        self.assertEqual(data["metadata"]["data_provenance"], "manual_annotation_real_youtube")

        overall = data["overall_test_metrics"]
        self.assertGreater(overall["macro_f1"], 0.0)
        self.assertGreater(overall["accuracy"], 0.0)
        self.assertEqual(overall["total_samples"], 24)

    def test_indic_annotation_pool_schema_and_text_retention(self):
        """Verify to_annotate_indic_comments.csv preserves raw_text, clean_text, provenance, and empty stance."""
        self.assertTrue(self.to_annotate_path.is_file(), "Annotation pool file must exist.")
        df = pd.read_csv(self.to_annotate_path, encoding="utf-8", keep_default_na=False)

        # 1. Column presence
        for col in ANNOTATION_COLUMNS:
            self.assertIn(col, df.columns, f"Annotation column '{col}' missing.")

        # 2. Text retention
        self.assertFalse((df["raw_text"].astype(str).str.strip() == "").any(), "raw_text must not contain empty strings.")
        self.assertFalse((df["clean_text"].astype(str).str.strip() == "").any(), "clean_text must not contain empty strings.")

        # 3. Stance is strictly empty string (no automated labelling)
        self.assertTrue((df["stance"] == "").all(), "All stance entries must be empty prior to human annotation.")

        # 4. Languages are strictly Indic script (hi, bn)
        unique_langs = set(df["detected_language"].unique())
        self.assertTrue(unique_langs.issubset({"hi", "bn"}), f"Invalid languages in Indic pool: {unique_langs}")

        # 5. Length verification (all 482 harvested Indic comments preserved without deletion)
        self.assertEqual(len(df), 482, "All harvested Devanagari/Bengali comments must be retained.")

    def test_annotation_candidate_250_balance_and_reproducibility(self):
        """Verify annotation_candidate_250.csv has exact 125/125 balance, video diversity, and reproducibility."""
        self.assertTrue(self.candidate_250_path.is_file(), "Candidate 250 file must exist.")
        df = pd.read_csv(self.candidate_250_path, encoding="utf-8", keep_default_na=False)

        # 1. Total count, schema and balance
        self.assertEqual(len(df), 250, "Candidate sample must contain exactly 250 comments.")
        for col in CANDIDATE_COLUMNS:
            self.assertIn(col, df.columns, f"Candidate column '{col}' missing.")
        lang_counts = df["detected_language"].value_counts().to_dict()
        self.assertEqual(lang_counts.get("hi"), 125, "Hindi candidate sample must be exactly 125.")
        self.assertEqual(lang_counts.get("bn"), 125, "Bengali candidate sample must be exactly 125.")

        # 2. Multi-video diversity (sampling distributed across multiple videos)
        hi_videos = df[df["detected_language"] == "hi"]["video_id"].nunique()
        bn_videos = df[df["detected_language"] == "bn"]["video_id"].nunique()
        self.assertGreaterEqual(hi_videos, 10, f"Expected at least 10 Hindi videos, got {hi_videos}")
        self.assertGreaterEqual(bn_videos, 8, f"Expected at least 8 Bengali videos, got {bn_videos}")

        # 3. Text integrity and unlabelled stance / notes
        self.assertFalse((df["raw_text"].astype(str).str.strip() == "").any(), "raw_text must not be empty.")
        self.assertTrue((df["stance"] == "").all(), "Candidate stance must be unassigned.")
        self.assertIn("annotation_notes", df.columns, "annotation_notes column must exist.")
        self.assertTrue((df["annotation_notes"] == "").all(), "Candidate annotation_notes must be blank.")

        # 4. Reproducibility test
        df_reproduced = sample_annotation_candidates(
            pool_input=self.to_annotate_path,
            output_path=self.candidate_250_path.parent / "_test_reproduced_250.csv",
            target_per_lang=125,
            random_state=42,
        )
        try:
            self.assertEqual(df["comment_id"].tolist(), df_reproduced["comment_id"].tolist())
        finally:
            (self.candidate_250_path.parent / "_test_reproduced_250.csv").unlink(missing_ok=True)

    def test_other_script_categorization(self):
        """Verify that Romanized Hinglish/Banglish and other Latin comments are recognized and separated."""
        self.assertEqual(categorize_other_script_comment("kya bol rahe ho bhai yeh sab galat hai"), "romanized_hinglish")
        self.assertEqual(categorize_other_script_comment("amra shobai bhalo bhabe jani"), "romanized_banglish")
        self.assertEqual(categorize_other_script_comment("This is an important climate documentary."), "english_or_other_latin")
        self.assertEqual(categorize_other_script_comment("https://youtu.be/xyz"), "noise_or_url")
    def test_processed_stance_dataset_integrity(self):
        """Verify data/processed/stance_dataset.csv integrity, row counts, labels, and provenance."""
        self.assertTrue(self.stance_dataset_path.is_file(), "data/processed/stance_dataset.csv must exist.")
        df = pd.read_csv(self.stance_dataset_path, encoding="utf-8", keep_default_na=False)

        # 1. Total row count and uniqueness
        self.assertEqual(len(df), 250, "Processed stance dataset must have exactly 250 rows.")
        self.assertEqual(df["comment_id"].nunique(), 250, "Comment IDs must be unique.")
        self.assertEqual(df["video_id"].nunique(), 23, "All 23 source videos must be represented.")

        # 2. Required columns
        expected_cols = [
            "id", "comment_id", "video_id", "author_display_name", "published_at",
            "like_count", "language", "detected_language", "raw_text", "text",
            "clean_text", "stance", "annotation_notes", "provenance", "target_claim"
        ]
        for col in expected_cols:
            self.assertIn(col, df.columns, f"Expected column '{col}' missing from stance_dataset.csv.")

        # 3. Valid stance labels and no NaNs / empty strings
        unique_labels = set(df["stance"].unique())
        self.assertTrue(unique_labels.issubset(VALID_LABELS), f"Invalid labels found: {unique_labels}")
        self.assertFalse((df["stance"] == "").any(), "No stance entry may be empty.")

        # 4. Human corrections verified
        row1 = df[df["comment_id"] == "UgzdgoZDeOu_mjn0s4p4AaABAg"].iloc[0]
        self.assertEqual(row1["stance"], "None", "Human correction for UgzdgoZDeOu_mjn0s4p4AaABAg failed.")
        row2 = df[df["comment_id"] == "UgzmJl_8l4PL07JFRs54AaABAg"].iloc[0]
        self.assertEqual(row2["stance"], "None", "Human correction for UgzmJl_8l4PL07JFRs54AaABAg failed.")

        # 5. Language balance
        lang_counts = df["detected_language"].value_counts().to_dict()
        self.assertEqual(lang_counts.get("hi"), 125, "Hindi count must be 125.")
        self.assertEqual(lang_counts.get("bn"), 125, "Bengali count must be 125.")

        # 6. Provenance tag
        self.assertTrue((df["provenance"] == "ai_assisted_not_fully_human_verified").all(), "Provenance must reflect AI-assisted status.")

    def test_video_grouped_split_leakage_and_class_coverage(self):
        """Verify video-grouped split prevents leakage and provides 3-class coverage in both folds."""
        from src.train_real_baseline import create_video_grouped_split
        df = pd.read_csv(self.stance_dataset_path, encoding="utf-8", keep_default_na=False)

        train_df, test_df, meta = create_video_grouped_split(df, test_size=0.20, random_state=21)

        # 1. Zero leakage
        train_vids = set(train_df["video_id"].unique())
        test_vids = set(test_df["video_id"].unique())
        self.assertEqual(len(train_vids.intersection(test_vids)), 0, "Video IDs must not overlap between train and test.")

        # 2. Split sizes
        self.assertEqual(len(train_df), 198)
        self.assertEqual(len(test_df), 52)
        self.assertEqual(len(train_df) + len(test_df), 250)

        # 3. 3-Class coverage in both splits
        self.assertEqual(set(train_df["stance"].unique()), VALID_LABELS)
        self.assertEqual(set(test_df["stance"].unique()), VALID_LABELS)

        # 4. Both languages in test
        self.assertIn("hi", test_df["detected_language"].values)
        self.assertIn("bn", test_df["detected_language"].values)

    def test_word_tfidf_pipeline_structure(self):
        """Verify word-level TF-IDF pipeline configuration and step names."""
        from src.pipeline import build_word_tfidf_pipeline
        pipe = build_word_tfidf_pipeline(random_state=42, max_iter=500)
        self.assertIn("tfidf", pipe.named_steps)
        self.assertIn("classifier", pipe.named_steps)
        self.assertEqual(pipe.named_steps["tfidf"].analyzer, "word")
        self.assertEqual(pipe.named_steps["tfidf"].ngram_range, (1, 1))
        self.assertEqual(pipe.named_steps["classifier"].class_weight, "balanced")

    def test_baseline_results_artifacts_exist(self):
        """Verify all baseline output files exist under results/ and data/processed/."""
        results_dir = PROJECT_ROOT / "results"
        metrics_file = results_dir / "baseline_metrics.json"
        preds_file = results_dir / "baseline_predictions.csv"
        cm_file = results_dir / "confusion_matrix.json"
        report_file = results_dir / "baseline_report.md"
        assign_file = PROJECT_ROOT / "data" / "processed" / "split_assignments.csv"

        self.assertTrue(metrics_file.is_file(), "results/baseline_metrics.json must exist")
        self.assertTrue(preds_file.is_file(), "results/baseline_predictions.csv must exist")
        self.assertTrue(cm_file.is_file(), "results/confusion_matrix.json must exist")
        self.assertTrue(report_file.is_file(), "results/baseline_report.md must exist")
        self.assertTrue(assign_file.is_file(), "data/processed/split_assignments.csv must exist")

        # Verify predictions schema and row count
        preds_df = pd.read_csv(preds_file, encoding="utf-8", keep_default_na=False)
        self.assertEqual(len(preds_df), 52, "Predictions file must contain exactly 52 held-out test predictions")
        for col in ["comment_id", "video_id", "detected_language", "clean_text", "true_stance", "predicted_stance"]:
            self.assertIn(col, preds_df.columns, f"Expected column '{col}' missing from predictions CSV")

        # Verify metrics contents
        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
        self.assertIn("overall_test_metrics", metrics_data)
        self.assertIn("macro_f1", metrics_data["overall_test_metrics"])
        self.assertIn("accuracy", metrics_data["overall_test_metrics"])
        self.assertIn("language_test_metrics", metrics_data)
        self.assertIn("hi", metrics_data["language_test_metrics"])
        self.assertIn("bn", metrics_data["language_test_metrics"])


if __name__ == "__main__":
    unittest.main()
