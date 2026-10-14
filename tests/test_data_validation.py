"""Unit tests for dataset loading and validation constraints."""

import sys
import tempfile
import unittest
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import VALID_LABELS, load_and_validate_dataset


class TestDataValidation(unittest.TestCase):
    """Test suite for dataset validation rules."""

    def setUp(self):
        self.real_dataset_path = PROJECT_ROOT / "data" / "illustrative_dataset.csv"

    def test_real_dataset_integrity(self):
        """Verify the actual illustrative dataset passes all strict integrity checks."""
        self.assertTrue(self.real_dataset_path.is_file(), "illustrative_dataset.csv must exist")
        df, report = load_and_validate_dataset(self.real_dataset_path)

        self.assertTrue(report["passed"])
        self.assertEqual(report["empty_text_count"], 0)
        self.assertEqual(report["invalid_label_count"], 0)
        self.assertEqual(report["duplicate_id_count"], 0)
        self.assertGreater(len(df), 0)

        # Ensure all labels are valid
        self.assertTrue(set(df["stance"].unique()).issubset(VALID_LABELS))

        # Ensure 'None' label was not coerced into NaN
        self.assertIn("None", df["stance"].unique())
        self.assertFalse(df["stance"].isna().any())

    def test_invalid_label_raises_error(self):
        """Ensure invalid stance labels are caught and raise ValueError."""
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as tmp:
            tmp.write("id,language,text,stance,provenance\n")
            tmp.write("1,hi,ग्लोबल वॉर्मिंग सच है,InvalidLabel,test\n")
            tmp.write("2,hi,यह गलत है,Against,test\n")
            tmp.write("3,hi,कुछ भी,None,test\n")
            tmp.write("4,hi,हाँ सच है,Favour,test\n")
            tmp_path = Path(tmp.name)

        try:
            with self.assertRaises(ValueError) as ctx:
                load_and_validate_dataset(tmp_path, min_samples_per_class=1)
            self.assertIn("invalid label", str(ctx.exception).lower())
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_empty_text_raises_error(self):
        """Ensure empty or blank text fields trigger a validation error."""
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as tmp:
            tmp.write("id,language,text,stance,provenance\n")
            tmp.write("1,hi,\"   \",Favour,test\n")
            tmp.write("2,hi,यह गलत है,Against,test\n")
            tmp.write("3,hi,कुछ भी,None,test\n")
            tmp_path = Path(tmp.name)

        try:
            with self.assertRaises(ValueError) as ctx:
                load_and_validate_dataset(tmp_path, min_samples_per_class=1)
            self.assertIn("empty text", str(ctx.exception).lower())
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_duplicate_id_raises_error(self):
        """Ensure duplicate record IDs trigger a validation error."""
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as tmp:
            tmp.write("id,language,text,stance,provenance\n")
            tmp.write("ID1,hi,ग्लोबल वॉर्मिंग सच है,Favour,test\n")
            tmp.write("ID1,hi,यह गलत है,Against,test\n")
            tmp.write("ID2,hi,कुछ भी,None,test\n")
            tmp_path = Path(tmp.name)

        try:
            with self.assertRaises(ValueError) as ctx:
                load_and_validate_dataset(tmp_path, min_samples_per_class=1)
            self.assertIn("duplicate id", str(ctx.exception).lower())
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_insufficient_class_samples_raises_error(self):
        """Ensure classes below minimum threshold raise ValueError."""
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as tmp:
            tmp.write("id,language,text,stance,provenance\n")
            tmp.write("1,hi,ग्लोबल वॉर्मिंग सच है,Favour,test\n")
            tmp.write("2,hi,यह गलत है,Against,test\n")
            tmp.write("3,hi,कुछ भी,None,test\n")
            tmp_path = Path(tmp.name)

        try:
            with self.assertRaises(ValueError) as ctx:
                load_and_validate_dataset(tmp_path, min_samples_per_class=5)
            self.assertIn("minimum required", str(ctx.exception).lower())
        finally:
            tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
