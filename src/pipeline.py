import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
import joblib


def build_stance_pipeline(
    random_state: int = 42,
    c_param: float = 1.0,
    max_iter: int = 1000,
) -> Pipeline:
    """Construct an end-to-end scikit-learn Pipeline.

    Architecture:
    1. Feature Union:
       - Word-level TF-IDF (unigrams & bigrams): captures domain vocabulary & key phrases.
       - Character n-gram TF-IDF (char_wb, ngrams 3 to 5): captures subword roots and
         inflectional morphology across Indic scripts (Hindi Devanagari and Bengali).
    2. Classifier:
       - Multinomial Logistic Regression with balanced class weights.

    The pipeline ensures feature extraction and vectorizer vocabulary are strictly
    fitted only on training splits, preventing validation data leakage.

    Args:
        random_state: Seed for reproducible solver optimization.
        c_param: Inverse regularization parameter C for Logistic Regression.
        max_iter: Maximum solver iterations.

    Returns:
        Unfitted scikit-learn Pipeline.
    """
    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )

    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        sublinear_tf=True,
    )

    features = FeatureUnion(
        transformer_list=[
            ("word_tfidf", word_vectorizer),
            ("char_tfidf", char_vectorizer),
        ]
    )

    classifier = LogisticRegression(
        C=c_param,
        max_iter=max_iter,
        random_state=random_state,
        class_weight="balanced",
    )

    pipeline = Pipeline(
        steps=[
            ("features", features),
            ("classifier", classifier),
        ]
    )

    return pipeline


def save_pipeline(pipeline: Pipeline, filepath: str | Path) -> Path:
    """Save fitted pipeline to disk using joblib.

    Args:
        pipeline: Fitted scikit-learn Pipeline.
        filepath: Target destination path.

    Returns:
        Resolved Path to saved artifact.
    """
    dest = Path(filepath)
    dest.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, dest)
    return dest.resolve()


def load_pipeline(filepath: str | Path) -> Pipeline:
    """Load fitted pipeline from disk using joblib.

    Args:
        filepath: Source artifact path.

    Returns:
        Loaded scikit-learn Pipeline.
    """
    source = Path(filepath)
    if not source.is_file():
        raise FileNotFoundError(f"Pipeline artifact not found at: {source.resolve()}")
    return joblib.load(source)
