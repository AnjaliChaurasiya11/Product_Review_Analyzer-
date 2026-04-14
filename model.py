import os
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
import joblib

from preprocessing import clean_text

MODEL_DIR = "models"
DEFAULT_CONFIDENCE_THRESHOLD = 0.45


class SentimentModel:
    """Manages training, loading, and inference for Logistic Regression & Naive Bayes."""

    def __init__(self, confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD):
        self.confidence_threshold = confidence_threshold

        self.nb_pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2))),
            ("clf", MultinomialNB(alpha=0.5)),
        ])

        self.lr_pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2))),
            ("clf", LogisticRegression(random_state=42, max_iter=1000, C=1.5)),
        ])

    # ─── Training ──────────────────────────────────────────────────────────

    def train_and_save(self, data_path: str = "data/sample_reviews.csv"):
        """Train both pipelines on the labelled CSV, then persist to disk."""
        df = pd.read_csv(data_path)
        if df.empty:
            raise ValueError("Dataset is empty.")

        X, y = df["Sentence"], df["Sentiment"]

        print("Training Naive Bayes…")
        self.nb_pipeline.fit(X, y)

        print("Training Logistic Regression…")
        self.lr_pipeline.fit(X, y)

        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(self.nb_pipeline, os.path.join(MODEL_DIR, "naive_bayes.pkl"))
        joblib.dump(self.lr_pipeline, os.path.join(MODEL_DIR, "logistic_regression.pkl"))
        print("Models saved successfully.")

    # ─── Loading ───────────────────────────────────────────────────────────

    def load_models(self):
        """Load persisted pipelines; train fresh if not found."""
        nb_path = os.path.join(MODEL_DIR, "naive_bayes.pkl")
        lr_path = os.path.join(MODEL_DIR, "logistic_regression.pkl")
        try:
            self.nb_pipeline = joblib.load(nb_path)
            self.lr_pipeline = joblib.load(lr_path)
        except FileNotFoundError:
            print("Saved models not found — training now…")
            self.train_and_save()

    # ─── Rule-based Fallback ───────────────────────────────────────────────

    def rule_based_fallback(self, sentence: str) -> str:
        """
        Negation-aware keyword fallback used when ML confidence is below threshold.
        Handles patterns like "not good", "never fast", "doesn't work".
        """
        positive_words = {
            "good", "great", "excellent", "amazing", "love", "loved",
            "best", "awesome", "stunning", "impressive", "outstanding",
            "perfect", "fantastic", "brilliant", "smooth", "fast",
        }
        negative_words = {
            "bad", "terrible", "horrible", "worst", "poor", "hate",
            "awful", "lag", "expensive", "slow", "disappointing",
            "frustrating", "blurry", "weak", "flimsy", "crack",
        }

        # Tokenise into (negation_flag, word) pairs
        tokens = re.findall(r"\b\w+\b", sentence.lower())
        negation_words = {"not", "no", "never", "cannot", "can't", "doesn't",
                          "isn't", "wasn't", "won't", "hardly", "barely"}

        pos_count = neg_count = 0
        negate = False
        for token in tokens:
            if token in negation_words:
                negate = True
                continue
            if token in positive_words:
                if negate:
                    neg_count += 1
                else:
                    pos_count += 1
                negate = False
            elif token in negative_words:
                if negate:
                    pos_count += 1
                else:
                    neg_count += 1
                negate = False
            else:
                negate = False  # reset negation after irrelevant word

        if pos_count > neg_count:
            return "Positive"
        elif neg_count > pos_count:
            return "Negative"
        return "Neutral"

    # ─── Single Prediction ─────────────────────────────────────────────────

    def predict(self, sentence: str, model_choice: str = "Logistic Regression"):
        """
        Predict sentiment for one sentence.

        Returns:
            (predicted_label, proba_dict, used_fallback)
            • predicted_label  – "Positive" | "Negative" | "Neutral"
            • proba_dict       – {class: probability}
            • used_fallback    – True if confidence was below threshold
        """
        pipeline = self.nb_pipeline if model_choice == "Naive Bayes" else self.lr_pipeline

        pred = pipeline.predict([sentence])[0]

        try:
            proba_arr = pipeline.predict_proba([sentence])[0]
            classes = pipeline.classes_
            proba_dict = {classes[i]: float(proba_arr[i]) for i in range(len(classes))}
            max_prob = float(max(proba_arr))

            if max_prob < self.confidence_threshold:
                fallback_pred = self.rule_based_fallback(sentence)
                return fallback_pred, proba_dict, True

        except AttributeError:
            proba_dict = {}
            max_prob = 1.0

        return pred, proba_dict, False

    # ─── Batch Prediction ──────────────────────────────────────────────────

    def predict_batch(self, sentences: list[str], model_choice: str = "Logistic Regression"):
        """
        Efficiently predict sentiment for a list of sentences in one pass.
        Returns a list of (label, proba_dict, used_fallback) tuples.
        """
        if not sentences:
            return []

        pipeline = self.nb_pipeline if model_choice == "Naive Bayes" else self.lr_pipeline

        pred_labels = pipeline.predict(sentences)
        try:
            proba_matrix = pipeline.predict_proba(sentences)
            classes = pipeline.classes_
        except AttributeError:
            proba_matrix = None
            classes = None

        results = []
        for i, sentence in enumerate(sentences):
            label = pred_labels[i]
            if proba_matrix is not None:
                proba_dict = {classes[j]: float(proba_matrix[i][j]) for j in range(len(classes))}
                max_prob = float(max(proba_matrix[i]))
                if max_prob < self.confidence_threshold:
                    label = self.rule_based_fallback(sentence)
                    used_fallback = True
                else:
                    used_fallback = False
            else:
                proba_dict = {}
                used_fallback = False

            results.append((label, proba_dict, used_fallback))

        return results

    # ─── Evaluation ────────────────────────────────────────────────────────

    def evaluate(self, data_path: str = "data/sample_reviews.csv") -> dict:
        """
        Evaluate both models on the CSV. Returns metrics including F1-score.
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, confusion_matrix,
        )

        df = pd.read_csv(data_path)
        X, y_true = df["Sentence"], df["Sentiment"]

        nb_preds = self.nb_pipeline.predict(X)
        lr_preds = self.lr_pipeline.predict(X)
        classes = self.lr_pipeline.classes_

        def _metrics(y_pred):
            return {
                "Accuracy":         accuracy_score(y_true, y_pred),
                "Precision":        precision_score(y_true, y_pred, average="weighted", zero_division=0),
                "Recall":           recall_score(y_true, y_pred, average="weighted", zero_division=0),
                "F1":               f1_score(y_true, y_pred, average="weighted", zero_division=0),
                "Confusion Matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
            }

        return {
            "classes":             classes.tolist(),
            "Naive Bayes":         _metrics(nb_preds),
            "Logistic Regression": _metrics(lr_preds),
        }


if __name__ == "__main__":
    m = SentimentModel()
    m.train_and_save()
