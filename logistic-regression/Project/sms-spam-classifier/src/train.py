import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from src.model import LogisticRegressionScratch
from src.preprocessing import clean_text

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "SMSSpamCollection"
MODEL_DIR = ROOT / "models"

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset missing: {DATA_PATH}. Put SMSSpamCollection in the data folder."
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        DATA_PATH,
        sep="\t",
        names=["label", "message"],
        encoding="utf-8"
    )
    df = df.dropna(subset=["label", "message"]).drop_duplicates().copy()
    df["clean_message"] = df["message"].apply(clean_text)
    df["label_encoded"] = df["label"].map({"ham": 0, "spam": 1})
    df = df.dropna(subset=["label_encoded"]).copy()
    df["label_encoded"] = df["label_encoded"].astype(int)

    X = df["clean_message"]
    y = df["label_encoded"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(
        max_features=5000, min_df=2,
        ngram_range=(1, 2), sublinear_tf=True
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegressionScratch(
        learning_rate=0.5, epochs=1000, l2=0.01
    )
    model.fit(X_train_vec, y_train.to_numpy())


    # Training diagnostics
    train_probs = model.predict_proba(X_train_vec)
    test_probs = model.predict_proba(X_test_vec)

    print("\n--- Training Diagnostics ---")

    print("Initial loss:", model.loss_history[0])
    print("Final loss:", model.loss_history[-1])

    print("\nTraining predictions:")
    print("HAM:", np.sum(train_probs < 0.5))
    print("SPAM:", np.sum(train_probs >= 0.5))

    print("\nTest predictions:")
    print("HAM:", np.sum(test_probs < 0.5))
    print("SPAM:", np.sum(test_probs >= 0.5))

    print("\nSpam probability statistics:")
    print("Minimum:", np.min(test_probs))
    print("Maximum:", np.max(test_probs))
    print("Mean:", np.mean(test_probs))


    predictions = model.predict(X_test_vec)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(
            y_test, predictions, target_names=["ham", "spam"],
            zero_division=0, output_dict=True
        ),
        "training_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "vocabulary_size": int(len(vectorizer.vocabulary_))
    }

    np.save(MODEL_DIR / "weights.npy", model.weights)
    np.save(MODEL_DIR / "bias.npy", np.array([model.bias]))
    joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.joblib")
    with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print("Training complete")
    for key in ["accuracy", "precision", "recall", "f1_score"]:
        print(f"{key}: {metrics[key]:.4f}")
    print("Confusion matrix:")
    print(np.array(metrics["confusion_matrix"]))
    print(f"Artifacts saved to: {MODEL_DIR}")

if __name__ == "__main__":
    main()
