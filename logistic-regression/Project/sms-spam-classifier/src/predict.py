
from pathlib import Path
import json
import numpy as np
import joblib

from src.preprocessing import clean_text


MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def load_artifacts():
    weights_path = MODEL_DIR / "weights.npy"
    bias_path = MODEL_DIR / "bias.npy"
    vectorizer_path = MODEL_DIR / "vectorizer.joblib"
    metrics_path = MODEL_DIR / "metrics.json"

    required_files = [
        weights_path,
        bias_path,
        vectorizer_path,
        metrics_path
    ]

    for file in required_files:
        if not file.exists():
            raise FileNotFoundError(
                f"Missing model artifact: {file.name}. Train the model first."
            )

    weights = np.load(weights_path)
    bias = float(np.load(bias_path))
    vectorizer = joblib.load(vectorizer_path)

    with open(metrics_path, "r") as file:
        metrics = json.load(file)

    return weights, bias, vectorizer, metrics


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def predict_message(message):

    weights, bias, vectorizer, _ = load_artifacts()
    cleaned_message = clean_text(message)
    message_vector = vectorizer.transform([cleaned_message])
    spam_probability = float(
        sigmoid(message_vector @ weights + bias)[0]
    )

    ham_probability = 1 - spam_probability
    label = "SPAM" if spam_probability >= 0.5 else "HAM"
    confidence = max(spam_probability, ham_probability)

    return {
        "label": label,
        "confidence": confidence,
        "spam_probability": spam_probability,
        "ham_probability": ham_probability
    }
