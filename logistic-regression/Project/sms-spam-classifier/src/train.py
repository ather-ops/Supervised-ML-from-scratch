import json
import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.preprocessing import clean_text
from src.model import LogisticRegressionScratch


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "SMSSpamCollection"
MODEL_DIR = ROOT / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


print("\nLoading dataset...")

df = pd.read_csv(
    DATA_PATH,
    sep="\t",
    names=["label", "message"],
    encoding="utf-8"
)

print("Original dataset:", len(df))

# Remove missing values and duplicates
df = df.dropna(subset=["label", "message"])
df = df.drop_duplicates().copy()

# Clean messages
df["clean_message"] = df["message"].apply(clean_text)

# Encode labels
df["label_encoded"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

df = df.dropna(subset=["label_encoded"]).copy()
df["label_encoded"] = df["label_encoded"].astype(int)

print("Dataset after cleaning:", len(df))

print("\nClass distribution:")
print(df["label"].value_counts())


X = df["clean_message"]
y = df["label_encoded"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


print("\nApplying TF-IDF...")

vectorizer = TfidfVectorizer(
    max_features=5000,
    min_df=2,
    ngram_range=(1, 2),
    sublinear_tf=True
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("Vocabulary size:", len(vectorizer.vocabulary_))


print("\nTraining class-weighted Logistic Regression...")

model = LogisticRegressionScratch(
    learning_rate=0.5,
    epochs=1000,
    l2=0.01
)

model.fit(
    X_train_vec,
    y_train.to_numpy()
)

print("Training completed!")


y_prob = model.predict_proba(X_test_vec)
THRESHOLD = 0.5

y_pred = (y_prob >= THRESHOLD).astype(int)


accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["HAM", "SPAM"],
        zero_division=0
    )
)

train_probs = model.predict_proba(X_train_vec)
print("Initial loss:", model.loss_history[0])
print("Final loss:", model.loss_history[-1])

print("\nTraining predictions:")
print("HAM:", np.sum(train_probs < THRESHOLD))
print("SPAM:", np.sum(train_probs >= THRESHOLD))

print("\nTesting predictions:")
print("HAM:", np.sum(y_prob < THRESHOLD))
print("SPAM:", np.sum(y_prob >= THRESHOLD))

print("\nSpam probability statistics:")
print("Minimum:", np.min(y_prob))
print("Maximum:", np.max(y_prob))
print("Mean:", np.mean(y_prob))


print("\nSaving model artifacts...")

np.save(
    MODEL_DIR / "weights.npy",
    model.weights
)

np.save(
    MODEL_DIR / "bias.npy",
    np.array([model.bias])
)

joblib.dump(
    vectorizer,
    MODEL_DIR / "tfidf_vectorizer.joblib"
)



metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "threshold": THRESHOLD,
    "training_samples": int(len(X_train)),
    "test_samples": int(len(X_test)),
    "vocabulary_size": int(len(vectorizer.vocabulary_)),
    "confusion_matrix": cm.tolist()
}

with open(MODEL_DIR / "metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)


print(f"Accuracy:  {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall:    {recall * 100:.2f}%")
print(f"F1 Score:  {f1 * 100:.2f}%")

print("\nModel artifacts saved successfully!")
print("Location:", MODEL_DIR)
