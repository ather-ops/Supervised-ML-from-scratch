# SpamShield AI

SMS Spam Classification using TF-IDF and Logistic Regression implemented from scratch with NumPy.

## Dataset

Place the tab-separated SMS Spam Collection file, with no header, at:

`data/SMSSpamCollection`

## Install dependencies

```bash
pip install -r requirements.txt
```

## Train the model

Run from the project root:

```bash
python -m src.train
```

Training saves weights, bias, the fitted TF-IDF vectorizer and test metrics in `models/`.

## Run the Streamlit app

```bash
streamlit run app.py
```

## Project notes

- Logistic Regression is implemented using NumPy, not scikit-learn's classifier.
- Scikit-learn is used for TF-IDF, train-test splitting and evaluation metrics.
- Test accuracy is a dataset-level metric. Per-message confidence is a model probability, not a guarantee of correctness.
