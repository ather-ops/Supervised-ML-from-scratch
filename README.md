# Supervised Machine Learning from Scratch

<p align="center">
  <img src="Assets/repo-cover.png" alt="Supervised Machine Learning from Scratch" width="100%">
</p>

Machine learning algorithms built from the ground up using only Python, NumPy, Pandas, and Matplotlib. No Scikit-Learn, no TensorFlow, no black boxes. Every gradient, loss function, and normalization step is implemented by hand.

**If you can build it from scratch, you truly understand it.**

> Looking for the applied version? See the companion repo, [Supervised ML with Scikit-Learn](https://github.com/ather-ops/Supervised-ML-with-Scikit-Learn), which uses industry-standard tools and workflows.

## Live Demo

[**Launch the House Price Predictor →**](https://ather-ops-supervised-ml-fro-linear-regressionprojectsapp-g6x3xg.streamlit.app/)

## What's Inside

- Linear Regression from scratch
- Gradient descent and the math behind it
- Feature and target scaling
- Evaluation metrics (MAE, MSE, RMSE)
- End-to-end projects with preprocessing and model development

## Featured Project: House Price Predictor

<p align="center">
  <img src="linear-regression/Projects/assets/cover.png" alt="House Price Predictor" width="100%">
</p>

Predicts house prices with a NumPy-only Linear Regression model, trained on 545 records with five numerical features.

- Data inspection and preprocessing
- Custom train-test split
- Feature and target scaling
- Gradient descent training
- Evaluation with MAE, MSE, and RMSE
- Saving and loading trained parameters
- Interactive Streamlit app for predictions

**Location:** [`linear-regression/Projects/`](linear-regression/Projects/)

## Getting Started

```bash
git clone https://github.com/ather-ops/Supervised-ML-from-scratch.git
cd Supervised-ML-from-scratch
pip install -r requirements.txt
```

**Run the student scores example:**

```bash
python "Linear-Regression/student_scores_example.py"
```

**Run the House Price Predictor:**

```bash
cd linear-regression/Projects
pip install -r requirements.txt
python train.py
streamlit run app.py
```

> `all_models.py` will be added once the individual model implementations are complete.

## Requirements

- Python with `numpy>=1.21.0`, `pandas>=1.3.0`, `matplotlib>=3.4.0`
- `streamlit` (House Price Predictor only)

## Learning Path

| Stage | Focus |
|:---:|---|
| 01 | Mathematics behind machine learning |
| 02 | Implementing algorithms from scratch |
| 03 | Training and evaluating models |
| 04 | Building practical ML projects |
| 05 | Understanding the engineering behind ML systems |

## Related Repositories

| Repository | Purpose |
|---|---|
| [Supervised ML with Scikit-Learn](https://github.com/ather-ops/Supervised-ML-with-Scikit-Learn/tree/main/02-Linear-Regression) | The same concepts using Scikit-Learn, pipelines, and practical projects |

Understand the math first, then see how professional libraries organize the same workflows.

## License

MIT License. Use freely.

## Author

[ather-ops](https://github.com/ather-ops)
