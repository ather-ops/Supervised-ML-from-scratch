# Supervised Machine Learning from Scratch

<p align="center">
  <img src="Assets/repo-cover.png" alt="Supervised Machine Learning from Scratch" width="100%">
</p>

Building machine learning algorithms from the ground up — no Scikit-Learn, no TensorFlow, and no black-box abstractions.

Every gradient, loss function, and normalization step is implemented by hand using Python, NumPy, Pandas, and Matplotlib.

The goal is simple: **if you can build it from scratch, you truly understand it.**

This repository focuses on the mathematical foundations of supervised machine learning. The companion repository, [Supervised ML with Scikit-Learn](https://github.com/ather-ops/Supervised-ML-with-Scikit-Learn), applies these concepts using industry-standard tools and workflows.


## What You'll Find Here

- Linear Regression implemented from scratch
- Mathematical foundations and gradient descent
- Feature and target scaling
- Model evaluation and error metrics
- Practical machine learning projects
- Data preprocessing and model development

## Featured Project: House Price Predictor

A complete machine learning project that predicts house prices using Linear Regression implemented from scratch with NumPy.

<p align="center">
  <img src="linear-regression/Projects/assets/cover.png" alt="House Price Predictor" width="100%">
</p>

**What the project covers:**

- Data inspection and preprocessing
- Custom train-test splitting
- Feature scaling and target scaling
- Gradient descent and model training
- Model evaluation using MAE, MSE, and RMSE
- Saving and loading trained model parameters
- Interactive prediction application built with Streamlit

**Dataset:** 545 housing records with five selected numerical features.

**Project location:** [`linear-regression/Projects/`](linear-regression/Projects/)

**Technologies:** Python, NumPy, Pandas, Matplotlib, Streamlit.


## Getting Started

Clone the repository:

```bash
git clone https://github.com/ather-ops/Supervised-ML-from-scratch.git
cd Supervised-ML-from-scratch
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the existing student scores example:

```bash
python "Linear-Regression/student_scores_example.py"
```

To run the House Price Predictor project:

```bash
cd linear-regression/Projects
pip install -r requirements.txt
python train.py
streamlit run app.py
```

The `all_models.py` entry point will be added when the individual model implementations are ready.



## Dependencies

```text
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
```

The House Price Predictor project additionally uses Streamlit.

No Scikit-Learn for model implementation. No shortcut libraries. That is the point.


## Learning Path

| Stage | Focus |
|---|---|
| 01 | Mathematics behind machine learning |
| 02 | Implementing algorithms from scratch |
| 03 | Training and evaluating models |
| 04 | Building practical ML projects |
| 05 | Understanding the engineering behind ML systems |


## Related Repository

| Repository | Purpose |
|---|---|
| [Supervised ML from Scratch](https://github.com/ather-ops/Supervised-ML-from-scratch) | Mathematical foundations and pure NumPy implementations |
| [Supervised ML with Scikit-Learn](https://github.com/ather-ops/Supervised-ML-with-Scikit-Learn/tree/main/02-Linear-Regression) | Applying the same concepts with Scikit-Learn, pipelines, and practical projects |

The idea is to understand the mathematics first, then learn how professional machine learning libraries implement and organize these workflows.


## License

MIT License. Use freely.

---

## Author

[ather-ops](https://github.com/ather-ops)
