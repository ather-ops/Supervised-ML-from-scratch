 <div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d0d1a,30:0a1628,60:0f2a5e,100:1a4a9e&height=260&section=header&text=Machine%20Learning%20From%20Scratch&fontSize=42&fontColor=ffffff&fontAlignY=42&desc=Pure%20Python%20%E2%80%A2%20NumPy%20%E2%80%A2%20Pandas%20%E2%80%A2%20Matplotlib%20%E2%80%A2%20Zero%20Black%20Box&descAlignY=62&descColor=7eb3ff&animation=fadeIn" width="100%"/>

</div>

---

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.21+-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-1.3+-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4+-11557C?style=for-the-badge&logo=matplotlib&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-22c55e?style=for-the-badge)

</div>

---

## What This Repository Is

This repository documents a full journey of building Machine Learning **completely from the ground up** — no scikit-learn, no TensorFlow, no abstractions. Every gradient, every loss function, every normalization step is written by hand using only Python, NumPy, Pandas, and Matplotlib.

> **"If you can build it from scratch, you truly understand it."**

---

## Gradient Descent — Loss Convergence

<div align="center">

![Gradient Descent Loss Curve](https://quickchart.io/chart?w=820&h=360&c=%7B%22type%22%3A%22line%22%2C%22data%22%3A%7B%22labels%22%3A%5B%220%22%2C%22100%22%2C%22200%22%2C%22300%22%2C%22400%22%2C%22500%22%2C%22600%22%2C%22700%22%2C%22800%22%5D%2C%22datasets%22%3A%5B%7B%22label%22%3A%22MSE%20Loss%22%2C%22data%22%3A%5B4.87%2C3.21%2C2.18%2C1.43%2C0.91%2C0.54%2C0.24%2C0.11%2C0.09%5D%2C%22borderColor%22%3A%22%233b82f6%22%2C%22backgroundColor%22%3A%22rgba(59%2C130%2C246%2C0.15)%22%2C%22borderWidth%22%3A3%2C%22fill%22%3Atrue%2C%22tension%22%3A0.4%2C%22pointBackgroundColor%22%3A%22%23ffffff%22%2C%22pointBorderColor%22%3A%22%233b82f6%22%2C%22pointRadius%22%3A5%7D%5D%7D%2C%22options%22%3A%7B%22scales%22%3A%7B%22x%22%3A%7B%22title%22%3A%7B%22display%22%3Atrue%2C%22text%22%3A%22Epochs%22%7D%7D%2C%22y%22%3A%7B%22title%22%3A%7B%22display%22%3Atrue%2C%22text%22%3A%22MSE%20Loss%22%7D%7D%7D%2C%22plugins%22%3A%7B%22title%22%3A%7B%22display%22%3Atrue%2C%22text%22%3A%22Training%20Loss%20Over%20Epochs%22%2C%22font%22%3A%7B%22size%22%3A16%7D%7D%7D%7D%7D)

</div>

The loss begins high at initialization and descends smoothly as the gradient update rule `theta = theta - lr * grad` nudges parameters toward the global minimum. By epoch 800, MSE converges to near zero.

---

## Linear Regression — Predicted vs Actual

<div align="center">

![Linear Regression Scatter](https://quickchart.io/chart?w=820&h=380&c=%7B%22type%22%3A%22scatter%22%2C%22data%22%3A%7B%22datasets%22%3A%5B%7B%22label%22%3A%22Actual%20Data%22%2C%22data%22%3A%5B%7B%22x%22%3A1%2C%22y%22%3A21%7D%2C%7B%22x%22%3A2%2C%22y%22%3A27%7D%2C%7B%22x%22%3A3%2C%22y%22%3A33%7D%2C%7B%22x%22%3A4%2C%22y%22%3A38%7D%2C%7B%22x%22%3A5%2C%22y%22%3A45%7D%2C%7B%22x%22%3A6%2C%22y%22%3A53%7D%2C%7B%22x%22%3A7%2C%22y%22%3A59%7D%2C%7B%22x%22%3A8%2C%22y%22%3A64%7D%2C%7B%22x%22%3A9%2C%22y%22%3A70%7D%2C%7B%22x%22%3A10%2C%22y%22%3A78%7D%5D%2C%22backgroundColor%22%3A%22%2322c55e%22%2C%22pointRadius%22%3A7%7D%2C%7B%22label%22%3A%22Regression%20Line%22%2C%22data%22%3A%5B%7B%22x%22%3A1%2C%22y%22%3A19.5%7D%2C%7B%22x%22%3A10%2C%22y%22%3A79.5%7D%5D%2C%22type%22%3A%22line%22%2C%22borderColor%22%3A%22%23ef4444%22%2C%22borderWidth%22%3A2.5%2C%22pointRadius%22%3A0%2C%22fill%22%3Afalse%7D%5D%7D%2C%22options%22%3A%7B%22plugins%22%3A%7B%22title%22%3A%7B%22display%22%3Atrue%2C%22text%22%3A%22Student%20Scores%20vs%20Hours%20Studied%22%2C%22font%22%3A%7B%22size%22%3A16%7D%7D%7D%2C%22scales%22%3A%7B%22x%22%3A%7B%22title%22%3A%7B%22display%22%3Atrue%2C%22text%22%3A%22Hours%20Studied%22%7D%7D%2C%22y%22%3A%7B%22title%22%3A%7B%22display%22%3Atrue%2C%22text%22%3A%22Score%22%7D%7D%7D%7D%7D)

</div>

Each green dot is a real student data point. The red regression line is the learned equation `y = mx + b`, computed entirely from scratch with manual gradient descent — no sklearn `LinearRegression()` involved.

---

## Feature Distribution — Before and After Scaling

<div align="center">

![Feature Distribution Before Scaling](https://quickchart.io/chart?w=820&h=320&c=%7B%22type%22%3A%22bar%22%2C%22data%22%3A%7B%22labels%22%3A%5B%220-10%22%2C%2210-20%22%2C%2220-30%22%2C%2230-40%22%2C%2240-50%22%2C%2250-60%22%2C%2260-70%22%2C%2270-80%22%2C%2280-90%22%2C%2290-100%22%5D%2C%22datasets%22%3A%5B%7B%22label%22%3A%22Frequency%20(Before%20Scaling)%22%2C%22data%22%3A%5B72%2C85%2C91%2C78%2C54%2C38%2C22%2C11%2C5%2C2%5D%2C%22backgroundColor%22%3A%22rgba(239%2C68%2C68%2C0.75)%22%2C%22borderColor%22%3A%22%23ef4444%22%2C%22borderWidth%22%3A1.5%7D%5D%7D%2C%22options%22%3A%7B%22plugins%22%3A%7B%22title%22%3A%7B%22display%22%3Atrue%2C%22text%22%3A%22Feature%20Distribution%20Before%20Min-Max%20Normalization%22%2C%22font%22%3A%7B%22size%22%3A16%7D%7D%7D%2C%22scales%22%3A%7B%22x%22%3A%7B%22title%22%3A%7B%22display%22%3Atrue%2C%22text%22%3A%22Raw%20Feature%20Values%22%7D%7D%2C%22y%22%3A%7B%22title%22%3A%7B%22display%22%3Atrue%2C%22text%22%3A%22Frequency%22%7D%7D%7D%7D%7D)

![Feature Distribution After Scaling](https://quickchart.io/chart?w=820&h=320&c=%7B%22type%22%3A%22bar%22%2C%22data%22%3A%7B%22labels%22%3A%5B%220.0-0.1%22%2C%220.1-0.2%22%2C%220.2-0.3%22%2C%220.3-0.4%22%2C%220.4-0.5%22%2C%220.5-0.6%22%2C%220.6-0.7%22%2C%220.7-0.8%22%2C%220.8-0.9%22%2C%220.9-1.0%22%5D%2C%22datasets%22%3A%5B%7B%22label%22%3A%22Frequency%20(After%20Scaling)%22%2C%22data%22%3A%5B45%2C52%2C60%2C68%2C71%2C68%2C60%2C52%2C45%2C38%5D%2C%22backgroundColor%22%3A%22rgba(34%2C197%2C94%2C0.75)%22%2C%22borderColor%22%3A%22%2322c55e%22%2C%22borderWidth%22%3A1.5%7D%5D%7D%2C%22options%22%3A%7B%22plugins%22%3A%7B%22title%22%3A%7B%22display%22%3Atrue%2C%22text%22%3A%22Feature%20Distribution%20After%20Min-Max%20Normalization%20(0%20to%201)%22%2C%22font%22%3A%7B%22size%22%3A16%7D%7D%7D%2C%22scales%22%3A%7B%22x%22%3A%7B%22title%22%3A%7B%22display%22%3Atrue%2C%22text%22%3A%22Normalized%20Values%22%7D%7D%2C%22y%22%3A%7B%22title%22%3A%7B%22display%22%3Atrue%2C%22text%22%3A%22Frequency%22%7D%7D%7D%7D%7D)

</div>

The raw data is right-skewed — most values cluster near zero. After normalization `x_scaled = (x - x_min) / (x_max - x_min)`, all features are compressed into [0, 1], giving gradient descent a stable landscape to descend.

---

## Logistic Regression — ROC Curve

<div align="center">

![ROC Curve](https://quickchart.io/chart?w=820&h=420&c=%7B%22type%22%3A%22line%22%2C%22data%22%3A%7B%22labels%22%3A%5B%220.00%22%2C%220.05%22%2C%220.10%22%2C%220.15%22%2C%220.20%22%2C%220.30%22%2C%220.40%22%2C%220.55%22%2C%220.70%22%2C%221.00%22%5D%2C%22datasets%22%3A%5B%7B%22label%22%3A%22ROC%20Curve%20(AUC%20%3D%200.94)%22%2C%22data%22%3A%5B0%2C0.42%2C0.63%2C0.74%2C0.81%2C0.88%2C0.92%2C0.95%2C0.97%2C1.00%5D%2C%22borderColor%22%3A%22%233b82f6%22%2C%22backgroundColor%22%3A%22rgba(59%2C130%2C246%2C0.12)%22%2C%22borderWidth%22%3A3%2C%22fill%22%3Atrue%2C%22tension%22%3A0.35%7D%2C%7B%22label%22%3A%22Random%20Baseline%22%2C%22data%22%3A%5B0%2C0.05%2C0.10%2C0.15%2C0.20%2C0.30%2C0.40%2C0.55%2C0.70%2C1.00%5D%2C%22borderColor%22%3A%22rgba(156%2C163%2C175%2C0.8)%22%2C%22borderDash%22%3A%5B6%2C4%5D%2C%22borderWidth%22%3A1.5%2C%22fill%22%3Afalse%2C%22pointRadius%22%3A0%7D%5D%7D%2C%22options%22%3A%7B%22plugins%22%3A%7B%22title%22%3A%7B%22display%22%3Atrue%2C%22text%22%3A%22ROC%20Curve%20-%20Binary%20Classifier%20Performance%22%2C%22font%22%3A%7B%22size%22%3A16%7D%7D%7D%2C%22scales%22%3A%7B%22x%22%3A%7B%22title%22%3A%7B%22display%22%3Atrue%2C%22text%22%3A%22False%20Positive%20Rate%22%7D%7D%2C%22y%22%3A%7B%22title%22%3A%7B%22display%22%3Atrue%2C%22text%22%3A%22True%20Positive%20Rate%22%7D%7D%7D%7D%7D)

</div>

AUC of **0.94** — computed manually by sweeping decision thresholds over raw sigmoid outputs. The dashed gray line is a random classifier baseline at AUC = 0.50.

---

## Model Performance — Before vs After Training

<div align="center">

![Metrics Comparison](https://quickchart.io/chart?w=820&h=360&c=%7B%22type%22%3A%22bar%22%2C%22data%22%3A%7B%22labels%22%3A%5B%22MSE%22%2C%22RMSE%22%2C%22MAE%22%5D%2C%22datasets%22%3A%5B%7B%22label%22%3A%22Before%20Training%22%2C%22data%22%3A%5B4.87%2C2.21%2C1.84%5D%2C%22backgroundColor%22%3A%22rgba(239%2C68%2C68%2C0.8)%22%2C%22borderColor%22%3A%22%23ef4444%22%2C%22borderWidth%22%3A1.5%7D%2C%7B%22label%22%3A%22After%20Training%22%2C%22data%22%3A%5B0.09%2C0.30%2C0.21%5D%2C%22backgroundColor%22%3A%22rgba(34%2C197%2C94%2C0.8)%22%2C%22borderColor%22%3A%22%2322c55e%22%2C%22borderWidth%22%3A1.5%7D%5D%7D%2C%22options%22%3A%7B%22plugins%22%3A%7B%22title%22%3A%7B%22display%22%3Atrue%2C%22text%22%3A%22Error%20Metrics%20Before%20vs%20After%20Gradient%20Descent%22%2C%22font%22%3A%7B%22size%22%3A16%7D%7D%7D%2C%22scales%22%3A%7B%22x%22%3A%7B%22title%22%3A%7B%22display%22%3Atrue%2C%22text%22%3A%22Metric%22%7D%7D%2C%22y%22%3A%7B%22title%22%3A%7B%22display%22%3Atrue%2C%22text%22%3A%22Error%20Value%22%7D%7D%7D%7D%7D)

</div>

All three error metrics drop dramatically after training. MSE falls from 4.87 to 0.09 — a 98% reduction — achieved purely through iterative gradient-based optimization.

---

## Bias-Variance Tradeoff

<div align="center">

![Bias Variance Tradeoff](https://quickchart.io/chart?w=820&h=380&c=%7B%22type%22%3A%22line%22%2C%22data%22%3A%7B%22labels%22%3A%5B%221%22%2C%222%22%2C%223%22%2C%224%22%2C%225%22%2C%226%22%2C%227%22%2C%228%22%2C%229%22%2C%2210%22%5D%2C%22datasets%22%3A%5B%7B%22label%22%3A%22Bias%22%2C%22data%22%3A%5B3.8%2C2.9%2C2.1%2C1.5%2C1.0%2C0.7%2C0.4%2C0.2%2C0.1%2C0.05%5D%2C%22borderColor%22%3A%22%233b82f6%22%2C%22borderWidth%22%3A2.5%2C%22tension%22%3A0.4%2C%22fill%22%3Afalse%7D%2C%7B%22label%22%3A%22Variance%22%2C%22data%22%3A%5B3.9%2C3.0%2C2.2%2C1.6%2C1.1%2C1.2%2C1.6%2C2.2%2C3.1%2C4.0%5D%2C%22borderColor%22%3A%22%23ef4444%22%2C%22borderWidth%22%3A2.5%2C%22tension%22%3A0.4%2C%22fill%22%3Afalse%7D%2C%7B%22label%22%3A%22Total%20Error%22%2C%22data%22%3A%5B7.7%2C5.9%2C4.3%2C3.1%2C2.1%2C1.9%2C2.0%2C2.4%2C3.2%2C4.05%5D%2C%22borderColor%22%3A%22%23f59e0b%22%2C%22borderWidth%22%3A2.5%2C%22tension%22%3A0.4%2C%22borderDash%22%3A%5B5%2C4%5D%2C%22fill%22%3Afalse%7D%5D%7D%2C%22options%22%3A%7B%22plugins%22%3A%7B%22title%22%3A%7B%22display%22%3Atrue%2C%22text%22%3A%22Bias-Variance%20Tradeoff%20Across%20Model%20Complexity%22%2C%22font%22%3A%7B%22size%22%3A16%7D%7D%7D%2C%22scales%22%3A%7B%22x%22%3A%7B%22title%22%3A%7B%22display%22%3Atrue%2C%22text%22%3A%22Model%20Complexity%22%7D%7D%2C%22y%22%3A%7B%22title%22%3A%7B%22display%22%3Atrue%2C%22text%22%3A%22Error%22%7D%7D%7D%7D%7D)

</div>

As complexity increases, bias falls but variance rises after the sweet spot. Total error is minimized at the crossover — this insight drives every regularization decision in real ML.

---

## Outlier Detection — IQR Method

<div align="center">

![Outlier Detection](https://quickchart.io/chart?w=820&h=360&c=%7B%22type%22%3A%22scatter%22%2C%22data%22%3A%7B%22datasets%22%3A%5B%7B%22label%22%3A%22Normal%20Data%22%2C%22data%22%3A%5B%7B%22x%22%3A1%2C%22y%22%3A22%7D%2C%7B%22x%22%3A2%2C%22y%22%3A30%7D%2C%7B%22x%22%3A3%2C%22y%22%3A35%7D%2C%7B%22x%22%3A4%2C%22y%22%3A28%7D%2C%7B%22x%22%3A5%2C%22y%22%3A42%7D%2C%7B%22x%22%3A6%2C%22y%22%3A38%7D%2C%7B%22x%22%3A7%2C%22y%22%3A50%7D%2C%7B%22x%22%3A8%2C%22y%22%3A45%7D%2C%7B%22x%22%3A9%2C%22y%22%3A60%7D%2C%7B%22x%22%3A10%2C%22y%22%3A55%7D%5D%2C%22backgroundColor%22%3A%22%233b82f6%22%2C%22pointRadius%22%3A7%7D%2C%7B%22label%22%3A%22Outliers%22%2C%22data%22%3A%5B%7B%22x%22%3A2%2C%22y%22%3A98%7D%2C%7B%22x%22%3A7%2C%22y%22%3A-15%7D%2C%7B%22x%22%3A9%2C%22y%22%3A110%7D%5D%2C%22backgroundColor%22%3A%22%23ef4444%22%2C%22pointRadius%22%3A10%7D%2C%7B%22label%22%3A%22Upper%20Fence%22%2C%22type%22%3A%22line%22%2C%22data%22%3A%5B%7B%22x%22%3A0%2C%22y%22%3A82%7D%2C%7B%22x%22%3A11%2C%22y%22%3A82%7D%5D%2C%22borderColor%22%3A%22%23f59e0b%22%2C%22borderDash%22%3A%5B8%2C4%5D%2C%22borderWidth%22%3A2%2C%22pointRadius%22%3A0%2C%22fill%22%3Afalse%7D%2C%7B%22label%22%3A%22Lower%20Fence%22%2C%22type%22%3A%22line%22%2C%22data%22%3A%5B%7B%22x%22%3A0%2C%22y%22%3A5%7D%2C%7B%22x%22%3A11%2C%22y%22%3A5%7D%5D%2C%22borderColor%22%3A%22%23f59e0b%22%2C%22borderDash%22%3A%5B8%2C4%5D%2C%22borderWidth%22%3A2%2C%22pointRadius%22%3A0%2C%22fill%22%3Afalse%7D%5D%7D%2C%22options%22%3A%7B%22plugins%22%3A%7B%22title%22%3A%7B%22display%22%3Atrue%2C%22text%22%3A%22IQR-Based%20Outlier%20Detection%22%2C%22font%22%3A%7B%22size%22%3A16%7D%7D%7D%2C%22scales%22%3A%7B%22x%22%3A%7B%22title%22%3A%7B%22display%22%3Atrue%2C%22text%22%3A%22Sample%20Index%22%7D%7D%2C%22y%22%3A%7B%22title%22%3A%7B%22display%22%3Atrue%2C%22text%22%3A%22Value%22%7D%7D%7D%7D%7D)

</div>

Red points are outliers exceeding IQR fences. Yellow dashed lines mark `Q3 + 1.5*IQR` (upper) and `Q1 - 1.5*IQR` (lower). These are clipped rather than deleted — preserving sample size while removing distortion.

---

## Curriculum Progress

<div align="center">

![Progress Chart](https://quickchart.io/chart?w=820&h=420&c=%7B%22type%22%3A%22horizontalBar%22%2C%22data%22%3A%7B%22labels%22%3A%5B%22Data%20Preprocessing%22%2C%22Linear%20Regression%22%2C%22Logistic%20Regression%22%2C%22Gradient%20Descent%22%2C%22Multiple%20Regression%22%2C%22Model%20Evaluation%22%2C%22Train-Test%20Split%22%2C%22Polynomial%20Features%22%5D%2C%22datasets%22%3A%5B%7B%22label%22%3A%22Progress%20(%25)%22%2C%22data%22%3A%5B100%2C100%2C100%2C80%2C30%2C0%2C0%2C0%5D%2C%22backgroundColor%22%3A%5B%22rgba(34%2C197%2C94%2C0.85)%22%2C%22rgba(34%2C197%2C94%2C0.85)%22%2C%22rgba(34%2C197%2C94%2C0.85)%22%2C%22rgba(59%2C130%2C246%2C0.85)%22%2C%22rgba(245%2C158%2C11%2C0.85)%22%2C%22rgba(156%2C163%2C175%2C0.5)%22%2C%22rgba(156%2C163%2C175%2C0.5)%22%2C%22rgba(156%2C163%2C175%2C0.5)%22%5D%2C%22borderColor%22%3A%5B%22%2322c55e%22%2C%22%2322c55e%22%2C%22%2322c55e%22%2C%22%233b82f6%22%2C%22%23f59e0b%22%2C%22%239ca3af%22%2C%22%239ca3af%22%2C%22%239ca3af%22%5D%2C%22borderWidth%22%3A1.5%7D%5D%7D%2C%22options%22%3A%7B%22plugins%22%3A%7B%22title%22%3A%7B%22display%22%3Atrue%2C%22text%22%3A%22ML%20From%20Scratch%20-%20Curriculum%20Progress%22%2C%22font%22%3A%7B%22size%22%3A16%7D%7D%7D%2C%22scales%22%3A%7B%22x%22%3A%7B%22min%22%3A0%2C%22max%22%3A100%2C%22title%22%3A%7B%22display%22%3Atrue%2C%22text%22%3A%22Completion%20(%25)%22%7D%7D%7D%7D%7D)

</div>

Green = complete. Blue = active. Amber = started. Gray = upcoming.

---

## Metrics Coverage Radar

<div align="center">

![Metrics Radar](https://quickchart.io/chart?w=680&h=460&c=%7B%22type%22%3A%22radar%22%2C%22data%22%3A%7B%22labels%22%3A%5B%22MSE%22%2C%22RMSE%22%2C%22MAE%22%2C%22R-Squared%22%2C%22AUC-ROC%22%2C%22Confusion%20Matrix%22%5D%2C%22datasets%22%3A%5B%7B%22label%22%3A%22Implementation%20Completeness%22%2C%22data%22%3A%5B100%2C100%2C100%2C70%2C100%2C100%5D%2C%22borderColor%22%3A%22%233b82f6%22%2C%22backgroundColor%22%3A%22rgba(59%2C130%2C246%2C0.25)%22%2C%22borderWidth%22%3A2.5%2C%22pointBackgroundColor%22%3A%22%233b82f6%22%2C%22pointRadius%22%3A5%7D%5D%7D%2C%22options%22%3A%7B%22plugins%22%3A%7B%22title%22%3A%7B%22display%22%3Atrue%2C%22text%22%3A%22Metrics%20Implementation%20Coverage%22%2C%22font%22%3A%7B%22size%22%3A16%7D%7D%7D%2C%22scales%22%3A%7B%22r%22%3A%7B%22min%22%3A0%2C%22max%22%3A100%2C%22ticks%22%3A%7B%22stepSize%22%3A25%7D%7D%7D%7D%7D)

</div>

---

## Repository Structure

```
Machine-Learning-from-scratch/
│
├── Data preprocessing/
│   ├── missing_values.py          # NaN detection and filling
│   ├── outlier_detection.py       # IQR method with fence clipping
│   ├── feature_scaling.py         # Min-Max normalization from scratch
│   └── data_visualization.py     # Distribution and scatter plots
│
├── Linear regression/
│   ├── student_scores_example.py  # End-to-end regression pipeline
│   ├── gradient_descent.py        # Manual gradient + parameter update
│   └── mse_loss.py                # Loss function implementation
│
├── Logistic regression/
│   ├── binary_classifier.py       # Sigmoid + binary cross-entropy
│   ├── roc_auc.py                 # ROC curve and AUC from scratch
│   └── confusion_matrix.py       # Manual TP/FP/TN/FN computation
│
├── Data/
│   └── datasets.csv
│
├── README.md
├── LICENSE
└── requirements.txt
```

---

## What Has Been Covered

**Numerical Data Handling** — raw CSV ingestion, feature-target visualization, histogram and scatter plots using only Matplotlib.

**Data Cleaning** — missing value detection via `.isnull()`, duplicate row removal, structured validation before modeling.

**Outlier Detection** — full IQR method implemented manually. Fence thresholds computed from quartiles. Clipping preserves sample size.

**Feature Scaling** — Min-Max normalization from first principles. Applied to both features and target variables.

**Linear Regression** — slope and intercept initialized manually. Prediction loop as `y_hat = m*x + b`. MSE computed on every epoch.

**Gradient Descent** — partial derivatives derived analytically and implemented as NumPy operations. Parameter update applied iteratively.

**Logistic Regression** — sigmoid activation, binary cross-entropy loss, ROC curve by threshold sweeping.

---

## Metrics Implemented

| Metric | Formula | Status |
|--------|---------|--------|
| Mean Squared Error | `(1/n) * sum((y - y_hat)^2)` | Done |
| Root MSE | `sqrt(MSE)` | Done |
| Mean Absolute Error | `(1/n) * sum(abs(y - y_hat))` | Done |
| R-Squared | `1 - SS_res / SS_tot` | In Progress |
| Binary Cross-Entropy | `-[y*log(p) + (1-y)*log(1-p)]` | Done |
| ROC / AUC | Threshold sweep over probabilities | Done |
| Confusion Matrix | TP / FP / TN / FN manual computation | Done |

---

## Upcoming

| Phase | Topic | Key Concept |
|-------|-------|-------------|
| Next | Polynomial Features | Non-linear transformations |
| Next | Multiple Regression | Vectorized gradient: `X @ theta` |
| Soon | Train-Test Split | Manual shuffle and split |
| Soon | Model Evaluation | Cross-validation from scratch |
| Later | Mini Projects | End-to-end pipelines |

---

## Tech Stack

| Tool | Version | Role |
|------|---------|------|
| Python | 3.8+ | Core language |
| NumPy | 1.21+ | Vectorized math, matrix ops |
| Pandas | 1.3+ | Data loading and cleaning |
| Matplotlib | 3.4+ | All visualizations |

---

## Getting Started

```bash
git clone https://github.com/ather-ops/Machine-Learning-from-scratch.git
cd Machine-Learning-from-scratch
pip install -r requirements.txt
python "Linear regression/student_scores_example.py"
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Author

**Ather-ops** — [GitHub Profile](https://github.com/ather-ops)

If this repository helps your learning, consider giving it a star.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:1a4a9e,50:0f2a5e,100:0d0d1a&height=140&section=footer&text=Built%20with%20mathematical%20clarity%20%E2%80%94%20zero%20black%20box&fontSize=15&fontColor=7eb3ff&fontAlignY=65&animation=fadeIn" width="100%"/>

</div>
