import pandas as pd
import numpy as np

print("MACHINE LEARNING COMPLETE PIPELINE")
print("=" * 80)

# ================= STEP 1: WORKING WITH NUMERICAL DATA =================
data = {
    "House_size": [500, 1000, 1500, 6000, 4500],
    "price": [10, 20, 80, 40, 50]
}

df = pd.DataFrame(data)
print(df)
print("=" * 80)

# ================= STEP 2: SCRUBBING =================
# No missing or duplicate values → skipping
# (Logic is known but not applied here)

# ================= STEP 3: OUTLIER DETECTION (IQR METHOD) =================
Q1 = df["House_size"].quantile(0.25)
Q3 = df["House_size"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# ================= STEP 4: IDENTIFY OUTLIERS =================
outliers = df[
    (df["House_size"] < lower_bound) |
    (df["House_size"] > upper_bound)
]

print("Outliers detected:\n", outliers)

# ================= STEP 5: FIX OUTLIERS (CLIPPING — INDUSTRY SAFE) =================
df["House_size"] = df["House_size"].clip(lower_bound, upper_bound)

# ================= STEP 6: FINAL CLEAN DATA =================
print("\nClean Data:\n", df)
print("=" * 80)

# ================= STEP 7: INITIAL PARAMETERS =================
m = 0
b = 0
lr = 0.1
epochs = 1000
n = len(df)

X = df["House_size"].values
Y = df["price"].values

# ================= STEP 8: FEATURE SCALING (MIN–MAX) =================
X_min, X_max = X.min(), X.max()
Y_min, Y_max = Y.min(), Y.max()

X_scaled = (X - X_min) / (X_max - X_min)
Y_scaled = (Y - Y_min) / (Y_max - Y_min)

# Convert to feature vectors
X_scaled = X_scaled.reshape(-1, 1)
Y_scaled = Y_scaled.reshape(-1, 1)

# ================= STEP 9–12: GRADIENT DESCENT =================
for epoch in range(epochs):
    Y_pred = m * X_scaled + b
    error = Y_pred - Y_scaled
    
    MSE = np.mean(error ** 2)

    dm = (2 / n) * np.sum(X_scaled * error)
    db = (2 / n) * np.sum(error)

    m = m - lr * dm
    b = b - lr * db

# ================= STEP 13: FINAL PREDICTION =================
Y_pred_final = m * X_scaled + b
final_error = Y_pred_final - Y_scaled
final_MSE = np.mean(final_error ** 2)

# ================= STEP 14: RESULTS =================
df["Y_pred_scaled"] = Y_pred_final.flatten()
df["Final_Error"] = final_error.flatten()

print(f"Final m: {m}, Final b: {b}")
print(f"Final MSE: {final_MSE}")
print("=" * 80)
print("Final DataFrame:\n", df)
