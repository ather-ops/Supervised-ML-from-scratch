"""
Normalization Techniques Implementation from Scratch
This script demonstrates three fundamental data normalization techniques implemented manually.
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# =================================== NORMALIZATION TECHNIQUES FROM SCRATCH ===================================
"""
NORMALIZATION DEFINITIONS:
1. Min-Max Scaling (Linear Scaling): 
   Rescales features to a fixed range [0, 1] by subtracting min and dividing by range.
   Formula: X_scaled = (X - X_min) / (X_max - X_min)

2. Z-Score Standardization:
   Centers data around mean=0 with standard deviation=1.
   Formula: Z = (X - μ) / σ
   Where μ = mean, σ = standard deviation

3. Log Scaling (Log Transformation):
   Applies natural logarithm to handle skewed data and reduce range of values.
   Formula: X_log = log(X)
   Useful for data with exponential growth patterns.
"""

# Step 1: Create sample data
print("=" * 80)
print("STEP 1: CREATING SAMPLE DATA")
print("=" * 80)

data = {
    "house_size": [500, 600, 700, 800, 900, 1300, 5000, 6700],
    "price": [5, 6, 7, 8, 9, 13, 50, 67],
    "colour": ["green", "red", "yellow", "white", "black", "blue", "orange", "red"],
    "city": ["delhi", "banglore", "hyderabad", "noida", "pune", "kolkata", "mysore", "chennai"]
}

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)
print("\nDataset Shape:", df.shape)
print("=" * 80)

# Step 2: Visualize original data
print("\nSTEP 2: VISUALIZING ORIGINAL DATA")
print("=" * 80)

plt.figure(figsize=(15, 10))

# Original data plot
plt.subplot(2, 2, 1)
plt.scatter(df["house_size"], df["price"], color='blue', alpha=0.7)
plt.title("Original Data: House Size vs Price")
plt.xlabel("House Size (sq ft)")
plt.ylabel("Price (in $10,000)")
plt.grid(True, alpha=0.3)

# Step 3: Min-Max Scaling (Linear Scaling) from scratch
print("\nSTEP 3: MIN-MAX SCALING (LINEAR SCALING)")
print("=" * 80)

X = df["house_size"].values
x_max = X.max()
x_min = X.min()
X_scaled = (X - x_min) / (x_max - x_min)

df["min_max_scaled"] = X_scaled
print("Min-Max Scaling Applied:")
print(f"  Min value: {x_min}")
print(f"  Max value: {x_max}")
print(f"  Range: {x_max - x_min}")
print("\nDataFrame with Min-Max Scaled column:")
print(df[["house_size", "min_max_scaled", "price"]])

# Visualization after Min-Max scaling
plt.subplot(2, 2, 2)
plt.scatter(df["min_max_scaled"], df["price"], color='green', alpha=0.7)
plt.title("After Min-Max Scaling\nRange: [0, 1]")
plt.xlabel("Min-Max Scaled House Size")
plt.ylabel("Price (in $10,000)")
plt.grid(True, alpha=0.3)

# Step 4: Z-Score Standardization from scratch
print("\n" + "=" * 80)
print("STEP 4: Z-SCORE STANDARDIZATION")
print("=" * 80)

mean = X.mean()
std = X.std()
Z_score = (X - mean) / std

df["z_score_standardized"] = Z_score
print("Z-Score Standardization Applied:")
print(f"  Mean (μ): {mean:.2f}")
print(f"  Standard Deviation (σ): {std:.2f}")
print("\nDataFrame with Z-Score Standardized column:")
print(df[["house_size", "z_score_standardized", "price"]])

# Visualization after Z-Score standardization
plt.subplot(2, 2, 3)
plt.scatter(df["z_score_standardized"], df["price"], color='red', alpha=0.7)
plt.title("After Z-Score Standardization\nMean: 0, Std: 1")
plt.xlabel("Z-Score Standardized House Size")
plt.ylabel("Price (in $10,000)")
plt.grid(True, alpha=0.3)

# Step 5: Log Scaling from scratch
print("\n" + "=" * 80)
print("STEP 5: LOG SCALING (LOG TRANSFORMATION)")
print("=" * 80)

X_log = np.log(X)
df["log_scaled"] = X_log

print("Log Scaling Applied:")
print(f"  Natural logarithm applied to house sizes")
print("\nDataFrame with Log Scaled column:")
print(df[["house_size", "log_scaled", "price"]])

# Visualization after Log scaling
plt.subplot(2, 2, 4)
plt.scatter(df["log_scaled"], df["price"], color='purple', alpha=0.7)
plt.title("After Log Scaling")
plt.xlabel("Log Scaled House Size (ln(x))")
plt.ylabel("Price (in $10,000)")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Step 6: Summary Statistics
print("\n" + "=" * 80)
print("STEP 6: SUMMARY STATISTICS OF NORMALIZED FEATURES")
print("=" * 80)

summary_df = pd.DataFrame({
    'Original': df['house_size'],
    'Min-Max Scaled': df['min_max_scaled'],
    'Z-Score Standardized': df['z_score_standardized'],
    'Log Scaled': df['log_scaled']
})

print("\nDescriptive Statistics:")
print(summary_df.describe())

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("Successfully implemented 3 normalization techniques from scratch:")
print("1. Min-Max Scaling ✓")
print("2. Z-Score Standardization ✓")
print("3. Log Scaling ✓")
print(f"\nDataset size: {df.shape[0]} samples, {df.shape[1]} features")

print("All transformations applied without using sklearn!")
