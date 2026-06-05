"""
Complete Data Preprocessing Pipeline for Machine Learning
10-step data preprocessing and feature engineering pipeline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ======================================================================
# STEP 1: Data Creation
# ======================================================================
print("=" * 100)
print("STEP 1: CREATING INITIAL DATASET")
print("=" * 100)

data = {
    "house_size": [500, 600, 700, 800, 900, 1300, 5000, 6700],
    "price": [5, 6, 7, 8, 9, 13, 50, 67],
    "colour": ["green", "red", "yellow", "white", "black", "blue", "orange", "red"],
    "city": ["delhi", "banglore", "hyderabad", "noida", "pune", "kolkata", "mysore", "chennai"]
}

df = pd.DataFrame(data)
print("Initial DataFrame:")
print(df)
print("\nDataset Information:")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Total Records: {len(df)}")

# ======================================================================
# STEP 2: Initial Visualization
# ======================================================================
print("\n" + "=" * 100)
print("STEP 2: INITIAL DATA VISUALIZATION")
print("=" * 100)

plt.figure(figsize=(10, 6))
plt.scatter(df["house_size"], df["price"], c='blue', alpha=0.7, s=100)
plt.title("House Price vs Size - Raw Data", fontsize=14, fontweight='bold')
plt.xlabel("House Size (sq ft)", fontsize=12)
plt.ylabel("Price ($1000)", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ======================================================================
# STEP 3: Data Exploration
# ======================================================================
print("\n" + "=" * 100)
print("STEP 3: DATA EXPLORATION AND ANALYSIS")
print("=" * 100)

print("BASIC STATISTICS:")
print(df.describe())

print("\nDATA TYPES:")
print(df.dtypes)

print("\nMISSING VALUES:")
print(df.isnull().sum())

print("\nUNIQUE VALUES:")
print(f"Unique colours: {df['colour'].nunique()}")
print(f"Unique cities: {df['city'].nunique()}")

# ======================================================================
# STEP 4: Min-Max Normalization
# ======================================================================
print("\n" + "=" * 100)
print("STEP 4: MIN-MAX NORMALIZATION (Linear Scaling)")
print("=" * 100)

X = df["house_size"].values
x_max = X.max()
x_min = X.min()
df["size_minmax"] = (X - x_min) / (x_max - x_min)

print("After Min-Max Scaling:")
print(df[["house_size", "size_minmax"]])

plt.figure(figsize=(10, 6))
plt.scatter(df["size_minmax"], df["price"], c='green', alpha=0.7, s=100)
plt.title("Price vs Min-Max Scaled Size", fontsize=14, fontweight='bold')
plt.xlabel("Scaled House Size (0-1)", fontsize=12)
plt.ylabel("Price ($1000)", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ======================================================================
# STEP 5: Z-Score Standardization
# ======================================================================
print("\n" + "=" * 100)
print("STEP 5: Z-SCORE STANDARDIZATION")
print("=" * 100)

mean = X.mean()
std = X.std()
df["size_zscore"] = (X - mean) / std

print("After Z-Score Standardization:")
print(df[["house_size", "size_zscore"]])
print(f"\nMean of z-scores: {df['size_zscore'].mean():.6f}")
print(f"Std of z-scores: {df['size_zscore'].std():.6f}")

plt.figure(figsize=(10, 6))
plt.scatter(df["size_zscore"], df["price"], c='red', alpha=0.7, s=100)
plt.title("Price vs Z-Score Standardized Size", fontsize=14, fontweight='bold')
plt.xlabel("Standardized House Size", fontsize=12)
plt.ylabel("Price ($1000)", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ======================================================================
# STEP 6: Log Transformation
# ======================================================================
print("\n" + "=" * 100)
print("STEP 6: LOG TRANSFORMATION")
print("=" * 100)

df["size_log"] = np.log1p(X)

print("After Log Transformation:")
print(df[["house_size", "size_log"]])

plt.figure(figsize=(10, 6))
plt.scatter(df["size_log"], df["price"], c='purple', alpha=0.7, s=100)
plt.title("Price vs Log-Transformed Size", fontsize=14, fontweight='bold')
plt.xlabel("Log(House Size)", fontsize=12)
plt.ylabel("Price ($1000)", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ======================================================================
# STEP 7: Binning (Bucketing)
# ======================================================================
print("\n" + "=" * 100)
print("STEP 7: BINNING HOUSE SIZES INTO CATEGORIES")
print("=" * 100)

bins = [0, 1000, 5000, 7000]
labels = ["small_house", "medium_house", "large_house"]
df["house_bins"] = pd.cut(df["house_size"], bins=bins, labels=labels)

print("House Sizes with Bins:")
print(df[["house_size", "house_bins"]])
print(f"\nBin Counts:")
print(df["house_bins"].value_counts())

# ======================================================================
# STEP 8: Average Price by Bin
# ======================================================================
print("\n" + "=" * 100)
print("STEP 8: AVERAGE PRICE FOR EACH BIN")
print("=" * 100)

df["mean_price_by_bin"] = df.groupby("house_bins")["price"].transform("mean")
print("Average Price by House Size Bin:")
print(df[["house_bins", "price", "mean_price_by_bin"]].sort_values("house_bins"))

# ======================================================================
# STEP 9: Convert Bins to Feature Vectors (One-Hot Encoding)
# ======================================================================
print("\n" + "=" * 100)
print("STEP 9: CONVERTING CATEGORICAL DATA TO FEATURE VECTORS")
print("=" * 100)

# One-hot encode house bins
one_hot_bins = pd.get_dummies(df["house_bins"], prefix="bin")
print("One-hot encoded house bins:")
print(one_hot_bins)

# One-hot encode colour
one_hot_colour = pd.get_dummies(df["colour"], prefix="color")
print("\nOne-hot encoded colours:")
print(one_hot_colour)

# One-hot encode city
one_hot_city = pd.get_dummies(df["city"], prefix="city")
print("\nOne-hot encoded cities:")
print(one_hot_city)

# Combine all one-hot encoded columns
df = pd.concat([df, one_hot_bins, one_hot_colour, one_hot_city], axis=1)

print("\nDataFrame after all one-hot encoding:")
print(df.head())

# ======================================================================
# STEP 10: Prepare ML-Ready Dataset
# ======================================================================
print("\n" + "=" * 100)
print("STEP 10: PREPARING FINAL ML-READY DATASET")
print("=" * 100)

print("FINAL DATASET INFORMATION:")
print(f"Shape: {df.shape}")
print(f"Columns ({len(df.columns)} total):")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2}. {col}")

print("\nSAMPLE OF FINAL DATASET:")
print(df.head())

print("\n" + "=" * 100)
print("PREPROCESSING PIPELINE COMPLETE!")

print("=" * 100)
