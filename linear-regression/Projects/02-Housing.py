"""
House Price Prediction - From Scratch
Using only NumPy, Pandas, Matplotlib
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("="*60)
print("HOUSE PRICE PREDICTION PROJECT")
print("="*60)

# STEP 1: LOAD DATA
df = pd.read_csv("Housing.csv")
print("\nData loaded successfully")
print("Total rows:", len(df))
print("\nFirst 5 rows:")
print(df.head())

# STEP 2: DROP UNWANTED COLUMNS
dropping_columns = ["mainroad", "guestroom", "basement", "hotwaterheating"]
df.drop(columns=dropping_columns, inplace=True)
print("\nUnwanted columns dropped")
print("Remaining columns:", df.columns.tolist())

# STEP 3: CONVERT CATEGORICAL TO NUMBERS
df['airconditioning'] = df['airconditioning'].map({'yes': 1, 'no': 0})
df['prefarea'] = df['prefarea'].map({'yes': 1, 'no': 0})
df['furnishingstatus'] = df['furnishingstatus'].map({'furnished': 2, 'semi-furnished': 1, 'unfurnished': 0})

print("\nCategorical columns converted to numbers")
print(df.head())

# STEP 4: SEPARATE FEATURES AND TARGET
X = df.drop('price', axis=1).values
y = df['price'].values

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

# STEP 5: NORMALIZE THE DATA
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_normalized = (X - X_mean) / X_std

print("\nData normalized")

# STEP 6: SPLIT INTO TRAIN AND TEST
np.random.seed(42)
indices = np.random.permutation(len(X_normalized))
split = int(0.8 * len(X_normalized))

train_idx = indices[:split]
test_idx = indices[split:]

X_train = X_normalized[train_idx]
y_train = y[train_idx]
X_test = X_normalized[test_idx]
y_test = y[test_idx]

print("\nTrain samples:", len(X_train))
print("Test samples:", len(X_test))

# STEP 7: INITIALIZE PARAMETERS
n_features = X_train.shape[1]
weights = np.zeros(n_features)
bias = 0
learning_rate = 0.01
epochs = 500

print("\nParameters initialized")
print("Learning rate:", learning_rate)
print("Epochs:", epochs)

# STEP 8: TRAIN THE MODEL
loss_history = []

for epoch in range(epochs):
    predictions = np.dot(X_train, weights) + bias
    errors = predictions - y_train
    loss = np.mean(errors ** 2)
    loss_history.append(loss)
    
    gradient_weights = (2/len(X_train)) * np.dot(X_train.T, errors)
    gradient_bias = (2/len(X_train)) * np.sum(errors)
    
    weights = weights - learning_rate * gradient_weights
    bias = bias - learning_rate * gradient_bias
    
    if epoch % 100 == 0:
        print("Epoch", epoch, "Loss:", loss)

print("\nTraining complete")

# STEP 9: EVALUATE ON TEST DATA
y_pred = np.dot(X_test, weights) + bias

mae = np.mean(np.abs(y_pred - y_test))
mse = np.mean((y_pred - y_test) ** 2)
rmse = np.sqrt(mse)

print("\nModel Performance:")
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)

# STEP 10: COMPARE ACTUAL VS PREDICTED
print("\nFirst 10 test predictions:")
for i in range(10):
    print("Actual:", y_test[i], "Predicted:", round(y_pred[i], 2), "Error:", round(y_test[i] - y_pred[i], 2))

# STEP 11: PREDICT NEW HOUSE
new_house = np.array([[7500, 3, 2, 2, 1, 2, 1, 2]])
new_house_normalized = (new_house - X_mean) / X_std
new_prediction = np.dot(new_house_normalized, weights) + bias

print("\nNew House Prediction:")
print("Area: 7500, Bedrooms: 3, Bathrooms: 2, Stories: 2")
print("Airconditioning: Yes, Parking: 2, Prefarea: Yes, Furnishing: Furnished")
print("Predicted Price:", new_prediction[0])

# STEP 12: VISUALIZATION
plt.figure(figsize=(12, 8))

# Plot 1: Loss over time
plt.subplot(2, 2, 1)
plt.plot(loss_history)
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")

# Plot 2: Actual vs Predicted
plt.subplot(2, 2, 2)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted")

# Plot 3: Error distribution
plt.subplot(2, 2, 3)
errors_plot = y_test - y_pred
plt.hist(errors_plot, bins=20)
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.title("Error Distribution")

# Plot 4: Feature weights
plt.subplot(2, 2, 4)
feature_names = df.drop('price', axis=1).columns
plt.barh(feature_names, weights)
plt.xlabel("Weight Value")
plt.title("Feature Importance")

plt.tight_layout()
plt.show()

# STEP 13: SAVE MODEL PARAMETERS
np.save("weights.npy", weights)
np.save("bias.npy", bias)
np.save("X_mean.npy", X_mean)
np.save("X_std.npy", X_std)

print("\nModel saved successfully")
print("Files created: weights.npy, bias.npy, X_mean.npy, X_std.npy")
