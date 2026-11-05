# Artifacts
import numpy as np
from pathlib import Path

from src.preprocessing import (
    load_data,
    split_features_target,
    inspect_data,
    check_missing_values,
    check_duplicates,
    check_data_types,
    train_test_split,
    feature_scalig,
    scale_target
)

from src.model import (predict,
                       initialize_parameters,
                       mean_squared_error,
                       train,
                       evaluate_model)

df = load_data()
X,y = split_features_target(df)

if df is not None:

    inspect_data(df)

    missing = check_missing_values(df)
    duplicates = check_duplicates(df)
    data_types = check_data_types(df)
    X, y = split_features_target(df)

    print("\nFeatures Shape:", X.shape)
    print("Target Shape:", y.shape)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)


# Feature scaling
X_train_scaled,X_test_scaled,mean,std = feature_scalig(X_train,X_test)

print("X train scaled shape :",X_train_scaled.shape)
print("X test scaled shape  :",X_test_scaled.shape)


# Test the parameters
w , b = initialize_parameters(5)
print(f"W :{w} and b:{b}")

# Test the y_scaled
y_train_scaled, y_test_scaled, y_mean, y_std = scale_target(y_train, y_test)

print(f"Shape of Y train scaled :{y_train_scaled} | Shape of Y test Scaled {y_test_scaled}")


# Convert data into NumPy arrays
X_train_np = X_train_scaled.to_numpy()
X_test_np = X_test_scaled.to_numpy()

y_train_np = y_train_scaled.to_numpy()

# Train the model
w, b, loss_history = train(X_train_np, y_train_np)

print("Trained weights:", w)
print("Trained bias:", b)

# Predict
y_pred_scaled = predict(X_test_np, w, b)

# Convert predictions back to actual house prices
y_pred = y_pred_scaled * y_std + y_mean

# Evaluate
metrics = evaluate_model(y_test.to_numpy(), y_pred)

print("\nModel Evaluation:")

for name, value in metrics.items():
    print(f"{name}: {value:,.2f}")

# Visualising loss history
import matplotlib.pyplot as plt

plt.plot(loss_history)
plt.xlabel("Epochs")
plt.ylabel("Training MSE (scaled target)")
plt.title("Linear Regression - Learning Curve")
plt.show()

# Saving the model
# Save trained model and preprocessing parameters
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

np.savez(
    MODEL_DIR / "housing_linear_regression.npz",
    weights=w,
    bias=b,
    feature_mean=mean,
    feature_std=std,
    target_mean=y_mean,
    target_std=y_std,
    feature_names=np.array(X.columns)
)

print("\nModel saved successfully!")
