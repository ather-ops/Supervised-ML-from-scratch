from src.preprocessing import (
    load_data,
    split_features_target,
    inspect_data,
    check_missing_values,
    check_duplicates,
    check_data_types,
    train_test_split,
    feature_scalig
)
from src.model import initialize_parameters

df = load_data()
X,y =split_features_target(df)

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


# Tets the parameters
w , b = initialize_parameters(5)
print(f"W :{w} and b:{b}")
