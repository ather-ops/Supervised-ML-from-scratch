# importing the library
import pandas as pd
import numpy as np
from pathlib import Path

# Finding project directory
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "Housing.csv"


def load_data():
    try:
        df = pd.read_csv(DATA_PATH)
        return df

    except FileNotFoundError:
        print("File not found. Check the CSV path.")

    except Exception as e:
        print("Error while loading file:", e)

    return None

# Selecting features and target value
FEATURES = [
    "area",
    "bedrooms",
    "bathrooms",
    "stories",
    "parking"
]

TARGET = "price"

def split_features_target(df):
    X = df[FEATURES]
    y = df[TARGET]
    return X,y

# Data inspect function
def inspect_data(df):
    print("Dataset Shape:")
    print(df.shape)

    print("\nFirst 5 Rows:")
    print(df.head())

    print("\nDataset Information:")
    df.info()

    print("\nStatistical Summary:")
    print(df.describe())

    print("\nColumn Names:")
    print(df.columns.tolist())

# Checking missing value function
def check_missing_values(df):

    missing = df.isnull().sum()

    print("\nMissing Values:")
    print(missing)

    print("\nTotal Missing Values:", missing.sum())

    return missing

# Duplicate value function
def check_duplicates(df):
    duplicate_values = df.duplicated().sum()
    print("Duplicate value :",duplicate_values)
    return duplicate_values

# Checking the Datatype
def check_data_types(df):
    data_type = df.dtypes
    print("Data Type :",data_type)
    return data_type

# Train test split

def train_test_split(X, y, test_size=0.2, random_state=42):

    n = len(X)

    rng = np.random.default_rng(random_state)
    indices = rng.permutation(n)

    n_test = int(n * test_size)

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]

    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    return X_train, X_test, y_train, y_test

# Feature scaling
def feature_scalig(x_train,x_test):
    mean = np.mean(x_train,axis = 0)
    std = np.std(x_train,axis = 0)

    # handling zero std values
    std[std == 0] = 1

    x_train_scaled = (x_train - mean)/std
    x_test_scaled = (x_test - mean)/std

    return x_train_scaled,x_test_scaled,mean,std

# Scaling target values

def scale_target(y_train, y_test):

    mean = np.mean(y_train)
    std = np.std(y_train)

    if std == 0:
        std = 1.0

    y_train_scaled = (y_train-mean) / std
    y_test_scaled = (y_test - mean) / std

    return y_train_scaled, y_test_scaled, mean, std
