import pandas as pd
import numpy as np
import torch
import os

def load_and_preprocess_data(csv_path):
    print(f"Loading data from {csv_path}...")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found at {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise Exception(f"Error reading CSV: {e}")

    # --- Preprocessing ---
    print("Preprocessing data...")
    
    # 1. Select Features and Target
    target_col = 'Survived'
    feature_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    
    # Check if necessary columns exist
    missing_cols = [col for col in feature_cols + [target_col] if col not in df.columns]
    if missing_cols:
         raise ValueError(f"Missing columns in CSV: {missing_cols}")

    # 2. Handle Categorical Data
    if df['Sex'].dtype == object:
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    
    # 3. Handle Missing Values
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df['Fare'] = df['Fare'].fillna(df['Fare'].mean())

    # 4. Extract Data
    x_data = df[feature_cols].values.astype(np.float32)
    y_data = df[target_col].values.astype(np.int64)

    # 5. Normalization (Standard Scaling)
    mean = np.mean(x_data, axis=0)
    std = np.std(x_data, axis=0)
    std[std == 0] = 1.0
    x_data = (x_data - mean) / std

    # 6. Convert to PyTorch Tensors
    x_train = torch.tensor(x_data)
    y_train = torch.tensor(y_data)
    
    return x_train, y_train
