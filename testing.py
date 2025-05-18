#!/usr/bin/env python3
# evaluate_autoencoder.py

"""
Load artifacts, preprocess the UNSW-NB15 test set (dropping the same
correlated features), run the trained autoencoder, and display metrics
and a confusion matrix.
"""

import json
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix


# ───────────────────────────────────────────────────────────────────────
def remove_correlated_features(df: pd.DataFrame, removed: list) -> pd.DataFrame:
    """
    Drop the columns in `removed` from df (if present).
    """
    return df.drop(columns=removed, errors='ignore')


def preprocess_test_data(df: pd.DataFrame,
                         encoded_columns: list,
                         categories: dict
                        ) -> pd.DataFrame:
    """
    One-hot encode each categorical column using the saved category lists,
    then add any missing dummy columns, and reorder to encoded_columns.
    """
    for col, cats in categories.items():
        df[col] = pd.Categorical(df[col], categories=cats)

    df_enc = pd.get_dummies(df, columns=list(categories.keys()))
    for col in encoded_columns:
        if col not in df_enc.columns:
            df_enc[col] = 0

    return df_enc[encoded_columns]


# ───────────────────────────────────────────────────────────────────────
def main():
    # 1) Load artifacts
    model     = load_model('autoencoder.h5', compile=False)
    model.compile(optimizer='adam', loss='mse')

    scaler        = pickle.load(open('scaler.pkl','rb'))
    features      = pickle.load(open('features.pkl','rb'))
    encoded_cols  = pickle.load(open('encoded_columns.pkl','rb'))
    categories    = pickle.load(open('categories.pkl','rb'))
    threshold     = pickle.load(open('threshold.pkl','rb'))

    removed_feats = json.load(open('removed_features.json'))['removed_features']

    # 2) Load raw test set
    df_raw = pd.read_csv('UNSW_NB15_testing-set.csv')

    # 3) Remove the same correlated features
    df_clean = remove_correlated_features(df_raw, removed_feats)

    # 4) Preprocess (one-hot + align)
    df_enc = preprocess_test_data(df_clean, encoded_cols, categories)

    # 5) Extract labels and scale
    y_test = df_clean['label'].values
    X_test = scaler.transform(df_enc[features].values)

    # 6) Inference
    X_rec   = model.predict(X_test)
    mse_test = np.mean((X_test - X_rec) ** 2, axis=1)
    y_pred  = (mse_test > threshold).astype(int)

    # 7) Print classification results
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['normal','attack']))

    # 8) Confusion matrix (print + plot)
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:\n", cm)

    plt.figure(figsize=(6,5))
    plt.imshow(cm, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xticks([0,1], ['Normal','Attack'])
    plt.yticks([0,1], ['Normal','Attack'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i,j], ha='center', va='center', color='white')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
