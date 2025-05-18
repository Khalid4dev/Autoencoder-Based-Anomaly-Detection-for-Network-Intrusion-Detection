#!/usr/bin/env python3
# train_autoencoder.py

"""
Train an autoencoder on UNSW-NB15 data, removing
highly correlated features, and save all artifacts.
"""

import json
import pickle

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense


# ───────────────────────────────────────────────────────────────────────
def remove_correlated_features(df: pd.DataFrame,
                               threshold: float = 0.95
                              ) -> tuple[pd.DataFrame, list]:
    """
    Drop numeric features whose pairwise absolute correlation 
    exceeds `threshold`.
    Returns (cleaned_df, dropped_columns).
    """
    numeric = df.select_dtypes(include=['number']).columns
    corr = df[numeric].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    dropped = [col for col in upper.columns if any(upper[col] > threshold)]
    return df.drop(columns=dropped), dropped


# ───────────────────────────────────────────────────────────────────────
def main():
    # 1) Load raw CSVs
    train_df = pd.read_csv('UNSW_NB15_training-set.csv')
    test_df  = pd.read_csv('UNSW_NB15_testing-set.csv')

    # 2) Remove correlated features on BOTH
    train_clean, removed_feats = remove_correlated_features(train_df, threshold=0.95)
    test_clean  = test_df.drop(columns=removed_feats, errors='ignore')

    # Persist the list of removed features for inference
    with open('removed_features.json', 'w') as f:
        json.dump({'removed_features': removed_feats}, f)

    # 3) Build category mapping from train+test for one-hot encoding
    categorical_cols = ['proto', 'service', 'state']
    categories = {}
    for col in categorical_cols:
        cats = np.union1d(train_clean[col].unique(), test_clean[col].unique())
        categories[col] = list(cats)

    # 4) Concatenate and one-hot encode
    full = pd.concat([train_clean, test_clean], axis=0)
    full = pd.get_dummies(full, columns=categorical_cols)

    # Save the list of all dummy columns
    encoded_columns = full.columns.tolist()
    with open('encoded_columns.pkl', 'wb') as f:
        pickle.dump(encoded_columns, f)

    # 5) Scale numeric features to [0,1]
    features = [c for c in full.columns if c not in ['label', 'attack_cat']]
    scaler = MinMaxScaler()
    full[features] = scaler.fit_transform(full[features])

    # Persist scaler, feature list, and category mapping
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('features.pkl', 'wb') as f:
        pickle.dump(features, f)
    with open('categories.pkl', 'wb') as f:
        pickle.dump(categories, f)

    # 6) Split back into train / test
    train_proc = full.iloc[: len(train_clean)]
    X_train, y_train = train_proc[features].values, train_proc['label'].values

    # 7) Prepare autoencoder data (only normal samples)
    X_norm = X_train[y_train == 0]
    X_tr, X_val = train_test_split(X_norm, test_size=0.2, random_state=42)

    # 8) Build autoencoder architecture
    input_dim = X_tr.shape[1]
    inp = Input(shape=(input_dim,))
    enc = Dense(128, activation='relu')(inp)
    enc = Dense(32, activation='relu')(enc)
    dec = Dense(128, activation='relu')(enc)
    out = Dense(input_dim, activation='sigmoid')(dec)

    autoencoder = Model(inputs=inp, outputs=out)
    autoencoder.compile(optimizer='adam', loss='mse')

    # 9) Train
    autoencoder.fit(
        X_tr, X_tr,
        epochs=50,
        batch_size=256,
        shuffle=True,
        validation_data=(X_val, X_val)
    )

    # 10) Compute anomaly threshold = mean + 3·std on validation errors
    X_val_pred = autoencoder.predict(X_val)
    mse_val     = np.mean((X_val - X_val_pred) ** 2, axis=1)
    threshold   = mse_val.mean() + 3 * mse_val.std()

    # Persist model and threshold
    autoencoder.save('autoencoder.h5')
    with open('threshold.pkl', 'wb') as f:
        pickle.dump(threshold, f)

    print("✅ Training complete. Artifacts saved:")
    print("   - autoencoder.h5")
    print("   - scaler.pkl, features.pkl, categories.pkl")
    print("   - encoded_columns.pkl")
    print("   - removed_features.json")
    print("   - threshold.pkl")


if __name__ == '__main__':
    main()
