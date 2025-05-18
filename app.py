import json
import pickle

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras import Model
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.decomposition import PCA

# ───────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ───────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Autoencoder Anomaly Detection",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ───────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ───────────────────────────────────────────────────────────────────────
def remove_correlated_features(df: pd.DataFrame, removed: list) -> pd.DataFrame:
    """
    Drop features that were removed during training due to high correlation.
    """
    return df.drop(columns=removed, errors="ignore")


def preprocess_test_data(
    df: pd.DataFrame,
    encoded_cols: list,
    categories: dict
) -> pd.DataFrame:
    """
    One-hot encode test data with fixed categories, add any missing columns,
    and return DataFrame with exactly the encoded_cols in the right order.
    """
    # Set categorical dtype so pd.get_dummies creates all expected columns
    for col, vals in categories.items():
        df[col] = pd.Categorical(df[col], categories=vals)

    df_encoded = pd.get_dummies(df, columns=list(categories.keys()))

    # Ensure all trained dummy columns are present
    for col in encoded_cols:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    # Return in exactly the training order
    return df_encoded[encoded_cols]


@st.cache_resource
def load_artifacts():
    """
    Load and cache model artifacts:
      - autoencoder model
      - MinMaxScaler
      - anomaly threshold
      - feature list
      - list of one-hot encoded column names
      - original category mappings
      - list of removed (correlated) features
    """
    # Load trained autoencoder (architecture + weights)
    model = load_model("autoencoder.h5", compile=False)
    model.compile(optimizer="adam", loss="mse")

    # Load preprocessor and metadata
    scaler = pickle.load(open("scaler.pkl", "rb"))
    threshold = pickle.load(open("threshold.pkl", "rb"))
    features = pickle.load(open("features.pkl", "rb"))
    encoded_columns = pickle.load(open("encoded_columns.pkl", "rb"))
    categories = pickle.load(open("categories.pkl", "rb"))

    # Load list of correlated features removed during training
    with open("removed_features.json", "r") as f:
        removed_features = json.load(f)["removed_features"]

    return (
        model,
        scaler,
        threshold,
        features,
        encoded_columns,
        categories,
        removed_features,
    )


# ───────────────────────────────────────────────────────────────────────
# LOAD ARTIFACTS
# ───────────────────────────────────────────────────────────────────────
model, scaler, base_threshold, features, encoded_columns, categories, removed_features = load_artifacts()

# ───────────────────────────────────────────────────────────────────────
# APPLICATION TITLE & DESCRIPTION
# ───────────────────────────────────────────────────────────────────────
st.title("🔒 Autoencoder Anomaly Detection Dashboard")
st.write("Upload the UNSW-NB15 test set to detect anomalies using the trained autoencoder.")

# ───────────────────────────────────────────────────────────────────────
# DATA UPLOAD
# ───────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Choose UNSW-NB15 CSV (test set)", type="csv")
if not uploaded_file:
    st.info("👇 Please upload the `UNSW_NB15_testing-set.csv` file to begin.")
    st.stop()

# Read raw data
df_raw = pd.read_csv(uploaded_file)

# Optionally inspect raw data
with st.expander("Show Raw Test Data"):
    st.dataframe(df_raw)

# ───────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ───────────────────────────────────────────────────────────────────────
# 1) Remove correlated features
df_cleaned = remove_correlated_features(df_raw, removed_features)

# 2) One-hot encode and align to training columns
df_encoded = preprocess_test_data(df_cleaned, encoded_columns, categories)

# 3) Extract true labels for evaluation
y_true = df_raw["label"].values

# 4) Scale features using the saved scaler
X = scaler.transform(df_encoded[features].values)

# ───────────────────────────────────────────────────────────────────────
# THRESHOLD TUNING
# ───────────────────────────────────────────────────────────────────────
mult = st.sidebar.slider("Threshold multiplier", 1.0, 5.0, 3.0, step=0.1)
threshold = base_threshold * mult
st.sidebar.write("▶️ Current threshold:", f"{threshold:.5f}")

# ───────────────────────────────────────────────────────────────────────
# MODEL INFERENCE
# ───────────────────────────────────────────────────────────────────────
# Reconstruct inputs and compute per-sample MSE
X_rec = model.predict(X)
mse = np.mean((X - X_rec) ** 2, axis=1)

# Predict anomalies (1) vs. normal (0)
y_pred = (mse > threshold).astype(int)

# ───────────────────────────────────────────────────────────────────────
# VISUALIZATIONS & METRICS
# ───────────────────────────────────────────────────────────────────────
# 1️⃣ Performance Metrics
st.subheader("1️⃣ Performance Metrics")
auc_score = roc_auc_score(y_true, mse)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Precision", f"{precision_score(y_true, y_pred):.3f}")
col2.metric("Recall", f"{recall_score(y_true, y_pred):.3f}")
col3.metric("F1 Score", f"{f1_score(y_true, y_pred):.3f}")
col4.metric("ROC AUC", f"{auc_score:.3f}")
# 2️⃣ Reconstruction Error Distribution by Class
st.subheader("2️⃣ Reconstruction Error Distribution by Class")
fig1, ax1 = plt.subplots()
ax1.hist(mse[y_true == 0], bins=50, alpha=0.6, label="Normal")
ax1.hist(mse[y_true == 1], bins=50, alpha=0.6, label="Attack")
ax1.axvline(threshold, color="r", linestyle="--", label="Threshold")
ax1.set_xlabel("Reconstruction MSE")
ax1.set_ylabel("Count")
ax1.legend()
st.pyplot(fig1)

# 3️⃣ Confusion Matrix
st.subheader("3️⃣ Confusion Matrix")
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Attack"])
fig2, ax2 = plt.subplots()
disp.plot(ax=ax2, cmap="Blues", colorbar=False)
ax2.set_title("Actual vs. Predicted")
st.pyplot(fig2)

# 4️⃣ Precision–Recall Curve
st.subheader("4️⃣ Precision–Recall Curve")
precision, recall, _ = precision_recall_curve(y_true, mse)
fig3, ax3 = plt.subplots()
ax3.plot(recall, precision, lw=2)
ax3.set_xlabel("Recall")
ax3.set_ylabel("Precision")
ax3.set_title("Precision–Recall")
st.pyplot(fig3)

# 5️⃣ ROC Curve
st.subheader("5️⃣ ROC Curve")
fpr, tpr, _ = roc_curve(y_true, mse)
auc_score = roc_auc_score(y_true, mse)
fig4, ax4 = plt.subplots()
ax4.plot(fpr, tpr, lw=2, label=f"AUC = {auc_score:.2f}")
ax4.plot([0, 1], [0, 1], "--", lw=1)
ax4.set_xlabel("False Positive Rate")
ax4.set_ylabel("True Positive Rate")
ax4.set_title("ROC Curve")
ax4.legend()
st.pyplot(fig4)

# 6️⃣ Top-N Flagged Anomalies
st.subheader("6️⃣ Top 20 Flagged Anomalies")
results = df_raw.copy()
results["mse"] = mse
results["pred"] = y_pred
anoms = results[results["pred"] == 1].sort_values("mse", ascending=False)
st.dataframe(anoms.head(20), use_container_width=True)

# Download anomalies as CSV
csv_data = anoms.to_csv(index=False).encode()
st.download_button(
    label="📥 Download Flagged Anomalies",
    data=csv_data,
    file_name="flagged_anomalies.csv",
    mime="text/csv",
)

# 🔬 Optional: Latent-Space PCA
if st.checkbox("🔬 Show Latent-Space PCA (Advanced)"):
    encoder = Model(model.input, model.layers[-2].output)
    latents = encoder.predict(X)
    proj = PCA(2).fit_transform(latents)

    fig5, ax5 = plt.subplots()
    sc = ax5.scatter(proj[:, 0], proj[:, 1], c=y_pred, cmap="coolwarm", alpha=0.6)
    ax5.set_title("PCA of Latent Codes (red=anomaly)")
    st.pyplot(fig5)
