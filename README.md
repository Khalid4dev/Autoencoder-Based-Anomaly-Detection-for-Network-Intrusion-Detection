
# Network Anomaly Detection with Autoencoder

An unsupervised deep learning approach to detect network anomalies in the UNSW-NB15 dataset using an autoencoder neural network.

## Overview

This project implements an autoencoder-based anomaly detection system that:

1. Learns the normal patterns in network traffic data
2. Automatically removes highly correlated features
3. Detects anomalies based on reconstruction error
4. Provides evaluation metrics and visualizations

## Project Structure

- [training.py](training.py) - Main training script
- [testing.py](testing.py) - Evaluation script
- [app.py](app.py) - Interactive dashboard
- Various artifacts saved during training:
  - `autoencoder.h5` - Trained model
  - `scaler.pkl` - Fitted MinMaxScaler
  - `features.pkl` - Feature list
  - `categories.pkl` - Categorical mappings
  - `encoded_columns.pkl` - One-hot encoded columns
  - `threshold.pkl` - Computed anomaly threshold
  - `removed_features.json` - Dropped correlated features

## Requirements

```txt
tensorflow
pandas
numpy
scikit-learn
streamlit
```

## Dataset

Uses the [UNSW-NB15 dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset) for network intrusion detection:

- [UNSW_NB15_training-set.csv](UNSW_NB15_training-set.csv) - Training data
- [UNSW_NB15_testing-set.csv](UNSW_NB15_testing-set.csv) - Test data

## Usage

1. Train the model:

```bash
python training.py
```

2. Evaluate performance:

```bash
python testing.py
```

3. Launch interactive dashboard:

```bash
streamlit run app.py
```

## Model Architecture

The autoencoder consists of:

```
Input Layer (feature_dim) 
    ↓
Dense Layer (128, relu)
    ↓
Dense Layer (32, relu)
    ↓
Dense Layer (128, relu)
    ↓
Output Layer (feature_dim, sigmoid)
```

Trained using:

- Loss: Mean Squared Error
- Optimizer: Adam
- Batch Size: 256
- Epochs: 50

## Features

- Automatic removal of highly correlated features (threshold=0.95)
- MinMax scaling of numeric features
- One-hot encoding of categorical variables
- Training on normal traffic only
- Anomaly detection via reconstruction error threshold
- Interactive visualization dashboard

## Pipeline

1. Load and preprocess data
2. Remove correlated features
3. One-hot encode categorical variables
4. Scale numeric features
5. Train autoencoder on normal samples
6. Compute anomaly threshold
7. Save artifacts for inference

## License

MIT License

## Contributing

Pull requests welcome! Please open an issue first to discuss proposed changes.
