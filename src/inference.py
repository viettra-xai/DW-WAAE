import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

def run_inference(model_path, test_data_path, test_labels_path):
    """Loads a trained model, loads test data, computes R-scores, and evaluates anomaly detection performance."""

    # Load saved test dataset
    print(f"Loading test data from {test_data_path} and labels from {test_labels_path}...")
    X_test = pd.read_csv(test_data_path)
    y_test = pd.read_csv(test_labels_path)["label"].values  # Extract labels

    # Load the full class instance
    with open(model_path, "rb") as f:
        dw_waae = pickle.load(f)

    # Compute R-scores for the test set
    print("Computing R-scores...")
    y_pred = dw_waae.compute_r_score(X_test).numpy()

    # Set anomaly detection threshold (e.g., 50th percentile)
    anomaly_energy_threshold = np.percentile(y_pred, 50)
    print(f"Energy threshold to detect anomaly: {anomaly_energy_threshold:.3f}")

    # Detect anomalies based on the threshold
    y_pred_flag = np.where(y_pred >= anomaly_energy_threshold, 1, 0)

    # Calculate precision, recall, and F1-score
    prec, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred_flag, average="binary")

    # Display results
    print("\n--- Final Inference Results ---")
    print(f" Precision = {prec:.3f}")
    print(f" Recall    = {recall:.3f}")
    print(f" F1-Score  = {fscore:.3f}")

    return prec, recall, fscore

# Example usage
if __name__ == "__main__":
    model_path = "./models/dw_waae.pkl"
    test_data_path = "./data/processed/X_test.csv"
    test_labels_path = "./data/processed/y_test.csv"

    run_inference(model_path, test_data_path, test_labels_path)
