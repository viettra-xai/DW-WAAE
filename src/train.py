import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.model import DW_WAAE
from src.utils import load_data
from src.data_preprocessing import preprocess_data, split_and_sample_data, normalize_data
import pickle

# Configuration
DATA_PATH = "data/raw"
FILENAME = "winter_data.csv"
INPUT_DIM = 106  # Adjust based on dataset
LATENT_DIM = 20
BATCH_SIZE = 128
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10.0
EPOCHS = 50
PATIENCE = 5
DEFAULT_ABNORMAL_SAMPLE_RATIO = 0.2  # Default ratio of abnormal samples to keep
DEFAULT_NORM_TYPE = "minmax"  # Default normalization type


def train_model(data_path, filename, input_dim, latent_dim, batch_size, critic_iterations, lambda_gp, epochs, patience, abnormal_sample_ratio, norm_type):
    """Load data, preprocess it, train the model, and evaluate performance."""
    
    # Load and preprocess data
    print(f"Loading dataset: {filename} from {data_path}...")
    X, y = preprocess_data(data_path, filename)

    # Split and sample data dynamically
    print(f"Splitting data with abnormal sample ratio: {abnormal_sample_ratio}...")
    X_train, y_train, X_test, y_test = split_and_sample_data(X, y, abnormal_sample_ratio)

    # Normalize data with chosen method
    print(f"Normalizing data using {norm_type} scaler...")
    X_train, X_test = normalize_data(X_train, X_test, norm_type)

    # Save X_test and y_test for inference
    test_data_path = "data/processed/X_test.csv"
    test_labels_path = "data/processed/y_test.csv"

    pd.DataFrame(X_test).to_csv(test_data_path, index=False)
    pd.DataFrame(y_test, columns=["label"]).to_csv(test_labels_path, index=False)
    
    print(f"Saved test set to {test_data_path} and labels to {test_labels_path}")

    # Instantiate and train the DW_WAAE model
    print(f"Initializing and training DW_WAAE model with input_dim={input_dim}, latent_dim={latent_dim}...")
    dw_waae = DW_WAAE(input_dim=input_dim, latent_dim=latent_dim, batch_size=batch_size, critic_iterations=critic_iterations, lambda_gp=lambda_gp)
    dw_waae.train(X_train, X_test, y_test, initial_epochs=0, epochs=epochs, patience=patience)
    print("Training complete.")
    # Save the entire DW_WAAE class instance
    dw_waae_save_path = "models/dw_waae.pkl"
    with open(dw_waae_save_path, "wb") as f:
        pickle.dump(dw_waae, f)
    
    print(f"Saved DW_WAAE class instance to {dw_waae_save_path}")

    return dw_waae

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the DW_WAAE model with custom parameters.")

    parser.add_argument("--data_path", type=str, default=DATA_PATH, help="Path to dataset folder")
    parser.add_argument("--filename", type=str, required=True, help="Dataset filename (CSV)")
    parser.add_argument("--input_dim", type=int, required=True, help="Input dimension of data")
    parser.add_argument("--latent_dim", type=int, default=LATENT_DIM, help="Latent dimension")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--critic_iterations", type=int, default=CRITIC_ITERATIONS, help="Critic iterations")
    parser.add_argument("--lambda_gp", type=float, default=LAMBDA_GP, help="Gradient penalty coefficient")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs")
    parser.add_argument("--patience", type=int, default=PATIENCE, help="Early stopping patience")
    parser.add_argument("--abnormal_sample_ratio", type=float, default=DEFAULT_ABNORMAL_SAMPLE_RATIO, help="Ratio of abnormal samples to include in training")
    parser.add_argument("--norm_type", type=str, default=DEFAULT_NORM_TYPE, choices=["minmax", "standard", "robust"], help="Type of normalization (minmax, standard, robust)")

    args = parser.parse_args()

    train_model(
        data_path=args.data_path,
        filename=args.filename,
        input_dim=args.input_dim,
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        critic_iterations=args.critic_iterations,
        lambda_gp=args.lambda_gp,
        epochs=args.epochs,
        patience=args.patience,
        abnormal_sample_ratio=args.abnormal_sample_ratio,
        norm_type=args.norm_type
    )
