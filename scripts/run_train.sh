#!/bin/bash

# Script to execute train.py with command-line arguments.

# Default values
DATA_PATH="data/raw"
FILENAME="winter_data.csv"
INPUT_DIM=106
LATENT_DIM=20
BATCH_SIZE=128
CRITIC_ITERATIONS=5
LAMBDA_GP=10.0
EPOCHS=50
PATIENCE=5
ABNORMAL_SAMPLE_RATIO=0.2
NORM_TYPE="minmax"

# Allow overriding parameters from command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --data_path) DATA_PATH="$2"; shift ;;
        --filename) FILENAME="$2"; shift ;;
        --input_dim) INPUT_DIM="$2"; shift ;;
        --latent_dim) LATENT_DIM="$2"; shift ;;
        --batch_size) BATCH_SIZE="$2"; shift ;;
        --critic_iterations) CRITIC_ITERATIONS="$2"; shift ;;
        --lambda_gp) LAMBDA_GP="$2"; shift ;;
        --epochs) EPOCHS="$2"; shift ;;
        --patience) PATIENCE="$2"; shift ;;
        --abnormal_sample_ratio) ABNORMAL_SAMPLE_RATIO="$2"; shift ;;
        --norm_type) NORM_TYPE="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Run train.py with the specified arguments
python -m src/train.py \
    --data_path "$DATA_PATH" \
    --filename "$FILENAME" \
    --input_dim $INPUT_DIM \
    --latent_dim $LATENT_DIM \
    --batch_size $BATCH_SIZE \
    --critic_iterations $CRITIC_ITERATIONS \
    --lambda_gp $LAMBDA_GP \
    --epochs $EPOCHS \
    --patience $PATIENCE \
    --abnormal_sample_ratio $ABNORMAL_SAMPLE_RATIO \
    --norm_type $NORM_TYPE
