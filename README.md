# DW_WAAE: Deep Wasserstein Autoencoder for Anomaly Detection

## 📌 Overview
DW_WAAE is an **autoencoder-based anomaly detection model** that leverages the **Wasserstein loss function** and **gradient penalty (WAAE)** to identify anomalies in datasets.

---

## 📁 Project Structure
```
dw_waae_project/
│── data/                   # Raw and processed datasets
│── models/                 # Saved models and weights
│── scripts/                # Shell scripts for execution
│── src/                    # Source code
│   ├── __init__.py
│   ├── train.py            # Training script
│   ├── inference.py        # Inference script
│   ├── model.py            # DW_WAAE model definition
│   ├── utils.py            # Helper functions
│   ├── data_preprocessing.py # Data loading and preprocessing
│── requirements.txt        # Dependencies
│── run_train.sh            # Shell script to run training
│── README.md               # Project documentation
```

---

## ⚙️ Installation
### 1️⃣ **Clone the Repository**
```bash
git clone https://github.com/your-username/dw_waae_project.git
cd dw_waae_project
```

### 2️⃣ **Create a Virtual Environment (Recommended)**
```bash
python -m venv myenv
source myenv/bin/activate  # On macOS/Linux
myenv\Scripts\activate     # On Windows
```

### 3️⃣ **Install Dependencies**
```bash
pip install -r requirements.txt
```

---

## 🚀 Training the Model
Run the following command to train the model:
```bash
python src/train.py --filename winter_data.csv
```
Or use the script:
```bash
./scripts/run_train.sh
```

---

## 📊 Running Inference
To perform inference:
```bash
python src/inference.py --model_path models/dw_waae.pkl --test_data_path data/processed/X_test.csv --test_labels_path data/processed/y_test.csv
```

---

## ⚡ Key Features
✅ **Deep Autoencoder**: Learns low-dimensional representations  
✅ **Wasserstein Loss**: More stable training for anomaly detection  
✅ **Gradient Penalty (WAAE)**: Prevents overfitting  
✅ **Customizable Training**: Supports different normalization methods  

---

## 🛠️ Configuration Options
Modify the parameters in `train.py`:
```bash
python src/train.py --epochs 100 --batch_size 256 --norm_type standard
```

| **Parameter**  | **Description** |
|---------------|----------------|
| `--filename`  | Name of the dataset file |
| `--epochs`    | Number of training epochs |
| `--batch_size` | Training batch size |
| `--norm_type` | Normalization type (minmax, standard, robust) |

---

## 📝 License
This project is licensed under the **MIT License**.

---

## 👨‍💻 Contributors
- **Your Name** - [GitHub](https://github.com/your-username)

For questions, open an issue or reach out! 🚀

