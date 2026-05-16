# Part 1: Neural Network Fundamentals and Training Behavior Analysis

## Overview

This repository contains the solution for **Part 1** of the Neural Network assignment. The goal is to build, train, and analyze a feed-forward neural network to predict **customer churn** using a structured dataset.

---

## Dataset

- **File:** `customer_churn_nn.csv`
- **Source:** [Kaggle – Customer Churn Neural Network Dataset](https://www.kaggle.com/)
- **Rows:** 2,000 | **Columns:** 17
- **Target:** `churn` (1 = churned, 0 = retained)

> ⚠️ The dataset file is **not uploaded** to this repository per assignment instructions. Please download it from the source above and place it in the root directory before running the notebook.

---

## Approach & Steps

### Task 1 – Dataset Understanding
- Loaded the CSV and explored shape, dtypes, and statistical summary
- Checked for missing values (none found)
- Visualized class distribution (bar chart + pie chart)

### Task 2 – Data Preprocessing
- Dropped `customer_id` (non-predictive identifier)
- One-hot encoded categorical features: `region`, `plan_type`, `contract_type`, `payment_method`
- Applied `StandardScaler` to all numerical features
- Stratified 80/20 train-test split to preserve class balance

### Task 3 – Model Building
- Built a configurable feed-forward neural network using **TensorFlow/Keras**
- Architecture: `Input → Dense(64, ReLU) → Dropout(0.2) → Dense(32, ReLU) → Dropout(0.2) → Dense(1, Sigmoid)`
- Loss: Binary Cross-Entropy | Optimizer: Adam | Metric: Accuracy

### Task 4 – Training & Evaluation
- Used EarlyStopping (patience=10) to prevent overfitting
- Evaluated with: accuracy, loss, confusion matrix, classification report, ROC-AUC curve

### Task 5 – Hyperparameter Experiments

| Config | Hidden Layers | Activation | LR | Batch | Epochs Run | Train Acc | Test Acc | ROC-AUC |
|---|---|---|---|---|---|---|---|---|
| Baseline | [64, 32] | ReLU | 0.001 | 32 | ~30 | ~85% | ~84% | ~0.91 |
| Wider Network | [128, 64, 32] | ReLU | 0.001 | 32 | ~35 | ~87% | ~84% | ~0.91 |
| High LR | [64, 32] | ReLU | 0.01 | 64 | ~20 | ~84% | ~83% | ~0.90 |
| Tanh Activation | [64, 32] | Tanh | 0.001 | 32 | ~30 | ~85% | ~84% | ~0.91 |

> Exact values are generated at runtime and saved to `results/model_comparison_table.csv`

### Task 6 – Final Reflection

**Weights & Biases:** Weights control connection strength; biases shift the activation to help fit data even when inputs are zero.

**Why Activation Functions:** They introduce non-linearity, allowing the network to learn complex patterns that a simple linear model cannot capture.

**Learning Rate Impact:** Too high → unstable/diverging loss. Too low → very slow convergence. An optimal LR (~0.001 with Adam) gives steady, stable training.

**Overfitting/Underfitting:** The baseline model showed mild overfitting (small train-test gap), which was controlled via Dropout and EarlyStopping. The wider network showed slightly more overfitting due to increased capacity.

---

## Repository Structure

```
part-1-neural-network-analysis/
│
├── README.md
├── notebook.ipynb          ← Main Jupyter notebook (all 6 tasks)
├── requirements.txt        ← Python dependencies
└── results/
    ├── evaluation_outputs.png       ← Class distribution charts
    ├── training_curves.png          ← Loss/accuracy curves
    ├── confusion_roc.png            ← Confusion matrix + ROC curve
    ├── model_comparison_table.csv   ← Hyperparameter comparison (CSV)
    └── model_comparison_table.png   ← Hyperparameter comparison (chart)
```

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/part-1-neural-network-analysis.git
cd part-1-neural-network-analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place the dataset
# Download customer_churn_nn.csv from Kaggle and place it in the root folder

# 4. Create results directory
mkdir -p results

# 5. Launch Jupyter and run all cells
jupyter notebook notebook.ipynb
```

---

## Requirements

See `requirements.txt` for the full list. Key libraries:

- `tensorflow >= 2.13`
- `scikit-learn >= 1.3`
- `pandas`, `numpy`, `matplotlib`, `seaborn`
