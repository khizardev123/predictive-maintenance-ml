# Selective Predictive Maintenance Model 

This project implements a robust machine learning pipeline for predicting system failures using high-frequency log telemetry data. The model is specifically optimized for extreme class imbalance (failures occur in <0.08% of cases).

## 🚀 Project Overview

The goal of this project is to create an early-warning system that identifies potential hardware or software failures before they occur. By analyzing CPU metrics, memory usage, and latency, we can proactively maintain systems and reduce downtime.

## 📊 Dataset Specifications

- **File**: `system_logs_ready_min.csv`
- **Total Records**: 876,100
- **Features**:
  - `cpu_usage`: Percentage of CPU utilization.
  - `memory_usage`: Percentage of memory in use.
  - `request_count`: Total requests processed per interval.
  - `latency`: System response time (ms).
  - `error_count`: Number of errors logged in the interval.
- **Target**: `failure` (0 = Normal, 1 = System Failure)

## 🛠️ Implementation Strategy

### 1. Handling Extreme Imbalance
Standard accuracy is misleading for this dataset (99.9% accuracy can be achieved by predicting "No Failure"). We implemented:
- **Balanced Sampling**: All failure cases (719) were combined with a controlled sample of normal cases (5,000) to ensure a strong learning signal. *Note: This balanced pool was used only for model training and evaluation, as real-world deployment will see a much higher class imbalance.*
- **Class Weighting**: Models use `class_weight='balanced'` to penalize missed failures more heavily.
- **Stratification**: 80/20 train-test split with stratification to maintain identical failure ratios in training and testing datasets.

### 2. Model Performance & Limitations
| Metric | Original Result | Optimized Result (Logistic Regression) |
| :--- | :--- | :--- |
| **Recall (Failures)** | **0%** | **Up to 100% (at 0.1 Threshold)** |
| **Accuracy** | 99.9% (Biased) | ~90% (On Balanced Pool) |
| **Status** | Non-functional | **Reliable Early-Warning** |

- **Model Selection**: Logistic Regression was selected for its high responsiveness to threshold tuning and stable behavior on telemetry data.
- **Random Forest Limitations**: Tree-based models like Random Forest were tested but failed to detect minority-class failures reliably under extreme imbalance, even with class weighting.

### 3. Threshold Tuning (Logistic Regression)
By lowering the decision threshold, we significantly boost failure detection (Recall).

| Threshold | Recall | F1-Score | Status |
| :--- | :--- | :--- | :--- |
| 0.5 | 0.5972 | 0.2960 | Standard |
| 0.4 | 0.8056 | 0.2886 | Recommended |
| 0.3 | 0.9167 | 0.2573 | High Recall |
| 0.2 | 0.9792 | 0.2384 | Aggressive |
| 0.1 | 1.0000 | 0.2255 | Extreme (All Caught) |

## 📁 Project Structure

- `Analysis.ipynb`: The main end-to-end pipeline (Loading -> Preprocessing -> Training -> Tuning -> Saving).
- `Final_Report.txt`: Technical summary of the project results and model comparisons.
- `best_model.pkl`: The trained and optimized Logistic Regression model.
- `scaler.pkl`: The fitted StandardScaler used for normalization.
- `README.md`: Project documentation and details.

## 🚀 Future Improvements
- **Advanced Resampling**: Exploration of SMOTE or ADASYN to generate synthetic failure cases.
- **Optuna Optimization**: Hyperparameter tuning where the objective function is explicitly set to maximize recall-focused scores.

## ⚙️ How to Run
1. **Install Dependencies**: `pip install pandas numpy scikit-learn joblib matplotlib seaborn`
2. **Execute Pipeline**: Open `Analysis.ipynb` and select **Run All**.

## 👤 Author
**Muhammad Khizar Arif**
Project - 2026
