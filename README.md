**Repository Overview**

This repository contains the data and source code for the study: **"Text Embedding-Assisted Prediction of Corrosion Inhibition Efficiency Using Machine Learning under Small-Sample Conditions"**.

**File & Directory Structure**

* **Root Folder**
* `requirements.txt`: Python package dependencies and environment configurations.
* `result.txt`: Output results and performance metrics.


* **Data/**
* `BC-embedding.csv`: Molecular representations extracted using ChemBERTa.
* `D-embedding.csv`: Text-based embeddings generated via Qwen for Group D.
* `E-embedding.csv`: Text-based embeddings generated via Qwen for Group E.


* **Code/**
* `Group-ABC.py`: Model training and evaluation script for Group ABC (requires `data.csv` and `BC-embedding.csv` in the same directory).
* `Group-D.py`: Script for Group D analysis (requires `D-embedding.csv`).
* `Group-E.py`: Script for Group E training and evaluation (requires `E-embedding.csv`).
* `Group-E-pca.py`: PCA dimension analysis script for generating figure results (requires `E-embedding.csv`).
* `SI-E-xr.py`: Ablation analysis for Group E, evaluating different combinations of the Qwen embedding, PCA, `CATS3D_02_AP`, and `Mor04m` (requires `E-embedding.csv`).
* `SI-E-rs.py`: Random train-test split sensitivity analysis for Group E using multiple random seeds (requires `E-embedding.csv`).
* `SI-E-pca.py`: PCA variance-retention sensitivity analysis for Group E using different cumulative variance thresholds (requires `E-embedding.csv`).
* `emb-chem.py`: Feature extraction script using ChemBERTa.
* `emb-qwen.py`: Feature extraction script using Qwen text embeddings.


* **Other/**
* `prompt.txt`: Prompts utilized for text embedding generation.



## Quick Start

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Run a model or supporting analysis using:

```bash
python <script_name>.py
```

For example:

```bash
python Group-ABC.py
python Group-D.py
python Group-E.py
```

Supporting analyses can be executed using:

```bash
python Group-E-pca.py
python SI-E-xr.py
python SI-E-rs.py
python SI-E-pca.py
```

---

```
## XGBoost Hyperparameter Optimization

XGBoost was used as the downstream regression model for Groups A–E and the supporting analyses.

For the main experiments, the base model was defined as:

```python
xgb.XGBRegressor(
    objective='reg:squarederror',
    random_state=42,
    n_jobs=1
)
```

Hyperparameters were optimized using `RandomizedSearchCV`.

The following hyperparameter search space was used:

```python
param_distributions = {
    'n_estimators': [100, 300, 500, 800],
    'learning_rate': [0.005, 0.01, 0.05, 0.1],
    'max_depth': [2, 3, 4, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.4, 0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2, 0.5],
    'min_child_weight': [1, 3, 5],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 2, 5]
}
```

For the main experiments, hyperparameter optimization was performed using:

```python
RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_distributions,
    n_iter=500,
    scoring='neg_mean_squared_error',
    cv=3,
    n_jobs=4,
    random_state=42
)
```

Thus, 500 randomly sampled hyperparameter combinations were evaluated using 3-fold cross-validation on the training set.

The best-performing XGBoost model was selected according to the cross-validated mean squared error and subsequently evaluated on the held-out test set.

For the random train-test split sensitivity analysis in `SI-E-rs.py`, the corresponding split-specific random seed was used for XGBoost and `RandomizedSearchCV`.

---

## PCA Settings

For Groups C, D, and E, PCA was applied only to the embedding features.

The embedding features were first standardized using `StandardScaler`. The scaler was fitted using the training data only and then applied to the test data.

The main PCA configuration was:

```python
PCA(
    n_components=0.90,
    random_state=42
)
```

This setting retains the number of principal components required to explain at least 90% of the cumulative variance.

For the original Group E train-test split, 13 principal components were retained, explaining approximately 90.90% of the cumulative variance.

Together with the two explicitly retained numerical descriptors (`CATS3D_02_AP` and `Mor04m`), the final Group E feature vector contained 15 dimensions.

---

**Environment & Hardware Specs**

```text
==================================================
SOFTWARE & HARDWARE INFORMATION
==================================================
OS              : Linux 5.14.0-570.55.1.el9_6.x86_64 (x86_64)
Python Version  : 3.9.25
CPU Core Count  : 32
CUDA Available  : True
CUDA Version    : 12.1
GPU Device      : NVIDIA GeForce RTX 4060 Ti
GPU VRAM        : 15.57 GB
--------------------------------------------------
PYTHON PACKAGES
--------------------------------------------------
numpy           : 2.0.2
pandas          : 2.3.3
scipy           : 1.13.1
scikit-learn    : 1.6.1
xgboost         : 2.1.4
torch           : 2.5.1+cu121
rdkit           : 2025.09.2
transformers    : 4.57.5
openai          : 1.100.2
==================================================

```
