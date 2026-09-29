import os
import random
import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

warnings.filterwarnings('ignore')


def seed_everything(seed=42):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)


seed_everything(42)

EMBEDDING_FILE = 'BC-embedding.csv'
df = pd.read_csv(EMBEDDING_FILE)

X_raw_cols = ['LUMO / eV', 'CATS3D_02_AP', 'Mor04m', 'E1p', 'P_VSA_MR_5']
X_raw = df[X_raw_cols].values
y = df['ie_ze41'].values

emb_cols = [c for c in df.columns if c.startswith('emb_')]
embeddings_raw = df[emb_cols].values

X_emb_train, X_emb_test = embeddings_raw[:60], embeddings_raw[60:75]
X_desc_train, X_desc_test = X_raw[:60], X_raw[60:75]
y_train, y_test = y[:60], y[60:75]

scaler_desc = StandardScaler()
X_desc_train_scaled = scaler_desc.fit_transform(X_desc_train)
X_desc_test_scaled = scaler_desc.transform(X_desc_test)

scaler_emb = StandardScaler()
X_emb_train_scaled = scaler_emb.fit_transform(X_emb_train)
X_emb_test_scaled = scaler_emb.transform(X_emb_test)

pca = PCA(n_components=0.90, random_state=42)
X_emb_train_pca = pca.fit_transform(X_emb_train_scaled)
X_emb_test_pca = pca.transform(X_emb_test_scaled)

experiments = {
    'Group A': {
        'X_train': X_desc_train_scaled,
        'X_test': X_desc_test_scaled,
    },
    'Group B': {
        'X_train': np.hstack([X_emb_train_scaled, X_desc_train_scaled]),
        'X_test': np.hstack([X_emb_test_scaled, X_desc_test_scaled]),
    },
    'Group C': {
        'X_train': np.hstack([X_emb_train_pca, X_desc_train_scaled]),
        'X_test': np.hstack([X_emb_test_pca, X_desc_test_scaled]),
    },
}

param_distributions = {
    'n_estimators': [100, 300, 500, 800],
    'learning_rate': [0.005, 0.01, 0.05, 0.1],
    'max_depth': [2, 3, 4, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.4, 0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2, 0.5],
    'min_child_weight': [1, 3, 5],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 2, 5],
}


def run_experiment(X_train, y_train, X_test, y_test, seed=42):
  xgb_model = xgb.XGBRegressor(
      objective='reg:squarederror', random_state=seed, n_jobs=1
  )
  random_search = RandomizedSearchCV(
      estimator=xgb_model,
      param_distributions=param_distributions,
      n_iter=500,
      scoring='neg_mean_squared_error',
      cv=3,
      n_jobs=4,
      random_state=seed,
  )
  random_search.fit(X_train, y_train)
  best_model = random_search.best_estimator_

  y_test_pred = best_model.predict(X_test)
  r2 = r2_score(y_test, y_test_pred)
  mae = mean_absolute_error(y_test, y_test_pred)
  rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
  rho, _ = pearsonr(y_test, y_test_pred)

  return r2, mae, rmse, rho


for name, exp in experiments.items():
  r2, mae, rmse, rho = run_experiment(
      exp['X_train'], y_train, exp['X_test'], y_test
  )
  dim = exp['X_train'].shape[1]
  print(
      f'{name}: Dim={dim} | R2={r2:.4f} | MAE={mae:.4f}% | RMSE={rmse:.4f}% |'
      f' Rho={rho:.4f}'
  )