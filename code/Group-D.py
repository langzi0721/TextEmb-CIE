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

INPUT_EMBEDDING_FILE = 'D-embedding.csv'
df = pd.read_csv(INPUT_EMBEDDING_FILE)

y = df.iloc[:, 2].values
emb_cols = [c for c in df.columns if c.startswith('emb_')]
X_emb_raw = df[emb_cols].values

X_emb_train, X_emb_test = X_emb_raw[:60], X_emb_raw[-15:]
y_train, y_test = y[:60], y[-15:]

scaler_emb = StandardScaler()
X_emb_train_scaled = scaler_emb.fit_transform(X_emb_train)
X_emb_test_scaled = scaler_emb.transform(X_emb_test)

pca = PCA(n_components=0.90, random_state=42)
X_train = pca.fit_transform(X_emb_train_scaled)
X_test = pca.transform(X_emb_test_scaled)

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

xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror', random_state=42, n_jobs=1
)

random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_distributions,
    n_iter=500,
    scoring='neg_mean_squared_error',
    cv=3,
    n_jobs=4,
    random_state=42,
)

random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_

y_test_pred = best_model.predict(X_test)
r2_test = r2_score(y_test, y_test_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
pearson_test, _ = pearsonr(y_test, y_test_pred)

print(
    f'Group D: Dim={X_train.shape[1]} | R2={r2_test:.4f} |'
    f' MAE={mae_test:.4f}% | RMSE={rmse_test:.4f}% | Rho={pearson_test:.4f}'
)