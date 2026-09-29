import os
import random
import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

os.environ['OMP_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore')

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

INPUT_EMBEDDING_FILE = 'E-embedding.csv'
df = pd.read_csv(INPUT_EMBEDDING_FILE)

y = df.iloc[:, 2].values
X_num_raw = df[['CATS3D_02_AP', 'Mor04m']].values
emb_cols = [c for c in df.columns if c.startswith('emb_')]
X_emb_raw = df[emb_cols].values

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

random_seeds = list(range(36, 49))

r2_list, mae_list, rmse_list, rho_list = [], [], [], []

print(f"{'Seed':<6} | {'R2':<8} | {'MAE':<10} | {'RMSE':<10} | {'Rho':<8}")
print("-" * 55)

for seed in random_seeds:
    seed_everything(seed)
    
    X_emb_train, X_emb_test, X_num_train, X_num_test, y_train, y_test = train_test_split(
        X_emb_raw, X_num_raw, y, test_size=15, random_state=seed
    )
    
    scaler_emb = StandardScaler()
    X_emb_train_scaled = scaler_emb.fit_transform(X_emb_train)
    X_emb_test_scaled = scaler_emb.transform(X_emb_test)
    
    pca = PCA(n_components=0.90, random_state=seed)
    X_emb_train_pca = pca.fit_transform(X_emb_train_scaled)
    X_emb_test_pca = pca.transform(X_emb_test_scaled)
    
    scaler_num = StandardScaler()
    X_num_train_scaled = scaler_num.fit_transform(X_num_train)
    X_num_test_scaled = scaler_num.transform(X_num_test)
    
    X_train = np.hstack((X_num_train_scaled, X_emb_train_pca))
    X_test = np.hstack((X_num_test_scaled, X_emb_test_pca))
    
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror', random_state=seed, n_jobs=-1 
    )
    
    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_distributions,
        n_iter=500,
        scoring='neg_mean_squared_error',
        cv=3,
        n_jobs=1,  
        random_state=seed,
    )
    
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    
    y_pred = best_model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rho = pearsonr(y_test, y_pred)[0]
    
    r2_list.append(r2)
    mae_list.append(mae)
    rmse_list.append(rmse)
    rho_list.append(rho)

    print(f"{seed:<6} | {r2:<8.4f} | {mae:<10.2f} | {rmse:<10.2f} | {rho:<8.4f}")

print('\n' + '=' * 60)
print('Results (Mean +/- Std)')
print('-' * 60)
print(f"R2   : {np.mean(r2_list):.4f} +/- {np.std(r2_list):.4f}")
print(f"MAE  : {np.mean(mae_list):.4f} +/- {np.std(mae_list):.4f}")
print(f"RMSE : {np.mean(rmse_list):.4f} +/- {np.std(rmse_list):.4f}")
print(f"Rho  : {np.mean(rho_list):.4f} +/- {np.std(rho_list):.4f}")
print('=' * 60)