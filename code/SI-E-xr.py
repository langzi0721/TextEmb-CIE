import os
import random
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42)

INPUT_EMBEDDING_FILE = 'E-embedding.csv'

df = pd.read_csv(INPUT_EMBEDDING_FILE)

y = df.iloc[:, 2].values
emb_cols = [c for c in df.columns if c.startswith('emb_')]
X_emb_raw = df[emb_cols].values

y_train, y_test = y[:60], y[-15:]
X_emb_train, X_emb_test = X_emb_raw[:60], X_emb_raw[-15:]

ablation_configs = [
    ("1", "Descriptor (CATS3D+Mor04m)",        False, False, True,  True),
    ("2", "E-raw (No PCA, No Desc)",           True,  False, False, False),
    ("3", "E-PCA (PCA, No Desc)",              True,  True,  False, False),
    ("4", "E-PCA-C (PCA + CATS3D)",            True,  True,  True,  False),
    ("5", "E-PCA-M (PCA + Mor04m)",            True,  True,  False, True),
    ("6", "E-raw-CM (No PCA + CATS3D+Mor04m)", True,  False, True,  True),
    ("7", "E-PCA-CM (Full Model)",             True,  True,  True,  True)
]

final_results = []

for idx_str, desc, use_emb, use_pca, use_cats, use_mor in ablation_configs:
    train_feats = []
    test_feats = []
    
    if use_emb:
        scaler_emb = StandardScaler()
        X_emb_train_scaled = scaler_emb.fit_transform(X_emb_train)
        X_emb_test_scaled = scaler_emb.transform(X_emb_test)
        
        if use_pca:
            pca = PCA(n_components=0.90, random_state=42)
            X_emb_train_final = pca.fit_transform(X_emb_train_scaled)
            X_emb_test_final = pca.transform(X_emb_test_scaled)
        else:
            X_emb_train_final = X_emb_train_scaled
            X_emb_test_final = X_emb_test_scaled
            
        train_feats.append(X_emb_train_final)
        test_feats.append(X_emb_test_final)
        
    num_cols = []
    if use_cats: num_cols.append('CATS3D_02_AP')
    if use_mor: num_cols.append('Mor04m')
        
    if num_cols:
        X_num_train_raw = df[num_cols].values[:60]
        X_num_test_raw = df[num_cols].values[-15:]
        
        scaler_num = StandardScaler()
        X_num_train_scaled = scaler_num.fit_transform(X_num_train_raw)
        X_num_test_scaled = scaler_num.transform(X_num_test_raw)
        
        train_feats.append(X_num_train_scaled)
        test_feats.append(X_num_test_scaled)
        
    X_train = np.hstack(train_feats)
    X_test = np.hstack(test_feats)
    
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
    
    final_results.append({
        "ID": idx_str,
        "R2": f"{r2_test:.4f}",
        "MAE": f"{mae_test:.4f}",
        "RMSE": f"{rmse_test:.4f}",
        "r": f"{pearson_test:.4f}"
    })

results_df = pd.DataFrame(final_results)

print(f"{'ID':<10}{'R2':<15}{'MAE':<15}{'RMSE':<15}{'r':<15}")
print("-" * 70)
for idx, row in results_df.iterrows():
    print(f"{row['ID']:<10}{row['R2']:<15}{row['MAE']:<15}{row['RMSE']:<15}{row['r']:<15}")