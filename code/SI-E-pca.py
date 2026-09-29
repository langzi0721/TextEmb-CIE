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
X_numeric_raw = df[['CATS3D_02_AP', 'Mor04m']].values

emb_cols = [c for c in df.columns if c.startswith('emb_')]
X_emb_raw = df[emb_cols].values

X_emb_train, X_emb_test = X_emb_raw[:60], X_emb_raw[-15:]
X_num_train, X_num_test = X_numeric_raw[:60], X_numeric_raw[-15:]
y_train, y_test = y[:60], y[-15:]

pca_variance_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
results = []

for var in pca_variance_list:
    var_label = f"{int(var * 100)}%"
    
    scaler_emb = StandardScaler()
    X_emb_train_scaled = scaler_emb.fit_transform(X_emb_train)
    X_emb_test_scaled = scaler_emb.transform(X_emb_test)

    if var == 1.0:
        pca = PCA(random_state=42)
    else:
        pca = PCA(n_components=var, random_state=42)
        
    X_emb_train_pca = pca.fit_transform(X_emb_train_scaled)
    X_emb_test_pca = pca.transform(X_emb_test_scaled)
    
    retained_dims = pca.n_components_

    scaler_num = StandardScaler()
    X_num_train_scaled = scaler_num.fit_transform(X_num_train)
    X_num_test_scaled = scaler_num.transform(X_num_test)

    X_train = np.hstack((X_num_train_scaled, X_emb_train_pca))
    X_test = np.hstack((X_num_test_scaled, X_emb_test_pca))
    
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
    
    results.append({
        "Variance": var_label,
        "Dims": retained_dims,
        "R2": f"{r2_test:.4f}",
        "MAE": f"{mae_test:.4f}",
        "RMSE": f"{rmse_test:.4f}",
        "r": f"{pearson_test:.4f}"
    })

results_df = pd.DataFrame(results)

print(f"{'Variance':<15}{'Dims':<12}{'R2':<12}{'MAE':<12}{'RMSE':<12}{'r':<12}")
print("-" * 75)
for idx, row in results_df.iterrows():
    print(f"{row['Variance']:<15}{row['Dims']:<12}{row['R2']:<12}{row['MAE']:<12}{row['RMSE']:<12}{row['r']:<12}")