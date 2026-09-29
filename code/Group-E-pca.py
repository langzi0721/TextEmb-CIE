import os
import random
import warnings
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


def seed_everything(seed=42):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)


seed_everything(42)

INPUT_EMBEDDING_FILE = 'E-embedding.csv'
df = pd.read_csv(INPUT_EMBEDDING_FILE)

emb_cols = [c for c in df.columns if c.startswith('emb_')]
X_emb_raw = df[emb_cols].values
X_emb_train = X_emb_raw[:60]

scaler_emb = StandardScaler()
X_emb_train_scaled = scaler_emb.fit_transform(X_emb_train)

pca = PCA(n_components=25, random_state=42)
pca.fit(X_emb_train_scaled)

exp_var = pca.explained_variance_ratio_ * 100
cum_var = np.cumsum(exp_var)

for i in range(25):
  print(
      f'PC{i+1:<2}: Var={exp_var[i]:.2f}% | CumVar={cum_var[i]:.2f}%'
      f"{' (Optimal: PC13)' if i == 12 else ''}"
  )