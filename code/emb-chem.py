import os
import random
import warnings

# 1. 屏蔽 Python 原生 Warning 与 Transformers 日志
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoModel, AutoTokenizer

transformers.logging.set_verbosity_error()

# ================= 2. 严格锁定所有随机种子 =================
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

seed_everything(42)

# ================= 3. 数据与模型配置 =================
INPUT_FILE = 'data.csv'
EMBEDDING_FILE = 'Chem-embedding.csv'
MODEL_NAME = 'seyonec/ChemBERTa-zinc-base-v1'

df = pd.read_csv(INPUT_FILE)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
model = AutoModel.from_pretrained(
    MODEL_NAME, local_files_only=True, use_safetensors=True
).to(device)

def get_embedding(smiles):
    model.eval()
    inputs = tokenizer(
        smiles,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=128,
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]

total_samples = len(df)
step = max(1, total_samples // 10)
embeddings = []

for i, smiles in enumerate(df['isomeric_smiles'], 1):
    embeddings.append(get_embedding(smiles))
    if i % step == 0 or i == total_samples:
        pct = int(round((i / total_samples) * 100))
        print(f"Embedding: {pct}%/100%")

embeddings_raw = np.stack(embeddings)
emb_df = pd.DataFrame(
    embeddings_raw,
    columns=[f'emb_{i}' for i in range(embeddings_raw.shape[1])],
)
full_df = pd.concat([df, emb_df], axis=1)
full_df.to_csv(EMBEDDING_FILE, index=False)

print(f"Save {EMBEDDING_FILE}")