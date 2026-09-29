import os
import time
import pandas as pd
from openai import OpenAI

INPUT_FILE = 'data.csv'
OUTPUT_FILE = 'Embedding.csv'

API_KEY = "YOUR_API_KEY"
BASE_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
EMBEDDING_MODEL = 'text-embedding-v4'
EMBEDDING_DIM = 1024

df = pd.read_csv(INPUT_FILE)
total_samples = len(df)
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

prompts = []
for _, row in df.iterrows():
# D and E in prompt.txt
    prompt_text = f""" """
    prompts.append(prompt_text)

def get_embedding(text):
    for attempt in range(3):
        try:
            clean_text = str(text).replace('\n', ' ').strip()
            resp = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=[clean_text],
                dimensions=EMBEDDING_DIM
            )
            return resp.data[0].embedding
        except Exception:
            if attempt < 2:
                time.sleep(2)
                continue
            return [0.0] * EMBEDDING_DIM

embeddings_list = []
for idx, p_text in enumerate(prompts, 1):
    step_start = time.time()
    emb = get_embedding(p_text)
    embeddings_list.append(emb)
    step_time = time.time() - step_start
    print(f"Progress: [{idx}/{total_samples}] | Time: {step_time:.2f}s")

emb_cols = [f'emb_{i}' for i in range(EMBEDDING_DIM)]
df_emb = pd.DataFrame(embeddings_list, columns=emb_cols)
df_final = pd.concat([df.reset_index(drop=True), df_emb], axis=1)
df_final.to_csv(OUTPUT_FILE, index=False)