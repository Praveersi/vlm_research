from huggingface_hub import hf_hub_download
import os

local_dir = "data/raw"
os.makedirs(local_dir, exist_ok=True)
path = hf_hub_download(repo_id="blumenstiel/Sen1Floods11", filename="sen1floods11_v1_1.tar.gz", repo_type="dataset", local_dir=local_dir)
print(f"Downloaded to {path}")
