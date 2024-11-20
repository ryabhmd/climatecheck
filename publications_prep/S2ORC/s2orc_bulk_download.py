import json
import os
import re
import requests
import wget
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--s2orc_key", type=str, help="S2ORC API token")
parser.add_argument("--local_path", type=str, help="Path to save the S2ORC data")
args = parser.parse_args()

s2orc_key = args.s2orc_key
local_path = args.local_path

# modify these
API_KEY = s2orc_key
DATASET_NAME = "s2orc"
LOCAL_PATH = local_path

# get latest release's ID
response = requests.get("https://api.semanticscholar.org/datasets/v1/release/latest").json()
RELEASE_ID = response["release_id"]
print(f"Latest release ID: {RELEASE_ID}")
os.makedirs(os.path.join(LOCAL_PATH, f"{RELEASE_ID}"), exist_ok=True)

# get the download links for the s2orc dataset; needs to pass API key through `x-api-key` header
# download via wget. this can take a while...
response = requests.get(f"https://api.semanticscholar.org/datasets/v1/release/{RELEASE_ID}/dataset/{DATASET_NAME}/", headers={"x-api-key": API_KEY}).json()
for url in tqdm(response["files"]):
    match = re.match(r"https://ai2-s2ag.s3.amazonaws.com/staging/(.*)/s2orc/(.*).gz(.*)", url)
    assert match.group(1) == RELEASE_ID
    SHARD_ID = match.group(2)
    wget.download(url, out=os.path.join(LOCAL_PATH, f"{SHARD_ID}.gz"))
print("Downloaded all shards.")
