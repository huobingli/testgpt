
hf_token = "xxx" #Put your own HF token here, do not publish it
from huggingface_hub import login
# Login directly with your Token (remember not to share this Token publicly)
login(token=hf_token)

import os
import shutil
     

if not os.path.exists('./data'):
    os.makedirs('./data')


jsonl_path = "../data/dataset_new.jsonl"
save_path = '../data/dataset_new'


if os.path.exists(jsonl_path):
    os.remove(jsonl_path)

if os.path.exists(save_path):
    shutil.rmtree(save_path)

directory = "../data"
if not os.path.exists(directory):
    os.makedirs(directory)