from huggingface_hub import snapshot_download

snapshot_download(repo_id="luodian/llama-7b-hf", ignore_patterns=["*.h5", "*.ot", "*.msgpack"], local_dir="D:/work/gpt/llama-7b-hf", local_dir_use_symlinks=False)