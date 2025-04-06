from huggingface_hub import snapshot_download
from huggingface_hub.utils.logging import set_verbosity_info


set_verbosity_info()
model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
snapshot_download(repo_id=model_id, allow_patterns="*.json")
