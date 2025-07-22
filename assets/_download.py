from huggingface_hub import snapshot_download
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
snapshot_download(
    repo_id="TianxingChen/RoboTwin2.0",
    allow_patterns=["background_texture/*", "embodiments/*", "objects/*"],
    local_dir=".",
    repo_type="dataset",
    resume_download=True,
    max_workers=8
)
