# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "modelscope",
# ]
# ///
from modelscope import snapshot_download

# Download models
snapshot_download("PAI/Wan2.1-Fun-1.3B-InP", local_dir="models/PAI/Wan2.1-Fun-1.3B-InP")
# snapshot_download("Wan-AI/Wan2.1-T2V-1.3B", local_dir="models/Wan-AI/Wan2.1-T2V-1.3B")