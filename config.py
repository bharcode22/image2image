import os

# Kurangi fragmentasi memori CUDA
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

OUTPUT_DIR = "/home/pod/folder/zaq/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH_FLUX = "/home/pod/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots/e7b7dc27f91deacad38e78976d1f2b499d76a294"
MODEL_PATH_NSFW = "/home/pod/.cache/huggingface/hub/models--UnfilteredAI--NSFW-gen-v2/snapshots/982782a450570e5f064016b404d4b7a1c19dbad5"
