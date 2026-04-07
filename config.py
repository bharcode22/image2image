import os

OUTPUT_DIR = "/home/pod/folder/zaq/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
URL="http://192.168.199.40:8000"

MODEL_PATH_FLUX = "/home/pod/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots/e7b7dc27f91deacad38e78976d1f2b499d76a294"

SMNTH_BASE_MODEL = "Tongyi-MAI/Z-Image-Turbo"
SMNTH_LORA_PATH = "/home/pod/.cache/huggingface/hub/models--Kakelaka--Smnth_v1_NSFW1/snapshots/a6239cf7523ff91a10969b9cc9ccf633537e0ebb/Smnth_v1_NSFW1.safetensors"
