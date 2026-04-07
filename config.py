import os

OUTPUT_DIR = "/home/pod/folder/zaq/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
URL="http://192.168.199.40:8000"

MODEL_PATH = "/home/pod/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b"
MODEL_PATH_FLUX = "/home/pod/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots/e7b7dc27f91deacad38e78976d1f2b499d76a294"

FLUX_DEV_PATH = "black-forest-labs/FLUX.1-dev"
HEARTSYNC_LORA_PATH = "/home/pod/.cache/huggingface/hub/models--Heartsync--Flux-NSFW-uncensored/snapshots/328160f16bc4072cf2bb7ee162d77f3b9b5f786c"

SMNTH_BASE_MODEL = "Tongyi-MAI/Z-Image-Turbo"
SMNTH_LORA_PATH = "/home/pod/.cache/huggingface/hub/models--Kakelaka--Smnth_v1_NSFW1/snapshots/a6239cf7523ff91a10969b9cc9ccf633537e0ebb/Smnth_v1_NSFW1.safetensors"
