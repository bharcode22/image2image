import threading
import torch
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image

from config import MODEL_PATH_FLUX

lock = threading.Lock()

pipeline = AutoPipelineForText2Image.from_pretrained(
    MODEL_PATH_FLUX,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
)
# Sequential offload: offload setiap layer ke CPU satu per satu → VRAM minimum
pipeline.enable_sequential_cpu_offload()
pipeline.enable_attention_slicing()

# VAE-level memory optimization untuk resolusi tinggi
if hasattr(pipeline, 'vae'):
    if hasattr(pipeline.vae, 'enable_slicing'):
        pipeline.vae.enable_slicing()
    if hasattr(pipeline.vae, 'enable_tiling'):
        pipeline.vae.enable_tiling()

img2img_pipeline = AutoPipelineForImage2Image.from_pipe(pipeline)
