import threading
import torch
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image

from config import MODEL_PATH_FLUX, MODEL_PATH_NSFW

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

nsfw_pipeline = AutoPipelineForText2Image.from_pretrained(
    MODEL_PATH_NSFW,
    torch_dtype=torch.float16,
    local_files_only=True,
)
nsfw_pipeline.enable_sequential_cpu_offload()
nsfw_pipeline.enable_attention_slicing()

if hasattr(nsfw_pipeline, 'vae'):
    if hasattr(nsfw_pipeline.vae, 'enable_slicing'):
        nsfw_pipeline.vae.enable_slicing()
    if hasattr(nsfw_pipeline.vae, 'enable_tiling'):
        nsfw_pipeline.vae.enable_tiling()

nsfw_img2img_pipeline = AutoPipelineForImage2Image.from_pipe(nsfw_pipeline)
