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
pipeline.enable_model_cpu_offload()
pipeline.enable_attention_slicing()

img2img_pipeline = AutoPipelineForImage2Image.from_pipe(pipeline)
