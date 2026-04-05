import threading
import torch
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image, DiffusionPipeline

from config import MODEL_PATH_FLUX, FLUX_DEV_PATH, HEARTSYNC_LORA_PATH, SMNTH_BASE_MODEL, SMNTH_LORA_PATH

lock = threading.Lock()

pipeline = AutoPipelineForText2Image.from_pretrained(
    MODEL_PATH_FLUX,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
)
pipeline.enable_model_cpu_offload()
pipeline.enable_attention_slicing()

img2img_pipeline = AutoPipelineForImage2Image.from_pipe(pipeline)

# --- Refiner ---
refiner_pipeline = None
refiner_lock = threading.Lock()


def get_refiner():
    global refiner_pipeline
    if refiner_pipeline is not None:
        return refiner_pipeline
    with refiner_lock:
        if refiner_pipeline is not None:
            return refiner_pipeline
        refiner_pipeline = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=pipeline.text_encoder_2,
            vae=pipeline.vae,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            variant="fp16",
        )
        refiner_pipeline.enable_model_cpu_offload()
        refiner_pipeline.enable_attention_slicing()
    return refiner_pipeline


# --- Uncensored ---
uncensored_pipeline = None
uncensored_lock = threading.Lock()


def get_uncensored_pipeline():
    global uncensored_pipeline
    if uncensored_pipeline is not None:
        return uncensored_pipeline
    with uncensored_lock:
        if uncensored_pipeline is not None:
            return uncensored_pipeline
        pipe = AutoPipelineForText2Image.from_pretrained(
            FLUX_DEV_PATH,
            torch_dtype=torch.bfloat16,
        )
        pipe.load_lora_weights(
            HEARTSYNC_LORA_PATH,
            weight_name="lora.safetensors",
            adapter_name="uncensored",
        )
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()
        uncensored_pipeline = pipe
    return uncensored_pipeline


# --- Smnth ---
smnth_pipeline = None
smnth_lock = threading.Lock()


def _offload_other_pipelines():
    """Move other pipelines off GPU before loading smnth to free VRAM."""
    for p in [pipeline, img2img_pipeline, refiner_pipeline, uncensored_pipeline]:
        if p is not None and hasattr(p, "to"):
            try:
                p.to("cpu")
            except Exception:
                pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_smnth_pipeline():
    global smnth_pipeline
    if smnth_pipeline is not None:
        return smnth_pipeline
    with smnth_lock:
        if smnth_pipeline is not None:
            return smnth_pipeline
        _offload_other_pipelines()
        pipe = AutoPipelineForText2Image.from_pretrained(
            SMNTH_BASE_MODEL,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        pipe.load_lora_weights(SMNTH_LORA_PATH)
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()
        smnth_pipeline = pipe
    return smnth_pipeline
