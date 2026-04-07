import gc
import threading
import torch
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image

from config import MODEL_PATH_FLUX, SMNTH_BASE_MODEL, SMNTH_LORA_PATH

lock = threading.Lock()

pipeline = AutoPipelineForText2Image.from_pretrained(
    MODEL_PATH_FLUX,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
)
pipeline.enable_model_cpu_offload()
pipeline.enable_attention_slicing()

img2img_pipeline = AutoPipelineForImage2Image.from_pipe(pipeline)

# --- Smnth ---
smnth_pipeline = None
smnth_lock = threading.Lock()


def _unload_pipeline(p):
    """Hapus satu pipeline dari RAM dan VRAM."""
    if p is None:
        return
    try:
        from accelerate.hooks import remove_hook_from_submodules
        remove_hook_from_submodules(p)
    except Exception:
        pass
    for attr in ("unet", "transformer", "text_encoder", "text_encoder_2", "vae", "image_encoder"):
        component = getattr(p, attr, None)
        if component is not None:
            try:
                component.to("cpu")
            except Exception:
                pass
            setattr(p, attr, None)
    del p


def _offload_other_pipelines():
    """Unload semua pipeline lain dari RAM/VRAM sebelum load smnth."""
    global pipeline, img2img_pipeline

    for p in [pipeline, img2img_pipeline]:
        _unload_pipeline(p)

    pipeline = None
    img2img_pipeline = None

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    print("[SMNTH] ♻️ All other pipelines unloaded from RAM")


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
            torch_dtype=torch.float16,
            local_files_only=True,
            safety_checker=None,
            requires_safety_checker=False,
        )

        # Disable safety checker if it was loaded anyway
        if hasattr(pipe, "safety_checker"):
            pipe.safety_checker = None

        pipe.load_lora_weights(SMNTH_LORA_PATH)

        pipe.enable_model_cpu_offload()

        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        if hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()

        smnth_pipeline = pipe

    return smnth_pipeline
