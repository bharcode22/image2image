import io
import uuid
import torch
from PIL import Image
from tqdm import tqdm
from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel

import jobs as job_store
from config import OUTPUT_DIR
from pipelines import pipeline, nsfw_pipeline, nsfw_img2img_pipeline, lock
from utils import save_image, resize_to_portrait
from typing import Optional

router = APIRouter()

# Quality boosters yang diappend ke setiap prompt
_QUALITY_TAGS = (
    "masterpiece, best quality, ultra realistic, photorealistic, "
    "8k uhd, highly detailed, sharp focus, cinematic lighting, "
    "skin texture, pores, subsurface scattering, RAW photo, "
    "professional photography, DSLR, f/1.8 aperture"
)

# Negative prompt default — hindari artefak umum SDXL
_DEFAULT_NEGATIVE = (
    "lowres, bad anatomy, bad hands, missing fingers, extra fingers, "
    "fewer digits, mutated hands, deformed, ugly, blurry, jpeg artifacts, "
    "signature, watermark, username, text, logo, worst quality, low quality, "
    "normal quality, cropped, out of frame, duplicate, morbid, gross proportions, "
    "long neck, extra limbs, disfigured, poorly drawn face, mutation, bad proportions, "
    "cartoon, anime, illustration, painting, drawing, art, sketch, 3d render, cgi"
)


def _enhance_prompt(prompt: str) -> str:
    return f"{prompt.rstrip(', ')}, {_QUALITY_TAGS}"


def _build_negative(user_negative: str) -> str:
    if user_negative:
        return f"{user_negative}, {_DEFAULT_NEGATIVE}"
    return _DEFAULT_NEGATIVE


# ─── txt2img ────────────────────────────────────────────────────────────────

def _run_nsfw_generate(job_id, prompt, negative_prompt, height, width, num_inference_steps, guidance_scale):
    try:
        print(f"[NSFW] 🚀 START job_id={job_id}")
        job_store.job_set(job_id, {"status": "processing"})

        height = (height // 8) * 8
        width = (width // 8) * 8

        enhanced_prompt = _enhance_prompt(prompt)
        full_negative = _build_negative(negative_prompt)

        seed = torch.randint(0, 2**32 - 1, (1,)).item()
        generator = torch.Generator(device="cpu").manual_seed(seed)

        pbar = tqdm(total=num_inference_steps, desc=f"[NSFW] job_id={job_id}")
        _progress = lambda pipe, step, timestep, cb: (pbar.update(1), cb)[1]

        nsfw_pipeline.set_progress_bar_config(disable=True)
        print(f"[NSFW] ⏳ Generating image job_id={job_id}")

        torch.cuda.empty_cache()
        with lock:
            with torch.inference_mode():
                image = nsfw_pipeline(
                    prompt=enhanced_prompt,
                    negative_prompt=full_negative,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    callback_on_step_end=_progress,
                    callback_on_step_end_tensor_inputs=[],
                ).images[0]

        pbar.close()

        filename, filepath = save_image(image, job_id)
        print(f"[NSFW] 💾 Saved: {filepath} job_id={job_id}")
        torch.cuda.empty_cache()

        job_store.job_set(job_id, {"status": "done", "filename": filename, "path": filepath, "seed": seed})
        print(f"[NSFW] ✅ DONE job_id={job_id}")

    except Exception as e:
        print(f"[NSFW] ❌ ERROR job_id={job_id} error={str(e)}")
        job_store.job_set(job_id, {"status": "error", "error": str(e)})


@router.post("/nsfw/generate")
def nsfw_generate(req: NsfwPromptRequest, request: Request):
    job_id = uuid.uuid4().hex
    print(f"[API] 📥 NSFW txt2img request job_id={job_id}")

    job_store.job_set(job_id, {"status": "pending"})
    future = job_store._executor.submit(
        _run_nsfw_generate,
        job_id,
        req.prompt,
        req.negative_prompt or "",
        req.height or 1024,
        req.width or 1024,
        req.num_inference_steps or 40,
        req.guidance_scale if req.guidance_scale is not None else 7.0,
    )
    future.result()

    job = job_store.job_get(job_id)
    if job.get("status") == "error":
        raise HTTPException(status_code=500, detail=job.get("error", "Generation failed"))

    return {
        "job_id": job_id,
        "status": "done",
        "filename": job.get("filename"),
        "path": job.get("path"),
        "download-url": str(request.base_url) + "nsfw/download/" + job_id,
        "seed": job.get("seed"),
    }


# ─── img2img ────────────────────────────────────────────────────────────────

def _run_nsfw_img2img(job_id, image_bytes, prompt, negative_prompt, height, width, num_inference_steps, guidance_scale, strength):
    try:
        print(f"[NSFW-IMG2IMG] 🚀 START job_id={job_id}")
        job_store.job_set(job_id, {"status": "processing"})

        height = (height // 8) * 8
        width = (width // 8) * 8

        enhanced_prompt = _enhance_prompt(prompt)
        full_negative = _build_negative(negative_prompt)

        seed = torch.randint(0, 2**32 - 1, (1,)).item()
        generator = torch.Generator(device="cpu").manual_seed(seed)

        init_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        init_image = resize_to_portrait(init_image, (width, height))

        print(f"[NSFW-IMG2IMG] ⏳ Generating image job_id={job_id}")
        torch.cuda.empty_cache()
        with lock:
            with torch.inference_mode():
                image = nsfw_img2img_pipeline(
                    prompt=enhanced_prompt,
                    negative_prompt=full_negative,
                    image=init_image,
                    strength=strength,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                ).images[0]

        filename, filepath = save_image(image, job_id)
        print(f"[NSFW-IMG2IMG] 💾 Saved: {filepath} job_id={job_id}")
        torch.cuda.empty_cache()

        job_store.job_set(job_id, {"status": "done", "filename": filename, "path": filepath, "seed": seed})
        print(f"[NSFW-IMG2IMG] ✅ DONE job_id={job_id}")

    except Exception as e:
        print(f"[NSFW-IMG2IMG] ❌ ERROR job_id={job_id} error={str(e)}")
        job_store.job_set(job_id, {"status": "error", "error": str(e)})


class NsfwPromptRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    height: Optional[int] = None
    width: Optional[int] = None
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None


@router.post("/nsfw/img2img")
def nsfw_img2img(
    request: Request,
    prompt: str = Form(...),
    file: UploadFile = File(...),
    negative_prompt: Optional[str] = Form(None),
    height: Optional[int] = Form(None),
    width: Optional[int] = Form(None),
    num_inference_steps: Optional[int] = Form(None),
    guidance_scale: Optional[float] = Form(None),
    strength: Optional[float] = Form(None),
):
    job_id = uuid.uuid4().hex
    print(f"[API] 📥 NSFW img2img request job_id={job_id}")

    image_bytes = file.file.read()
    job_store.job_set(job_id, {"status": "pending"})
    future = job_store._executor.submit(
        _run_nsfw_img2img,
        job_id,
        image_bytes,
        prompt,
        negative_prompt or "",
        height or 1024,
        width or 1024,
        # steps lebih tinggi → detail lebih baik; min 40 untuk img2img
        num_inference_steps or 40,
        # guidance 7.0: balance antara prompt adherence & naturalness
        guidance_scale if guidance_scale is not None else 7.0,
        # strength 0.5: pertahankan struktur input, tambah detail NSFW style
        strength if strength is not None else 0.5,
    )
    future.result()

    job = job_store.job_get(job_id)
    if job.get("status") == "error":
        raise HTTPException(status_code=500, detail=job.get("error", "Generation failed"))

    return {
        "job_id": job_id,
        "status": "done",
        "filename": job.get("filename"),
        "path": job.get("path"),
        "download-url": str(request.base_url) + "nsfw/download/" + job_id,
        "seed": job.get("seed"),
    }


# ─── FLUX → NSFW chain ──────────────────────────────────────────────────────
# FLUX generate image → hasilnya dipakai sebagai input ke NSFW img2img

def _run_flux_nsfw_chain(job_id, prompt, negative_prompt, height, width, flux_steps, nsfw_steps, guidance_scale, strength):
    try:
        print(f"[FLUX→NSFW] 🚀 START job_id={job_id}")
        job_store.job_set(job_id, {"status": "processing"})

        # FLUX butuh kelipatan 16
        flux_h = (height // 16) * 16
        flux_w = (width // 16) * 16

        seed = torch.randint(0, 2**32 - 1, (1,)).item()
        generator = torch.Generator(device="cpu").manual_seed(seed)

        # Step 1: FLUX txt2img — gunakan prompt bersih (FLUX lebih baik tanpa SDXL tags)
        pbar = tqdm(total=flux_steps, desc=f"[FLUX→NSFW] FLUX step job_id={job_id}")
        _progress_flux = lambda pipe, step, timestep, cb: (pbar.update(1), cb)[1]

        pipeline.set_progress_bar_config(disable=True)
        print(f"[FLUX→NSFW] ⏳ Step 1: FLUX generating job_id={job_id}")

        torch.cuda.empty_cache()
        with lock:
            with torch.inference_mode():
                flux_image = pipeline(
                    prompt=prompt,
                    height=flux_h,
                    width=flux_w,
                    num_inference_steps=flux_steps,
                    generator=generator,
                    callback_on_step_end=_progress_flux,
                    callback_on_step_end_tensor_inputs=[],
                ).images[0]
        pbar.close()

        # Step 2: NSFW img2img refine — apply ultra realistic style di atas struktur FLUX
        nsfw_h = (height // 8) * 8
        nsfw_w = (width // 8) * 8
        # Resize dengan LANCZOS untuk kualitas terbaik
        init_image = flux_image.resize((nsfw_w, nsfw_h), Image.LANCZOS)

        enhanced_prompt = _enhance_prompt(prompt)
        full_negative = _build_negative(negative_prompt)

        pbar2 = tqdm(total=nsfw_steps, desc=f"[FLUX→NSFW] NSFW step job_id={job_id}")
        _progress_nsfw = lambda pipe, step, timestep, cb: (pbar2.update(1), cb)[1]

        nsfw_img2img_pipeline.set_progress_bar_config(disable=True)
        print(f"[FLUX→NSFW] ⏳ Step 2: NSFW refining job_id={job_id}")

        torch.cuda.empty_cache()
        with lock:
            with torch.inference_mode():
                image = nsfw_img2img_pipeline(
                    prompt=enhanced_prompt,
                    negative_prompt=full_negative,
                    image=init_image,
                    strength=strength,
                    num_inference_steps=nsfw_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    callback_on_step_end=_progress_nsfw,
                    callback_on_step_end_tensor_inputs=[],
                ).images[0]
        pbar2.close()

        filename, filepath = save_image(image, job_id)
        print(f"[FLUX→NSFW] 💾 Saved: {filepath} job_id={job_id}")
        torch.cuda.empty_cache()

        job_store.job_set(job_id, {"status": "done", "filename": filename, "path": filepath, "seed": seed})
        print(f"[FLUX→NSFW] ✅ DONE job_id={job_id}")

    except Exception as e:
        print(f"[FLUX→NSFW] ❌ ERROR job_id={job_id} error={str(e)}")
        job_store.job_set(job_id, {"status": "error", "error": str(e)})


class ChainRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    height: Optional[int] = None
    width: Optional[int] = None
    flux_steps: Optional[int] = None
    nsfw_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    strength: Optional[float] = None


@router.post("/nsfw/chain")
def nsfw_chain(req: ChainRequest, request: Request):
    """FLUX txt2img → NSFW img2img. FLUX bangun struktur, NSFW apply ultra realistic style."""
    job_id = uuid.uuid4().hex
    print(f"[API] 📥 FLUX→NSFW chain request job_id={job_id}")

    job_store.job_set(job_id, {"status": "pending"})
    future = job_store._executor.submit(
        _run_flux_nsfw_chain,
        job_id,
        req.prompt,
        req.negative_prompt or "",
        req.height or 1024,
        req.width or 1024,
        # FLUX: 10 steps cukup untuk dasar komposisi yang solid
        req.flux_steps or 10,
        # NSFW: 35 steps untuk detail maksimal dengan DPM++ Karras
        req.nsfw_steps or 35,
        # guidance 7.5 untuk chain: perlu sedikit lebih kuat karena refining
        req.guidance_scale if req.guidance_scale is not None else 7.5,
        # strength 0.55: pertahankan ~45% struktur FLUX, apply 55% NSFW style
        req.strength if req.strength is not None else 0.55,
    )
    future.result()

    job = job_store.job_get(job_id)
    if job.get("status") == "error":
        raise HTTPException(status_code=500, detail=job.get("error", "Generation failed"))

    return {
        "job_id": job_id,
        "status": "done",
        "filename": job.get("filename"),
        "path": job.get("path"),
        "download-url": str(request.base_url) + "nsfw/download/" + job_id,
        "seed": job.get("seed"),
    }


# ─── shared download/image endpoints ────────────────────────────────────────

@router.get("/nsfw/image/{job_id}")
def nsfw_image(job_id: str):
    import os
    filepath = os.path.join(OUTPUT_DIR, f"{job_id}.png")
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(filepath, media_type="image/png", filename=f"{job_id}.png")


@router.get("/nsfw/download/{job_id}")
def nsfw_download(job_id: str):
    import os
    filepath = os.path.join(OUTPUT_DIR, f"{job_id}.png")
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(
        filepath,
        media_type="image/png",
        filename=f"{job_id}.png",
        headers={"Content-Disposition": f"attachment; filename={job_id}.png"},
    )
