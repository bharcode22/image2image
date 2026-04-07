import os
import uuid
import torch
from tqdm import tqdm
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

import jobs as job_store
from jobs import job_list
from config import OUTPUT_DIR, URL
from pipelines import pipeline, lock, get_refiner, get_uncensored_pipeline, get_smnth_pipeline, smnth_lock, uncensored_lock
from utils import save_image

router = APIRouter()


class PromptRequest(BaseModel):
    prompt: str


def _run_generate(job_id: str, req_prompt: str):
    try:
        print(f"[FLUX] 🚀 START job_id={job_id}")
        job_store.job_set(job_id, {"status": "processing"})

        prompt = (
            req_prompt
            + ", Ultra-realistic, photorealistic, natural skin textures, cinematic lighting, shallow depth of field, HDR, 8K. Highly detailed"
        )

        seed = torch.randint(0, 2**32 - 1, (1,)).item()
        generator = torch.Generator(device="cpu").manual_seed(seed)

        num_steps = 8

        pbar = tqdm(total=num_steps, desc=f"[FLUX] job_id={job_id}")
        _progress = lambda pipe, step, timestep, cb: (pbar.update(1), cb)[1]

        pipeline.set_progress_bar_config(disable=True)
        print(f"[FLUX] ⏳ Generating image job_id={job_id}")
        with lock:
            with torch.inference_mode():
                image = pipeline(
                    prompt=prompt,
                    height=1216,
                    width=832,
                    num_inference_steps=num_steps,
                    generator=generator,
                    callback_on_step_end=_progress,
                    callback_on_step_end_tensor_inputs=[],
                ).images[0]
        pbar.close()

        filename, filepath = save_image(image, job_id)
        print(f"[FLUX] 💾 Saved: {filepath} job_id={job_id}")

        torch.cuda.empty_cache()

        job_store.job_set(job_id, {
            "status": "done",
            "filename": filename,
            "path": filepath,
            "seed": seed,
        })

        print(f"[FLUX] ✅ DONE job_id={job_id}")

    except Exception as e:
        print(f"[FLUX] ❌ ERROR job_id={job_id} error={str(e)}")
        job_store.job_set(job_id, {"status": "error", "error": str(e)})


@router.post("/generate")
def generate(req: PromptRequest):
    job_id = uuid.uuid4().hex

    print(f"[API] 📥 Request received job_id={job_id}")

    job_store.job_set(job_id, {"status": "pending"})
    future = job_store._executor.submit(_run_generate, job_id, req.prompt)
    future.result()

    job = job_store.job_get(job_id)
    if job.get("status") == "error":
        raise HTTPException(status_code=500, detail=job.get("error", "Generation failed"))

    return {
        "job_id": job_id,
        "status": "done",
        "filename": job.get("filename"),
        "path": job.get("path"),
        "download-url": URL+'/generate/download/'+job_id,
        "seed": job.get("seed"),
    }


@router.post("/generate/difusion")
def generate_difusion(req: PromptRequest):
    n_steps = 50
    high_noise_frac = 0.8

    full_prompt = (
        req.prompt
        + "Ultra-realistic, photorealistic, natural skin textures, cinematic lighting, shallow depth of field, HDR, 8K. Highly detailed"
    )
    negative_prompt = (
        "cartoon, anime, illustration, painting, drawing, art, sketch, "
        "cgi, render, 3d, blurry, soft focus, low quality, worst quality, "
        "lowres, jpeg artifacts, distorted face, bad anatomy, disfigured, "
        "deformed, extra limbs, extra fingers, missing fingers, "
        "poorly drawn hands, poorly drawn face, mutation, "
        "watermark, signature, text, logo, oversaturated, overexposed"
    )

    with lock:
        latent = pipeline(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            height=1216,
            width=832,
            num_inference_steps=n_steps,
            guidance_scale=7.5,
            denoising_end=high_noise_frac,
            output_type="latent",
        ).images

        refiner = get_refiner()
        image = refiner(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=n_steps,
            guidance_scale=7.5,
            denoising_start=high_noise_frac,
            image=latent,
        ).images[0]

    filename, filepath = save_image(image)
    return {"filename": filename, "path": filepath}


def _run_smnth(job_id: str, req_prompt: str):
    try:
        print(f"[SMNTH] 🚀 START job_id={job_id}")

        job_store.job_set(job_id, {"status": "processing"})

        full_prompt = (
            "Smnth_v1, "
            + req_prompt
            + ", cinematic lighting, shallow depth of field, HDR, highly detailed, sharp focus"
        )

        negative_prompt = (
            "(deformed iris, deformed pupils), text, watermark, logo, signature, "
            "low quality, worst quality, blurry, grainy, low resolution, "
            "(plastic skin, fake skin, airbrushed), (cartoon, anime, 3d, render, illustration), "
            "(deformed, distorted, disfigured), poorly drawn, bad anatomy, wrong anatomy, "
            "extra limb, missing limb, floating limbs, (mutated hands and fingers), "
            "disconnected limbs, mutation, mutated, ugly, disgusting, amputation"
        )

        print(f"[SMNTH] 🎯 Prompt ready job_id={job_id}")

        seed = torch.randint(0, 2**32 - 1, (1,)).item()
        generator = torch.Generator(device="cpu").manual_seed(seed)

        pipe = get_smnth_pipeline()
        print(f"[SMNTH] ⚙️ Pipeline loaded job_id={job_id}")

        with smnth_lock:
            print(f"[SMNTH] ⏳ Generating image job_id={job_id}")
            with torch.inference_mode():
                image = pipe(
                    prompt=full_prompt,
                    negative_prompt=negative_prompt,
                    guidance_scale=6.0,
                    num_inference_steps=8,
                    width=512,
                    height=768,
                    generator=generator,
                ).images[0]

        filename, filepath = save_image(image, job_id)
        print(f"[SMNTH] 💾 Saved: {filepath} job_id={job_id}")

        torch.cuda.empty_cache()

        job_store.job_set(job_id, {
            "status": "done",
            "filename": filename,
            "path": filepath,
            "seed": seed
        })

        print(f"[SMNTH] ✅ DONE job_id={job_id}")

    except Exception as e:
        print(f"[SMNTH] ❌ ERROR job_id={job_id} error={str(e)}")

        job_store.job_set(job_id, {
            "status": "error",
            "error": str(e)
        })

@router.post("/generate/smnth")
def generate_smnth(req: PromptRequest):
    job_id = uuid.uuid4().hex

    print(f"[API] 📥 Request received job_id={job_id}")

    job_store.job_set(job_id, {"status": "pending"})
    job_store._executor.submit(_run_smnth, job_id, req.prompt)

    return {"job_id": job_id, "status": "pending"}


def _run_uncensored(job_id: str, req_prompt: str):
    try:
        job_store.job_set(job_id, {"status": "processing"})
        full_prompt = (
            req_prompt
            + ", cinematic lighting, shallow depth of field, HDR, 8K, highly detailed, sharp focus"
        )
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
        generator = torch.Generator(device="cpu").manual_seed(seed)

        with uncensored_lock:
            pipe = get_uncensored_pipeline()
            image = pipe(
                prompt=full_prompt,
                guidance_scale=7.0,
                num_inference_steps=28,
                width=832,
                height=1216,
                generator=generator,
            ).images[0]

        filename, filepath = save_image(image, job_id)
        job_store.job_set(job_id, {"status": "done", "filename": filename, "path": filepath, "seed": seed})
    except Exception as e:
        job_store.job_set(job_id, {"status": "error", "error": str(e)})


@router.post("/generate/uncensored")
def generate_uncensored(req: PromptRequest):
    job_id = uuid.uuid4().hex
    job_store.job_set(job_id, {"status": "pending"})
    job_store._executor.submit(_run_uncensored, job_id, req.prompt)
    return {"job_id": job_id, "status": "pending"}


@router.get("/generate/list")
def list_images():
    all_jobs = job_list()
    result = []
    for job_id, job in all_jobs.items():
        if job.get("status") == "done":
            result.append({
                "job_id": job_id,
                "filename": job.get("filename"),
                "seed": job.get("seed"),
                "download_url": URL + "/generate/download/" + job_id,
                "image_url": URL + "/generate/image/" + job_id,
            })
    return {"total": len(result), "images": result}


@router.get("/generate/image/{job_id}")
def get_image(job_id: str):
    filepath = os.path.join(OUTPUT_DIR, f"{job_id}.png")
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(filepath, media_type="image/png", filename=f"{job_id}.png")


@router.get("/generate/download/{job_id}")
def download_image(job_id: str):
    filepath = os.path.join(OUTPUT_DIR, f"{job_id}.png")
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(filepath, media_type="image/png", filename=f"{job_id}.png", headers={"Content-Disposition": f"attachment; filename={job_id}.png"})
