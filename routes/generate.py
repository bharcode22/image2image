import os
import uuid
import torch
from tqdm import tqdm
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

import jobs as job_store
from config import OUTPUT_DIR, URL
from pipelines import pipeline, lock
from utils import save_image

router = APIRouter()


class PromptRequest(BaseModel):
    prompt: str


def _run_generate(job_id: str, req_prompt: str):
    try:
        print(f"[FLUX] 🚀 START job_id={job_id}")
        job_store.job_set(job_id, {"status": "processing"})

        # prompt = (
        #     req_prompt
        #     + ", Ultra-realistic, photorealistic, natural skin textures, cinematic lighting, shallow depth of field, HDR, 8K. Highly detailed"
        # )
        prompt = (
            req_prompt
            + ", "
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


@router.get("/generate/list")
def list_images():
    files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".png")]
    result = []
    for f in sorted(files):
        job_id = os.path.splitext(f)[0]
        stat = os.stat(os.path.join(OUTPUT_DIR, f))
        result.append({
            "job_id": job_id,
            "filename": f,
            "size_bytes": stat.st_size,
            "created_at": stat.st_mtime,
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


@router.delete("/generate/delete/{job_id}")
def delete_image(job_id: str):
    filepath = os.path.join(OUTPUT_DIR, f"{job_id}.png")
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Image not found")
    os.remove(filepath)
    return {"message": "Image deleted", "job_id": job_id}
