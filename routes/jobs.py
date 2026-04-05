from fastapi import APIRouter, HTTPException

import jobs as job_store

router = APIRouter()


@router.get("/job/{job_id}")
def get_job(job_id: str):
    job = job_store.job_get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, **job}
