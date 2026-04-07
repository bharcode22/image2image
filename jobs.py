import threading
from concurrent.futures import ThreadPoolExecutor

_jobs: dict = {}
_jobs_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=2)


def job_set(job_id: str, data: dict):
    with _jobs_lock:
        _jobs[job_id] = data


def job_get(job_id: str):
    with _jobs_lock:
        return _jobs.get(job_id)


def job_list():
    with _jobs_lock:
        return dict(_jobs)
