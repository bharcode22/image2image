import torch
from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health():
    import psutil

    cpu_percent = psutil.cpu_percent(interval=0.1)
    cpu_cores = psutil.cpu_count(logical=False)
    cpu_threads = psutil.cpu_count(logical=True)

    ram = psutil.virtual_memory()
    ram_total_gb = round(ram.total / 1024**3, 2)
    ram_used_gb = round(ram.used / 1024**3, 2)
    ram_free_gb = round(ram.available / 1024**3, 2)
    ram_percent = ram.percent

    gpu_info = None
    if torch.cuda.is_available():
        vram_free, vram_total = torch.cuda.mem_get_info()
        gpu_info = {
            "name": torch.cuda.get_device_name(0),
            "vram_total_gb": round(vram_total / 1024**3, 2),
            "vram_used_gb": round((vram_total - vram_free) / 1024**3, 2),
            "vram_free_gb": round(vram_free / 1024**3, 2),
            "vram_used_percent": round((vram_total - vram_free) / vram_total * 100, 1),
        }

    return {
        "status": "ok",
        "cpu": {
            "usage_percent": cpu_percent,
            "cores_physical": cpu_cores,
            "cores_logical": cpu_threads,
        },
        "ram": {
            "total_gb": ram_total_gb,
            "used_gb": ram_used_gb,
            "free_gb": ram_free_gb,
            "used_percent": ram_percent,
        },
        "gpu": gpu_info,
    }
