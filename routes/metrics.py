import asyncio
import json
import subprocess

import psutil
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()


def get_gpu_stats() -> list[dict]:
    """Get GPU stats via nvidia-smi."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
        )
        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 8:
                continue
            gpus.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "utilization_pct": float(parts[2]),
                    "memory_used_mb": float(parts[3]),
                    "memory_total_mb": float(parts[4]),
                    "memory_pct": round(float(parts[3]) / float(parts[4]) * 100, 1),
                    "temperature_c": float(parts[5]),
                    "power_draw_w": float(parts[6]) if parts[6] != "[N/A]" else None,
                    "power_limit_w": float(parts[7]) if parts[7] != "[N/A]" else None,
                }
            )
        return gpus
    except Exception:
        return []


def get_system_stats() -> dict:
    cpu_pct = psutil.cpu_percent(interval=None)
    cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
    freq = psutil.cpu_freq()

    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()

    net_io = psutil.net_io_counters()
    disk_io = psutil.disk_io_counters()

    return {
        "cpu": {
            "total_pct": cpu_pct,
            "per_core_pct": cpu_per_core,
            "core_count": psutil.cpu_count(logical=True),
            "freq_mhz": round(freq.current, 1) if freq else None,
        },
        "memory": {
            "total_mb": round(mem.total / 1024**2, 1),
            "used_mb": round(mem.used / 1024**2, 1),
            "available_mb": round(mem.available / 1024**2, 1),
            "pct": mem.percent,
        },
        "swap": {
            "total_mb": round(swap.total / 1024**2, 1),
            "used_mb": round(swap.used / 1024**2, 1),
            "pct": swap.percent,
        },
        "network": {
            "bytes_sent": net_io.bytes_sent,
            "bytes_recv": net_io.bytes_recv,
            "packets_sent": net_io.packets_sent,
            "packets_recv": net_io.packets_recv,
        },
        "disk": {
            "read_bytes": disk_io.read_bytes if disk_io else None,
            "write_bytes": disk_io.write_bytes if disk_io else None,
        },
        "gpu": get_gpu_stats(),
    }


@router.websocket("/ws/metrics")
async def metrics_ws(websocket: WebSocket, interval: float = 1.0):
    """
    WebSocket endpoint that streams system metrics (CPU, RAM, GPU) at `interval` seconds.

    Connect: ws://<host>/ws/metrics?interval=1.0

    Sends JSON frames:
    {
      "cpu": { "total_pct": 12.5, "per_core_pct": [...], "core_count": 8, "freq_mhz": 3200 },
      "memory": { "total_mb": 32768, "used_mb": 8192, "available_mb": 24576, "pct": 25.0 },
      "swap": { "total_mb": 8192, "used_mb": 0, "pct": 0.0 },
      "network": { "bytes_sent": ..., "bytes_recv": ..., ... },
      "disk": { "read_bytes": ..., "write_bytes": ... },
      "gpu": [
        {
          "index": 0, "name": "NVIDIA A100",
          "utilization_pct": 80.0,
          "memory_used_mb": 20000, "memory_total_mb": 40960, "memory_pct": 48.8,
          "temperature_c": 65.0, "power_draw_w": 300.0, "power_limit_w": 400.0
        }
      ]
    }
    """
    interval = max(0.25, min(interval, 60.0))  # clamp: 250ms – 60s

    await websocket.accept()

    # warm-up call so first reading is not 0%
    psutil.cpu_percent(interval=None)

    try:
        while True:
            stats = get_system_stats()
            await websocket.send_text(json.dumps(stats))
            await asyncio.sleep(interval)
    except WebSocketDisconnect:
        pass
