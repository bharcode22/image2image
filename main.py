from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes import health, generate, img2img, jobs, metrics, nsfw

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(generate.router)
app.include_router(img2img.router)
app.include_router(jobs.router)
app.include_router(metrics.router)
app.include_router(nsfw.router)
