from fastapi import FastAPI

from routes import health, generate, img2img, jobs

app = FastAPI()

app.include_router(health.router)
app.include_router(generate.router)
app.include_router(img2img.router)
app.include_router(jobs.router)
