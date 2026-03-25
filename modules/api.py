"""Freyra REST API — Phase 5

Provides headless generation control via FastAPI endpoints mounted alongside Gradio.

Endpoints:
    POST /api/v1/generate   — Submit a generation job, returns job_id
    GET  /api/v1/jobs/{id}   — Poll job status + result image paths
    GET  /api/v1/models      — List available checkpoints, LoRAs, VAEs
    GET  /api/v1/presets      — List available presets
    GET  /api/v1/styles       — List available style names
"""

import uuid
import threading
import time
import os

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import modules.config as config
import modules.flags as flags
import modules.async_worker as worker

api_app = FastAPI(title="Freyra API", version="1.0.0")

# ---------------------------------------------------------------------------
# In-memory job store (lightweight, no DB needed for Colab sessions)
# ---------------------------------------------------------------------------
_jobs: dict = {}
_jobs_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class GenerateRequest(BaseModel):
    prompt: str = ""
    negative_prompt: str | None = None
    styles: list[str] | None = None
    performance: str | None = None
    aspect_ratio: str | None = None
    image_number: int = Field(default=1, ge=1, le=32)
    output_format: str = "png"
    seed: int = -1
    sharpness: float | None = None
    cfg_scale: float | None = None
    base_model: str | None = None
    refiner_model: str | None = None
    refiner_switch: float | None = None
    sampler: str | None = None
    scheduler: str | None = None
    vae: str | None = None
    loras: list[list] | None = None
    overwrite_step: int | None = None
    clip_skip: int | None = None
    read_wildcards_in_order: bool = False
    save_metadata_to_images: bool | None = None
    metadata_scheme: str | None = None


class GenerateResponse(BaseModel):
    job_id: str
    status: str = "queued"


class JobStatus(BaseModel):
    job_id: str
    status: str  # queued | running | completed | failed
    progress: int | None = None
    progress_text: str | None = None
    images: list[str] | None = None
    error: str | None = None


class ModelsResponse(BaseModel):
    checkpoints: list[str]
    loras: list[str]
    vaes: list[str]


class PresetsResponse(BaseModel):
    presets: list[str]


class StylesResponse(BaseModel):
    styles: list[str]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@api_app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    """Submit a text-to-image generation job."""
    job_id = str(uuid.uuid4())

    # Build params dict, only including explicitly set values
    params = {}
    for field_name, value in req.model_dump(exclude_none=True).items():
        params[field_name] = value

    # Create AsyncTask via the new dict constructor
    task = worker.AsyncTask.from_dict(params)

    with _jobs_lock:
        _jobs[job_id] = {
            "status": "queued",
            "task": task,
            "images": [],
            "error": None,
            "progress": 0,
            "progress_text": "",
        }

    # Enqueue for the existing worker thread
    worker.async_tasks.append(task)

    # Start a watcher thread to track completion
    watcher = threading.Thread(target=_watch_job, args=(job_id, task), daemon=True)
    watcher.start()

    return GenerateResponse(job_id=job_id, status="queued")


@api_app.get("/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str):
    """Poll the status of a generation job."""
    with _jobs_lock:
        job = _jobs.get(job_id)

    if job is None:
        return JobStatus(job_id=job_id, status="not_found")

    return JobStatus(
        job_id=job_id,
        status=job["status"],
        progress=job.get("progress"),
        progress_text=job.get("progress_text"),
        images=job.get("images"),
        error=job.get("error"),
    )


@api_app.get("/models", response_model=ModelsResponse)
def list_models():
    """List available checkpoints, LoRAs, and VAEs."""
    config.update_files()
    return ModelsResponse(
        checkpoints=config.model_filenames,
        loras=config.lora_filenames,
        vaes=config.vae_filenames,
    )


@api_app.get("/presets", response_model=PresetsResponse)
def list_presets():
    """List available preset names."""
    return PresetsResponse(presets=config.get_presets())


@api_app.get("/styles", response_model=StylesResponse)
def list_styles():
    """List available style names."""
    from modules.sdxl_styles import legal_style_names
    return StylesResponse(styles=legal_style_names)


@api_app.get("/outputs/{filename}")
def get_output_file(filename: str):
    """Serve a generated image file."""
    # Sanitize: only allow simple filenames, no path traversal
    safe_name = os.path.basename(filename)
    file_path = os.path.join(config.path_outputs, safe_name)
    if not os.path.isfile(file_path):
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)


# ---------------------------------------------------------------------------
# Job watcher — monitors AsyncTask completion
# ---------------------------------------------------------------------------
def _watch_job(job_id: str, task: worker.AsyncTask):
    """Background thread that watches an AsyncTask until it finishes."""
    with _jobs_lock:
        _jobs[job_id]["status"] = "running"

    while True:
        # Check yields for progress updates
        while len(task.yields) > 0:
            flag, product = task.yields.pop(0)
            if flag == "preview":
                if isinstance(product, tuple) and len(product) >= 2:
                    with _jobs_lock:
                        _jobs[job_id]["progress"] = product[0]
                        _jobs[job_id]["progress_text"] = str(product[1])
            elif flag == "finish":
                with _jobs_lock:
                    _jobs[job_id]["status"] = "completed"
                    _jobs[job_id]["progress"] = 100
                    _jobs[job_id]["images"] = [str(p) for p in product] if isinstance(product, list) else []
                return

        # Check if task is done via results
        if task.processing is False and len(task.results) > 0:
            with _jobs_lock:
                _jobs[job_id]["status"] = "completed"
                _jobs[job_id]["progress"] = 100
                _jobs[job_id]["images"] = [str(p) for p in task.results]
            return

        time.sleep(0.5)
