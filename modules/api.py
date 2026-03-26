"""Freyra REST API v3

Provides headless generation control via FastAPI endpoints.

Endpoints:
    POST /api/v1/generate        — Submit via raw params
    POST /api/v1/studio/generate — Submit via creative dimensions (Freyra v3)
    GET  /api/v1/jobs/{id}       — Poll job status + result image paths
    GET  /api/v1/models          — List available checkpoints, LoRAs, VAEs
    GET  /api/v1/shoot-types     — List available shoot types
    GET  /api/v1/characters      — List saved character profiles
    GET  /api/v1/styles          — List available style names
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


@api_app.get("/shoot-types")
def list_shoot_types():
    """List available Freyra shoot types with their configurations."""
    from modules.shoot_types import SHOOT_TYPES
    return {
        key: {
            'label': st['label'],
            'description': st['description'],
            'aspect_ratio': st['aspect_ratio'],
        }
        for key, st in SHOOT_TYPES.items()
    }


@api_app.get("/characters")
def list_characters():
    """List saved character profiles."""
    from modules.character_profiles import list_profiles
    profiles = list_profiles()
    return [
        {
            'id': p['id'],
            'name': p.get('name', ''),
            'description': p.get('description', ''),
            'image_count': len(p.get('images', [])),
        }
        for p in profiles
    ]


class StudioGenerateRequest(BaseModel):
    """Generate using Freyra creative dimensions (opinionated API)."""
    shoot_type: str = "Fashion Editorial"
    quality: str = "Standard"
    character_name: str | None = None
    skin_tone: str | None = None
    hair_style: str | None = None
    hair_color: str | None = None
    outfit: str | None = None
    pose: str | None = None
    makeup: str | None = None
    expression: str | None = None
    background: str | None = None
    lighting: str | None = None
    camera_angle: str | None = None
    footwear: str | None = None
    custom_prompt: str | None = None
    image_number: int = Field(default=1, ge=1, le=4)
    seed: int = -1


@api_app.post("/studio/generate", response_model=GenerateResponse)
def studio_generate(req: StudioGenerateRequest):
    """Generate using Freyra creative dimensions."""
    import random
    from modules.shoot_types import get_shoot_type, get_quality_mode, SHOOT_TYPES, QUALITY_MODES
    from modules.prompt_assembler import assemble_prompt
    import modules.constants as constants

    shoot = get_shoot_type(req.shoot_type)
    if shoot is None:
        shoot = list(SHOOT_TYPES.values())[0]

    quality = get_quality_mode(req.quality)
    if quality is None:
        quality = QUALITY_MODES['standard']

    assembled = assemble_prompt(
        shoot_type_config=shoot,
        skin_tone=req.skin_tone or '',
        hair_style=req.hair_style or '',
        hair_color=req.hair_color or '',
        outfit=req.outfit or '',
        pose=req.pose or '',
        makeup=req.makeup or '',
        expression=req.expression or '',
        background=req.background or '',
        lighting=req.lighting or '',
        camera_angle=req.camera_angle or '',
        footwear=req.footwear or '',
        custom_prompt=req.custom_prompt or '',
    )

    seed = req.seed if req.seed >= 0 else random.randint(constants.MIN_SEED, constants.MAX_SEED)

    params = {
        'prompt': assembled['prompt'],
        'negative_prompt': assembled['negative_prompt'],
        'styles': assembled.get('styles', []),
        'performance': quality['performance'],
        'generation_steps': quality['steps'],
        'aspect_ratio': assembled.get('aspect_ratio', '896*1152'),
        'image_number': req.image_number,
        'seed': seed,
        'cfg_scale': assembled.get('cfg_scale', 4.5),
        'sharpness': assembled.get('sharpness', 2.0),
        'sampler': 'dpmpp_2m_sde_gpu',
        'scheduler': 'karras',
    }

    task = worker.AsyncTask.from_dict(params)

    if req.character_name:
        from modules.character_profiles import load_profile
        from modules.face_engine import prepare_face_tasks
        face_images = load_profile(req.character_name)
        if face_images:
            task.cn_tasks = prepare_face_tasks(face_images)
            task.input_image_checkbox = True
            task.current_tab = 'ip'

    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "queued",
            "task": task,
            "images": [],
            "error": None,
            "progress": 0,
            "progress_text": "",
        }

    worker.async_tasks.append(task)
    watcher = threading.Thread(target=_watch_job, args=(job_id, task), daemon=True)
    watcher.start()

    return GenerateResponse(job_id=job_id, status="queued")


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
