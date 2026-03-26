"""Freyra Face Consistency Engine -- unified multi-method face identity.

Supports:
- IP-Adapter FaceID (default, T4-safe, ~7-8GB)
- More methods can be added: InstantID, PuLID (future)

Auto-selects the best method based on available VRAM.
"""

import os
import numpy as np

import modules.config
import modules.flags as flags


FACE_METHODS = {
    'ip_adapter_face': {
        'label': 'IP-Adapter Face',
        'description': 'Good identity preservation, lightweight',
        'vram_required_gb': 7.5,
        'stop': 0.75,
        'weight': 0.9,
    },
}

DEFAULT_METHOD = 'ip_adapter_face'


def get_available_methods() -> list[str]:
    """Return methods that can fit in current VRAM."""
    try:
        import ldm_patched.modules.model_management as mm
        info = mm.get_vram_info()
        free_gb = info.get('free_gb', 0)
    except Exception:
        free_gb = 15.0

    available = []
    for key, cfg in FACE_METHODS.items():
        if free_gb >= cfg['vram_required_gb'] * 0.7:
            available.append(key)

    if not available:
        available = [DEFAULT_METHOD]

    return available


def auto_select_method() -> str:
    """Auto-select the best face method for current VRAM."""
    available = get_available_methods()
    return available[0]


def prepare_face_tasks(
    face_images: list,
    method: str | None = None,
) -> dict:
    """Prepare face images for the generation pipeline.

    Returns a cn_tasks dict compatible with AsyncTask.cn_tasks format.
    """
    if method is None:
        method = auto_select_method()

    cfg = FACE_METHODS.get(method, FACE_METHODS[DEFAULT_METHOD])
    cn_tasks = {x: [] for x in flags.ip_list}

    for img in face_images:
        if img is not None:
            cn_tasks[flags.cn_ip_face].append([
                img,
                cfg['stop'],
                cfg['weight'],
            ])

    return cn_tasks


def compute_face_similarity(reference_img, generated_img) -> float:
    """Compute a basic face similarity score between reference and generated images.

    Returns a score from 0 to 100. Higher = more similar.
    Uses a lightweight pixel-based comparison on face regions.
    """
    try:
        from extras.face_crop import crop_image
        ref_crop = crop_image(reference_img)
        gen_crop = crop_image(generated_img)

        if ref_crop is None or gen_crop is None:
            return 0.0

        from PIL import Image
        ref_pil = Image.fromarray(ref_crop).resize((112, 112))
        gen_pil = Image.fromarray(gen_crop).resize((112, 112))

        ref_arr = np.array(ref_pil, dtype=np.float32) / 255.0
        gen_arr = np.array(gen_pil, dtype=np.float32) / 255.0

        diff = np.abs(ref_arr - gen_arr)
        similarity = max(0, 100 * (1 - np.mean(diff) * 2))

        return round(similarity, 1)
    except Exception:
        return 0.0
