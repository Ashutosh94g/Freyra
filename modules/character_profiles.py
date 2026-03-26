"""Freyra Character Profiles -- persistent face identity across sessions.

Saves face reference images and metadata to characters/ directory.
Profiles survive app restarts and can be shared.
"""

import os
import json
import time
import shutil
import uuid

import numpy as np
from PIL import Image

CHARACTERS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'characters')


def _ensure_dir():
    os.makedirs(CHARACTERS_DIR, exist_ok=True)


def list_profiles() -> list[dict]:
    """List all saved character profiles."""
    _ensure_dir()
    profiles = []
    for name in sorted(os.listdir(CHARACTERS_DIR)):
        profile_dir = os.path.join(CHARACTERS_DIR, name)
        meta_path = os.path.join(profile_dir, 'meta.json')
        if os.path.isdir(profile_dir) and os.path.exists(meta_path):
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                meta['id'] = name
                meta['path'] = profile_dir
                profiles.append(meta)
            except Exception:
                continue
    return profiles


def list_profile_names() -> list[str]:
    """Return list of profile display names for dropdown."""
    profiles = list_profiles()
    return ['None'] + [p.get('name', p['id']) for p in profiles]


def save_profile(
    name: str,
    face_images: list,
    description: str = '',
) -> str:
    """Save a new character profile.

    Args:
        name: Display name for the character
        face_images: List of numpy arrays (face reference images)
        description: Optional description

    Returns:
        Profile ID string
    """
    _ensure_dir()

    profile_id = f"{name.lower().replace(' ', '_')}_{uuid.uuid4().hex[:6]}"
    profile_dir = os.path.join(CHARACTERS_DIR, profile_id)
    os.makedirs(profile_dir, exist_ok=True)

    saved_images = []
    for i, img in enumerate(face_images):
        if img is not None:
            if isinstance(img, np.ndarray):
                pil_img = Image.fromarray(img)
            else:
                pil_img = img
            img_path = os.path.join(profile_dir, f'face_{i}.png')
            pil_img.save(img_path)
            saved_images.append(f'face_{i}.png')

    meta = {
        'name': name,
        'description': description,
        'images': saved_images,
        'created_at': time.time(),
        'face_method': 'ip_adapter_face',
    }

    with open(os.path.join(profile_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    return profile_id


def load_profile(profile_name: str) -> list | None:
    """Load face images from a saved profile by display name.

    Returns list of numpy arrays, or None if not found.
    """
    if not profile_name or profile_name == 'None':
        return None

    profiles = list_profiles()
    for p in profiles:
        if p.get('name') == profile_name:
            profile_dir = p['path']
            images = []
            for img_file in p.get('images', []):
                img_path = os.path.join(profile_dir, img_file)
                if os.path.exists(img_path):
                    pil_img = Image.open(img_path)
                    images.append(np.array(pil_img))
            return images if images else None

    return None


def delete_profile(profile_name: str) -> bool:
    """Delete a character profile by display name."""
    profiles = list_profiles()
    for p in profiles:
        if p.get('name') == profile_name:
            try:
                shutil.rmtree(p['path'])
                return True
            except Exception:
                return False
    return False
