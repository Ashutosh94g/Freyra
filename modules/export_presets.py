"""Freyra Social Media Export -- platform-specific cropping and EXIF.

Crops generated images to platform-optimal aspect ratios with
face-centered framing and applies appropriate EXIF metadata.
"""

import numpy as np
from PIL import Image


EXPORT_FORMATS = {
    'instagram_square': {
        'label': 'Instagram Square',
        'width': 1080,
        'height': 1080,
        'platform': 'Instagram',
    },
    'instagram_portrait': {
        'label': 'Instagram Portrait',
        'width': 1080,
        'height': 1350,
        'platform': 'Instagram',
    },
    'instagram_story': {
        'label': 'Instagram Story / Reels',
        'width': 1080,
        'height': 1920,
        'platform': 'Instagram',
    },
    'tiktok': {
        'label': 'TikTok',
        'width': 1080,
        'height': 1920,
        'platform': 'TikTok',
    },
    'twitter': {
        'label': 'Twitter / X',
        'width': 1600,
        'height': 900,
        'platform': 'Twitter',
    },
    'facebook': {
        'label': 'Facebook',
        'width': 1200,
        'height': 630,
        'platform': 'Facebook',
    },
}

EXPORT_FORMAT_LABELS = [v['label'] for v in EXPORT_FORMATS.values()]


def _find_face_center(img_array: np.ndarray) -> tuple[int, int]:
    """Find approximate face center in an image.

    Uses a simple heuristic: face is typically in the upper-center third.
    """
    h, w = img_array.shape[:2]
    return w // 2, h // 3


def crop_for_platform(
    img: Image.Image | np.ndarray,
    format_key: str,
) -> Image.Image:
    """Crop and resize image for a specific platform with face-centered framing."""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    fmt = EXPORT_FORMATS.get(format_key)
    if fmt is None:
        return img

    target_w = fmt['width']
    target_h = fmt['height']
    target_ratio = target_w / target_h

    src_w, src_h = img.size
    src_ratio = src_w / src_h

    img_array = np.array(img)
    face_x, face_y = _find_face_center(img_array)

    if src_ratio > target_ratio:
        new_w = int(src_h * target_ratio)
        new_h = src_h
        left = max(0, min(face_x - new_w // 2, src_w - new_w))
        top = 0
    else:
        new_w = src_w
        new_h = int(src_w / target_ratio)
        left = 0
        top = max(0, min(face_y - new_h // 3, src_h - new_h))

    cropped = img.crop((left, top, left + new_w, top + new_h))
    return cropped.resize((target_w, target_h), Image.LANCZOS)


def export_with_exif(
    img: Image.Image | np.ndarray,
    format_key: str,
    camera_profile: str = 'iPhone 15 Pro Max',
    output_path: str | None = None,
) -> Image.Image:
    """Export image with platform-appropriate EXIF and cropping."""
    cropped = crop_for_platform(img, format_key)

    try:
        from modules.metadata_spoof import apply_camera_exif
        exif_bytes = apply_camera_exif(camera_profile=camera_profile)
        if output_path and exif_bytes:
            cropped.save(output_path, exif=exif_bytes, quality=95)
    except Exception:
        if output_path:
            cropped.save(output_path, quality=95)

    return cropped
