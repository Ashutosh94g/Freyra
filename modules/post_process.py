"""Freyra Post-Processing Pipeline -- automatic image enhancement.

Applied automatically after generation for professional-grade output.
Keeps processing lightweight to avoid adding significant time.
"""

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance


def auto_enhance(
    img: Image.Image | np.ndarray,
    sharpen: bool = True,
    color_correct: bool = True,
    film_grain: bool = False,
) -> Image.Image:
    """Apply automatic post-processing to a generated image.

    All adjustments are subtle to preserve the natural look.
    """
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    img = img.copy()

    if color_correct:
        img = _auto_color_correct(img)

    if sharpen:
        img = _adaptive_sharpen(img)

    if film_grain:
        img = _add_film_grain(img)

    return img


def _auto_color_correct(img: Image.Image) -> Image.Image:
    """Subtle color correction for natural skin tones."""
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(1.02)

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.03)

    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.01)

    return img


def _adaptive_sharpen(img: Image.Image) -> Image.Image:
    """Light adaptive sharpening that avoids artifacts."""
    sharpened = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=30, threshold=2))
    return sharpened


def _add_film_grain(img: Image.Image, intensity: float = 0.02) -> Image.Image:
    """Add subtle film grain for authenticity."""
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, intensity * 255, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)
