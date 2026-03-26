"""Freyra Skin Tone Detector -- estimate skin tone from a face reference image.

Uses the median color of the central face region to classify into one of
the wildcard skin tone categories. Lightweight -- no ML models needed.
"""

import numpy as np


SKIN_TONE_MAP = [
    ('fair porcelain',     (230, 210, 200)),
    ('light ivory',        (220, 195, 175)),
    ('light beige',        (210, 180, 160)),
    ('medium warm',        (195, 160, 135)),
    ('medium olive',       (180, 150, 120)),
    ('tan golden',         (165, 130, 100)),
    ('brown caramel',      (140, 105, 80)),
    ('dark brown',         (110, 80, 60)),
    ('deep ebony',         (75, 55, 40)),
]


def _color_distance(c1: tuple, c2: tuple) -> float:
    return sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5


def detect_skin_tone(face_image: np.ndarray) -> str:
    """Estimate skin tone from a face image.

    Samples the central face region (avoiding hair/background at edges)
    and finds the closest match in our skin tone palette.

    Returns the matching skin tone label string, or empty string on failure.
    """
    if face_image is None:
        return ''

    try:
        h, w = face_image.shape[:2]

        # Sample center-face region (forehead to chin, avoiding edges)
        y1 = int(h * 0.25)
        y2 = int(h * 0.70)
        x1 = int(w * 0.30)
        x2 = int(w * 0.70)

        roi = face_image[y1:y2, x1:x2]

        if roi.size == 0:
            return ''

        # Use median to be robust against highlights/shadows
        median_color = tuple(int(v) for v in np.median(roi.reshape(-1, 3), axis=0)[:3])

        best_tone = ''
        best_dist = float('inf')
        for label, ref_color in SKIN_TONE_MAP:
            dist = _color_distance(median_color, ref_color)
            if dist < best_dist:
                best_dist = dist
                best_tone = label

        return best_tone
    except Exception:
        return ''
