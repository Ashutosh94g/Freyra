"""Freyra Image Interrogator -- extract per-dimension descriptions from reference images.

Wraps the existing BLIP interrogator to provide dimension-aware captions.
When a user uploads a reference image for a creative dimension (outfit, lighting,
background, etc.), this module extracts a relevant text description.
"""

import re

DIMENSION_KEYWORDS = {
    'outfit': [
        'wearing', 'dress', 'shirt', 'top', 'pants', 'jeans', 'skirt',
        'shorts', 'jacket', 'coat', 'blazer', 'sweater', 'hoodie', 'suit',
        'bikini', 'swimsuit', 'leggings', 'bodysuit', 'blouse', 'vest',
        'crop top', 'tank', 'cardigan', 'jumpsuit', 'romper', 'gown',
        'lingerie', 'bra', 'corset', 'trousers', 'turtleneck', 'denim',
        'lace', 'silk', 'satin', 'velvet', 'leather', 'cotton', 'linen',
        'sequin', 'metallic', 'floral', 'striped', 'plaid', 'mesh',
        'chiffon', 'knit', 'crochet', 'tulle', 'cashmere', 'wool',
        'activewear', 'sportswear', 'athletic', 'yoga', 'gym',
    ],
    'pose': [
        'standing', 'sitting', 'leaning', 'walking', 'running', 'laying',
        'crouching', 'kneeling', 'stretching', 'jumping', 'posing',
        'hand on hip', 'arms crossed', 'looking', 'reaching', 'bending',
        'reclining', 'squatting', 'turning', 'twisting',
    ],
    'background': [
        'beach', 'city', 'street', 'studio', 'gym', 'pool', 'garden',
        'forest', 'mountain', 'hotel', 'cafe', 'restaurant', 'office',
        'rooftop', 'balcony', 'park', 'desert', 'ocean', 'lake',
        'interior', 'exterior', 'indoor', 'outdoor', 'wall', 'window',
        'door', 'room', 'building', 'alley', 'market', 'temple',
        'museum', 'gallery', 'library', 'farm', 'field', 'bridge',
    ],
    'lighting': [
        'light', 'lighting', 'sunlight', 'daylight', 'shadow', 'backlit',
        'golden hour', 'sunset', 'sunrise', 'overcast', 'bright', 'dim',
        'dark', 'neon', 'flash', 'studio', 'natural', 'artificial',
        'warm', 'cool', 'diffused', 'harsh', 'soft', 'dramatic',
        'silhouette', 'spotlight', 'ambient', 'candlelight', 'moonlight',
    ],
    'makeup': [
        'makeup', 'lipstick', 'eyeshadow', 'mascara', 'eyeliner', 'blush',
        'foundation', 'contour', 'highlight', 'gloss', 'matte', 'dewy',
        'natural', 'glam', 'smoky', 'nude', 'bold', 'subtle', 'bronzed',
        'shimmer', 'glitter', 'lashes', 'brows', 'lip', 'skin',
    ],
    'expression': [
        'smiling', 'laughing', 'serious', 'confident', 'happy', 'sad',
        'angry', 'surprised', 'contemplative', 'relaxed', 'playful',
        'fierce', 'gentle', 'mysterious', 'sultry', 'joyful', 'calm',
        'intense', 'dreamy', 'pensive', 'determined', 'cheerful',
    ],
    'hair_style': [
        'hair', 'ponytail', 'braid', 'bun', 'curls', 'waves', 'straight',
        'bob', 'pixie', 'updo', 'bangs', 'layers', 'slicked', 'afro',
        'cornrows', 'twists', 'dreadlocks', 'mohawk',
    ],
    'hair_color': [
        'blonde', 'brunette', 'black', 'red', 'auburn', 'copper',
        'silver', 'grey', 'white', 'pink', 'blue', 'green', 'purple',
        'ombre', 'balayage', 'highlights', 'platinum', 'caramel',
        'chestnut', 'honey', 'strawberry', 'ash', 'golden', 'rose gold',
    ],
    'camera_angle': [
        'close-up', 'wide shot', 'portrait', 'full body', 'profile',
        'overhead', 'low angle', 'high angle', 'dutch angle', 'front',
        'side', 'back', 'medium shot', 'eye level', 'bird eye',
    ],
    'footwear': [
        'shoes', 'boots', 'heels', 'sneakers', 'sandals', 'flats',
        'pumps', 'loafers', 'slides', 'slippers', 'barefoot',
        'stilettos', 'platforms', 'wedges', 'mules', 'espadrilles',
    ],
    'skin_tone': [
        'fair', 'light', 'medium', 'tan', 'dark', 'deep', 'porcelain',
        'ivory', 'olive', 'brown', 'ebony', 'caramel', 'golden', 'beige',
    ],
}

_interrogator_instance = None


def _get_interrogator():
    global _interrogator_instance
    if _interrogator_instance is None:
        from extras.interrogate import Interrogator
        _interrogator_instance = Interrogator()
    return _interrogator_instance


def _caption_image(img_rgb) -> str:
    """Run BLIP captioning on an RGB image (numpy array or PIL)."""
    from PIL import Image
    import numpy as np

    if img_rgb is None:
        return ''

    if isinstance(img_rgb, np.ndarray):
        pil_img = Image.fromarray(img_rgb).convert('RGB')
    elif isinstance(img_rgb, Image.Image):
        pil_img = img_rgb.convert('RGB')
    else:
        return ''

    try:
        interrogator = _get_interrogator()
        caption = interrogator.interrogate(pil_img)
        return caption.strip() if caption else ''
    except Exception as e:
        print(f'[Freyra] Image interrogation failed: {e}')
        return ''


def _extract_relevant_phrases(caption: str, dimension: str) -> str:
    """Filter a BLIP caption to keep only phrases relevant to a dimension."""
    keywords = DIMENSION_KEYWORDS.get(dimension, [])
    if not keywords:
        return caption

    caption_lower = caption.lower()
    relevant_parts = []

    segments = re.split(r'[,;]+', caption)
    for segment in segments:
        seg_lower = segment.strip().lower()
        if any(kw in seg_lower for kw in keywords):
            relevant_parts.append(segment.strip())

    if relevant_parts:
        return ', '.join(relevant_parts)

    for kw in keywords:
        if kw in caption_lower:
            return caption

    return caption


def describe_for_dimension(image, dimension_name: str) -> str:
    """Extract a dimension-relevant description from a reference image.

    Args:
        image: numpy array or PIL Image
        dimension_name: one of the keys in DIMENSION_KEYWORDS
            (outfit, pose, background, lighting, makeup, expression,
             hair_style, hair_color, camera_angle, footwear, skin_tone)

    Returns:
        A text description relevant to that dimension, or empty string on failure.
    """
    if image is None:
        return ''

    caption = _caption_image(image)
    if not caption:
        return ''

    return _extract_relevant_phrases(caption, dimension_name)
