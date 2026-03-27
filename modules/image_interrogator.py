"""Freyra Image Interrogator -- extract per-dimension descriptions from reference images.

Uses Moondream2, a lightweight VLM (~1.7B params, ~3.5GB fp16), to answer
targeted questions about each creative dimension. Instead of captioning then
keyword-matching (the old BLIP approach), we ask dimension-specific questions
like "Describe the lighting in this photo" and get usable prompt fragments.

VRAM: Safe on T4 because interrogation runs before SDXL loads. The model is
unloaded from GPU after each call via the existing model_management system.
Falls back to BLIP captioning if Moondream cannot be loaded.
"""

import re

DIMENSION_QUESTIONS = {
    'outfit': (
        "Describe exactly what this person is wearing. Include colors, fabrics, "
        "and style details. Be concise, use comma-separated descriptors."
    ),
    'lighting': (
        "Describe the lighting in this photo: direction, quality, color temperature, "
        "and mood. Be concise, use comma-separated descriptors."
    ),
    'pose': (
        "Describe this person's body pose and positioning in detail. "
        "Be concise, use comma-separated descriptors."
    ),
    'background': (
        "Describe the background setting and environment in this photo. "
        "Be concise, use comma-separated descriptors."
    ),
    'makeup': (
        "Describe the makeup this person is wearing, including lip color, "
        "eye makeup, and skin finish. Be concise, use comma-separated descriptors."
    ),
    'expression': (
        "Describe this person's facial expression and mood in a few words."
    ),
    'hair_style': (
        "Describe this person's hairstyle: cut, texture, and how it is styled. "
        "Be concise, use comma-separated descriptors."
    ),
    'hair_color': (
        "What is this person's hair color? Be specific about shade and any "
        "highlights or gradients. Answer in a few words only."
    ),
    'hair_length': (
        "Describe this person's hair length in a few words "
        "(e.g. cropped, chin length, shoulder length, waist length)."
    ),
    'camera_angle': (
        "What camera angle and framing is used in this photo? Is it a close-up, "
        "medium shot, full body, high angle, low angle, etc? Answer in a few words."
    ),
    'footwear': (
        "Describe the shoes or footwear visible in this image. "
        "Include type, color, and style. Be concise."
    ),
    'skin_tone': (
        "Describe this person's skin tone in a few words "
        "(e.g. fair porcelain, medium olive, dark brown, deep ebony)."
    ),
}

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
    'hair_length': [
        'cropped', 'buzz', 'short', 'ear length', 'chin length', 'bob',
        'shoulder length', 'medium', 'long', 'waist length', 'hip length',
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


_moondream_model = None
_moondream_tokenizer = None
_use_blip_fallback = False


def _load_moondream():
    """Lazy-load Moondream2. Returns (model, tokenizer) or raises on failure."""
    global _moondream_model, _moondream_tokenizer

    if _moondream_model is not None:
        return _moondream_model, _moondream_tokenizer

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print('[Freyra] Loading Moondream2 VLM for image understanding...')

    device = 'cpu'
    dtype = torch.float32
    try:
        import ldm_patched.modules.model_management as mm
        if torch.cuda.is_available():
            device = mm.text_encoder_device()
            if mm.should_use_fp16(device=device):
                dtype = torch.float16
    except Exception:
        if torch.cuda.is_available():
            device = 'cuda'
            dtype = torch.float16

    _moondream_model = AutoModelForCausalLM.from_pretrained(
        'vikhyatk/moondream2',
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map={'': device},
    )
    _moondream_tokenizer = AutoTokenizer.from_pretrained(
        'vikhyatk/moondream2',
        trust_remote_code=True,
    )

    print(f'[Freyra] Moondream2 loaded on {device} ({dtype})')
    return _moondream_model, _moondream_tokenizer


def _unload_moondream():
    """Move Moondream to CPU / free VRAM for the generation pipeline."""
    global _moondream_model
    if _moondream_model is None:
        return
    try:
        import torch
        _moondream_model.to('cpu')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _query_moondream(pil_img, question: str) -> str:
    """Ask Moondream2 a question about an image. Returns the answer string."""
    model, tokenizer = _load_moondream()

    enc_image = model.encode_image(pil_img)
    answer = model.answer_question(enc_image, question, tokenizer)

    _unload_moondream()
    return answer.strip() if answer else ''


def _blip_fallback(pil_img) -> str:
    """Fall back to BLIP captioning when Moondream is unavailable."""
    try:
        from extras.interrogate import Interrogator
        interrogator = Interrogator()
        caption = interrogator.interrogate(pil_img)
        return caption.strip() if caption else ''
    except Exception as e:
        print(f'[Freyra] BLIP fallback also failed: {e}')
        return ''


def _clean_response(text: str) -> str:
    """Strip conversational filler from VLM output to get a clean prompt fragment."""
    text = text.strip()
    prefixes_to_strip = [
        'the person is wearing ', 'she is wearing ', 'they are wearing ',
        'the lighting is ', 'the background is ', 'the person has ',
        'she has ', 'the makeup is ', 'the expression is ',
        'the hairstyle is ', 'the hair color is ', 'the hair length is ',
        'the camera angle is ', 'the footwear is ', 'the skin tone is ',
        'in this photo, ', 'in this image, ', 'this photo shows ',
        'the image shows ', 'i can see ',
    ]
    text_lower = text.lower()
    for prefix in prefixes_to_strip:
        if text_lower.startswith(prefix):
            text = text[len(prefix):]
            text_lower = text.lower()

    text = text.rstrip('.')
    return text.strip()


def _extract_relevant_phrases(caption: str, dimension: str) -> str:
    """Filter a caption to keep only phrases relevant to a dimension.
    Used as post-processing for both Moondream and BLIP outputs.
    """
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

    Uses Moondream2 to answer a targeted question for the given dimension.
    Falls back to BLIP captioning + keyword extraction if Moondream fails.

    Args:
        image: numpy array or PIL Image
        dimension_name: one of the keys in DIMENSION_QUESTIONS

    Returns:
        A text description relevant to that dimension, or empty string on failure.
    """
    global _use_blip_fallback

    if image is None:
        return ''

    from PIL import Image
    import numpy as np

    if isinstance(image, np.ndarray):
        pil_img = Image.fromarray(image).convert('RGB')
    elif isinstance(image, Image.Image):
        pil_img = image.convert('RGB')
    else:
        return ''

    question = DIMENSION_QUESTIONS.get(dimension_name)

    if not _use_blip_fallback and question:
        try:
            answer = _query_moondream(pil_img, question)
            if answer:
                return _clean_response(answer)
        except Exception as e:
            print(f'[Freyra] Moondream2 failed, falling back to BLIP: {e}')
            _use_blip_fallback = True

    caption = _blip_fallback(pil_img)
    if not caption:
        return ''
    return _extract_relevant_phrases(caption, dimension_name)
