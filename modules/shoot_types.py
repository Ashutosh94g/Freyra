"""Freyra Shoot Types -- opinionated generation presets.

Each shoot type bundles: prompt template, negative prompt, CFG, LoRAs,
lighting bias, aspect ratio, and camera EXIF profile into a single
creative decision.
"""

SHOOT_TYPES = {
    "fashion_editorial": {
        "label": "Fashion Editorial",
        "description": "High-fashion magazine editorial",
        "prompt_template": (
            "editorial fashion photograph of {subject}, high-fashion magazine spread, "
            "professional studio lighting, designer clothing, clean background, "
            "sharp detail, Vogue aesthetic, Canon EOS R5, 85mm lens, f/2.0, "
            "natural skin texture, professional retouching"
        ),
        "negative_prompt": (
            "casual, low quality, blurry, amateur, cartoon, watermark, text, "
            "deformed, bad anatomy, extra limbs, poorly drawn face, disfigured, "
            "mutated, bad proportions, oversaturated, flat lighting"
        ),
        "cfg_scale": 4.5,
        "sharpness": 2.0,
        "loras": [
            ("SDXL_FILM_PHOTOGRAPHY_STYLE_V1.safetensors", 0.2),
            ("add-detail-xl.safetensors", 0.35),
        ],
        "aspect_ratio": "896*1152",
        "camera_exif": "Canon EOS R5",
        "lighting_bias": "studio softbox with rim light",
        "styles": ["Freyra V2", "SAI Photographic", "Freyra Negative"],
    },
    "gym_fitness": {
        "label": "Gym / Fitness",
        "description": "Athletic gym photoshoot",
        "prompt_template": (
            "photograph of {subject}, gym fitness photoshoot, athletic wear, "
            "professional gym lighting, high-end fitness studio, sharp focus, "
            "Canon EOS R5, 85mm lens, f/2.0, natural skin texture, "
            "motivated expression, dynamic energy"
        ),
        "negative_prompt": (
            "cartoon, anime, painting, 3d render, watermark, text, deformed, "
            "bad anatomy, extra limbs, poorly lit, flat lighting, amateur, "
            "blurry, low quality, disfigured"
        ),
        "cfg_scale": 4.5,
        "sharpness": 2.5,
        "loras": [
            ("add-detail-xl.safetensors", 0.4),
            ("epiCPhoto.safetensors", 0.25),
        ],
        "aspect_ratio": "896*1152",
        "camera_exif": "Canon EOS R5",
        "lighting_bias": "harsh gym lighting with overhead spots",
        "styles": ["Freyra V2", "SAI Photographic", "Freyra Negative"],
    },
    "beach_swimwear": {
        "label": "Beach / Swimwear",
        "description": "Tropical beach photoshoot",
        "prompt_template": (
            "beach photograph of {subject}, summer beachwear photoshoot, "
            "tropical beach background, golden sand, turquoise water, "
            "warm sunlight, swimwear editorial, professional beach photography, "
            "Canon EOS R5, 85mm, natural skin texture"
        ),
        "negative_prompt": (
            "indoor, cold, winter, watermark, text, deformed, bad anatomy, "
            "cartoon, 3d render, blurry, poorly lit, low quality, disfigured, "
            "extra limbs, bad proportions"
        ),
        "cfg_scale": 4.0,
        "sharpness": 2.0,
        "loras": [
            ("SDXL_FILM_PHOTOGRAPHY_STYLE_V1.safetensors", 0.3),
            ("add-detail-xl.safetensors", 0.3),
        ],
        "aspect_ratio": "896*1152",
        "camera_exif": "Canon EOS R5",
        "lighting_bias": "golden hour warm sunlight",
        "styles": ["Freyra V2", "SAI Photographic", "Freyra Negative"],
    },
    "instagram_lifestyle": {
        "label": "Instagram Lifestyle",
        "description": "Candid Instagram aesthetic",
        "prompt_template": (
            "lifestyle photograph of {subject}, candid Instagram aesthetic, "
            "golden hour natural light, warm color grade, casual chic, "
            "shot on iPhone 15 Pro Max, shallow depth of field, "
            "natural skin texture, authentic vibe"
        ),
        "negative_prompt": (
            "studio, harsh lighting, formal, watermark, text, deformed, "
            "bad anatomy, cartoon, 3d render, blurry, low quality, "
            "oversaturated, fake, plastic skin"
        ),
        "cfg_scale": 4.0,
        "sharpness": 1.5,
        "loras": [
            ("SDXL_FILM_PHOTOGRAPHY_STYLE_V1.safetensors", 0.35),
            ("add-detail-xl.safetensors", 0.25),
        ],
        "aspect_ratio": "896*1152",
        "camera_exif": "iPhone 15 Pro Max",
        "lighting_bias": "golden hour natural light",
        "styles": ["Freyra V2", "SAI Photographic", "Freyra Negative"],
    },
    "yoga_wellness": {
        "label": "Yoga / Wellness",
        "description": "Serene yoga and wellness",
        "prompt_template": (
            "yoga lifestyle photograph of {subject}, wellness and mindfulness shoot, "
            "soft morning light, serene setting, yoga pose, earthy tones, "
            "shallow depth of field, Sony A7IV, 85mm f/1.8, "
            "natural skin texture, peaceful expression"
        ),
        "negative_prompt": (
            "gym equipment, urban, dark, harsh lighting, watermark, text, "
            "deformed, bad anatomy, cartoon, 3d render, blurry, low quality, "
            "oversaturated, disfigured"
        ),
        "cfg_scale": 4.0,
        "sharpness": 1.8,
        "loras": [
            ("SDXL_FILM_PHOTOGRAPHY_STYLE_V1.safetensors", 0.3),
            ("add-detail-xl.safetensors", 0.3),
        ],
        "aspect_ratio": "896*1152",
        "camera_exif": "Sony A7IV",
        "lighting_bias": "soft morning light with natural warmth",
        "styles": ["Freyra V2", "SAI Photographic", "Freyra Negative"],
    },
    "luxury_travel": {
        "label": "Luxury / Travel",
        "description": "Five-star luxury aesthetic",
        "prompt_template": (
            "luxury lifestyle photograph of {subject}, five-star hotel suite, "
            "luxury travel influencer aesthetic, floor-to-ceiling windows, "
            "elegant outfit, warm interior lighting, aspirational, "
            "Sony A7IV 35mm, natural skin texture"
        ),
        "negative_prompt": (
            "outdoor nature, gym, beach, watermark, text, deformed, "
            "bad anatomy, cartoon, 3d render, blurry, low quality, "
            "amateur, poorly lit, disfigured"
        ),
        "cfg_scale": 4.5,
        "sharpness": 2.0,
        "loras": [
            ("SDXL_FILM_PHOTOGRAPHY_STYLE_V1.safetensors", 0.25),
            ("add-detail-xl.safetensors", 0.35),
        ],
        "aspect_ratio": "896*1152",
        "camera_exif": "Sony A7IV",
        "lighting_bias": "warm interior lighting with window light",
        "styles": ["Freyra V2", "SAI Photographic", "Freyra Negative"],
    },
    "streetwear_urban": {
        "label": "Streetwear / Urban",
        "description": "Urban street fashion",
        "prompt_template": (
            "street fashion photograph of {subject}, urban street style, "
            "city background, contemporary fashion district, editorial composition, "
            "natural overcast light, Leica Q3, sharp detail, "
            "natural skin texture, confident attitude"
        ),
        "negative_prompt": (
            "beach, gym, studio, tropical, watermark, text, deformed, "
            "bad anatomy, cartoon, 3d render, blurry, low quality, "
            "amateur, disfigured, bad proportions"
        ),
        "cfg_scale": 4.5,
        "sharpness": 2.2,
        "loras": [
            ("epiCPhoto.safetensors", 0.3),
            ("add-detail-xl.safetensors", 0.35),
        ],
        "aspect_ratio": "896*1152",
        "camera_exif": "Canon EOS R5",
        "lighting_bias": "natural overcast outdoor light",
        "styles": ["Freyra V2", "SAI Photographic", "Freyra Negative"],
    },
    "beauty_skincare": {
        "label": "Beauty / Skincare",
        "description": "Close-up beauty editorial",
        "prompt_template": (
            "beauty photograph of {subject}, skincare editorial, clean dewy skin, "
            "neutral studio background, soft diffused lighting, beauty brand aesthetic, "
            "Sony A7IV 100mm macro, extreme detail, skin texture visible, "
            "natural pores, professional beauty retouching"
        ),
        "negative_prompt": (
            "heavy makeup, dramatic, dark, outdoor, gym, watermark, text, "
            "deformed, bad anatomy, cartoon, 3d render, blurry, "
            "plastic skin, airbrushed, low quality"
        ),
        "cfg_scale": 5.0,
        "sharpness": 3.0,
        "loras": [
            ("add-detail-xl.safetensors", 0.45),
            ("DetailTweaker_xl.safetensors", 0.3),
        ],
        "aspect_ratio": "1024*1024",
        "camera_exif": "Sony A7IV",
        "lighting_bias": "soft diffused studio lighting with catchlights",
        "styles": ["Freyra V2", "SAI Photographic", "Freyra Negative"],
    },
    "activewear": {
        "label": "Activewear",
        "description": "Sports brand lookbook",
        "prompt_template": (
            "activewear photograph of {subject}, premium activewear editorial, "
            "sports brand lookbook, dynamic energy, razor-sharp focus, "
            "Nike aesthetic, Canon EOS R5, natural skin texture, "
            "athletic build, energetic expression"
        ),
        "negative_prompt": (
            "casual clothes, formal wear, watermark, text, deformed, "
            "bad anatomy, cartoon, 3d render, blurry, low quality, "
            "amateur, disfigured, poorly lit"
        ),
        "cfg_scale": 4.5,
        "sharpness": 2.5,
        "loras": [
            ("epiCPhoto.safetensors", 0.3),
            ("add-detail-xl.safetensors", 0.35),
        ],
        "aspect_ratio": "896*1152",
        "camera_exif": "Canon EOS R5",
        "lighting_bias": "crisp bright outdoor light",
        "styles": ["Freyra V2", "SAI Photographic", "Freyra Negative"],
    },
    "night_evening": {
        "label": "Night / Evening",
        "description": "Evening and nightlife aesthetic",
        "prompt_template": (
            "evening photograph of {subject}, nightlife aesthetic, city lights, "
            "dramatic lighting, evening outfit, urban night scene, "
            "neon reflections, Canon EOS R5 50mm f/1.4, "
            "natural skin texture, glamorous mood"
        ),
        "negative_prompt": (
            "daytime, indoor studio, gym, beach, watermark, text, deformed, "
            "bad anatomy, cartoon, 3d render, blurry, low quality, "
            "amateur, disfigured, overexposed"
        ),
        "cfg_scale": 5.0,
        "sharpness": 2.0,
        "loras": [
            ("SDXL_FILM_PHOTOGRAPHY_STYLE_V1.safetensors", 0.3),
            ("add-detail-xl.safetensors", 0.35),
        ],
        "aspect_ratio": "896*1152",
        "camera_exif": "Canon EOS R5",
        "lighting_bias": "neon ambient light with dramatic shadows",
        "styles": ["Freyra V2", "SAI Photographic", "Freyra Negative"],
    },
    "casual_coffee": {
        "label": "Casual / Coffee",
        "description": "Cafe and brunch aesthetic",
        "prompt_template": (
            "lifestyle photograph of {subject}, trendy cafe brunch aesthetic, "
            "cozy coffee shop setting, natural window light, warm bokeh background, "
            "Instagram food blogging vibes, iPhone 15 Pro Max, "
            "natural skin texture, relaxed smile"
        ),
        "negative_prompt": (
            "outdoor nature, gym, beach, watermark, text, deformed, "
            "bad anatomy, cartoon, 3d render, dark lighting, "
            "low quality, amateur, disfigured"
        ),
        "cfg_scale": 4.0,
        "sharpness": 1.5,
        "loras": [
            ("SDXL_FILM_PHOTOGRAPHY_STYLE_V1.safetensors", 0.35),
            ("add-detail-xl.safetensors", 0.25),
        ],
        "aspect_ratio": "896*1152",
        "camera_exif": "iPhone 15 Pro Max",
        "lighting_bias": "warm natural window light with bokeh",
        "styles": ["Freyra V2", "SAI Photographic", "Freyra Negative"],
    },
    "pool_summer": {
        "label": "Pool / Summer",
        "description": "Poolside luxury aesthetic",
        "prompt_template": (
            "pool lifestyle photograph of {subject}, luxury pool day shoot, "
            "infinity pool with ocean view, designer swimwear, afternoon sun, "
            "aspirational influencer aesthetic, Canon EOS R5, "
            "natural skin texture, sun-kissed glow"
        ),
        "negative_prompt": (
            "gym, indoor, winter, cold, watermark, text, deformed, "
            "bad anatomy, cartoon, 3d render, blurry, poorly lit, "
            "low quality, amateur, disfigured"
        ),
        "cfg_scale": 4.0,
        "sharpness": 2.0,
        "loras": [
            ("SDXL_FILM_PHOTOGRAPHY_STYLE_V1.safetensors", 0.3),
            ("add-detail-xl.safetensors", 0.3),
        ],
        "aspect_ratio": "896*1152",
        "camera_exif": "Canon EOS R5",
        "lighting_bias": "afternoon sun with warm golden tones",
        "styles": ["Freyra V2", "SAI Photographic", "Freyra Negative"],
    },
}

SHOOT_TYPE_KEYS = list(SHOOT_TYPES.keys())
SHOOT_TYPE_LABELS = [st["label"] for st in SHOOT_TYPES.values()]

QUALITY_MODES = {
    "fast": {
        "label": "Fast Preview",
        "performance": "Lightning",
        "steps": 4,
    },
    "standard": {
        "label": "Standard",
        "performance": "Speed",
        "steps": 30,
    },
    "ultra": {
        "label": "Ultra",
        "performance": "Quality",
        "steps": 60,
    },
}

QUALITY_MODE_LABELS = [v["label"] for v in QUALITY_MODES.values()]
QUALITY_MODE_KEYS = list(QUALITY_MODES.keys())


def get_shoot_type(label: str) -> dict | None:
    for st in SHOOT_TYPES.values():
        if st["label"] == label:
            return st
    return None


def get_shoot_type_key(label: str) -> str | None:
    for key, st in SHOOT_TYPES.items():
        if st["label"] == label:
            return key
    return None


def get_quality_mode(label: str) -> dict | None:
    for qm in QUALITY_MODES.values():
        if qm["label"] == label:
            return qm
    return None
