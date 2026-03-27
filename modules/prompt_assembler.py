"""Freyra Prompt Assembler -- builds optimized prompts from creative dimensions.

Takes user's creative choices (shoot type, pose, outfit, background, etc.)
and assembles them into the best possible prompt + negative prompt +
generation config for photorealistic influencer images.
"""

import os
import random
import modules.config


NONE_OPTION = 'None'

CAMERA_ANGLE_ASPECT_RATIOS = {
    'close-up': '1024*1024',
    'extreme close-up': '1024*1024',
    'face only': '1024*1024',
    'head and shoulders': '1024*1024',
    'portrait': '896*1152',
    'medium shot': '896*1152',
    'waist up': '896*1152',
    'three-quarter': '896*1152',
    'full body': '832*1216',
    'wide shot': '1152*896',
    'wide': '1152*896',
    'landscape': '1152*896',
}


def get_smart_aspect_ratio(camera_angle: str, default: str = '896*1152') -> str:
    """Pick the best aspect ratio for a given camera angle description."""
    if not camera_angle or camera_angle == NONE_OPTION:
        return default
    ca_lower = camera_angle.lower()
    for keyword, ratio in CAMERA_ANGLE_ASPECT_RATIOS.items():
        if keyword in ca_lower:
            return ratio
    return default

DIMENSION_FILES = {
    'skin_tone': 'skin_tones.txt',
    'hair_style': 'influencer_hair.txt',
    'hair_color': 'influencer_hair_colors.txt',
    'outfit': 'influencer_outfits.txt',
    'pose': 'influencer_poses.txt',
    'makeup': 'influencer_makeup.txt',
    'expression': 'influencer_expressions.txt',
    'background': 'influencer_settings.txt',
    'lighting': 'influencer_lighting.txt',
    'camera_angle': 'influencer_camera_angles.txt',
    'footwear': 'influencer_footwear.txt',
}


def load_options(wildcard_filename: str) -> list[str]:
    filepath = os.path.join(modules.config.path_wildcards, wildcard_filename)
    try:
        lines = open(filepath, encoding='utf-8').read().splitlines()
        lines = [x.strip() for x in lines if x.strip()]
    except FileNotFoundError:
        lines = []
    return [NONE_OPTION] + lines


def load_options_no_none(wildcard_filename: str) -> list[str]:
    filepath = os.path.join(modules.config.path_wildcards, wildcard_filename)
    try:
        lines = open(filepath, encoding='utf-8').read().splitlines()
        lines = [x.strip() for x in lines if x.strip()]
    except FileNotFoundError:
        lines = []
    return lines


def resolve_dimension_value(
    dropdown_val: str = '',
    custom_text: str = '',
    image_description: str = '',
) -> str:
    """Resolve the effective value for a creative dimension.

    Priority: image_description > custom_text > dropdown_val.
    Returns empty string if nothing is set.
    """
    if image_description and image_description.strip() and image_description.strip() != NONE_OPTION:
        return image_description.strip()
    if custom_text and custom_text.strip():
        return custom_text.strip()
    if dropdown_val and dropdown_val != NONE_OPTION and dropdown_val.strip():
        return dropdown_val
    return ''


def _is_set(val: str) -> bool:
    return val and val != NONE_OPTION and val.strip() != ''


def assemble_prompt(
    shoot_type_config: dict,
    skin_tone: str = '',
    hair_style: str = '',
    hair_color: str = '',
    outfit: str = '',
    pose: str = '',
    makeup: str = '',
    expression: str = '',
    background: str = '',
    lighting: str = '',
    camera_angle: str = '',
    footwear: str = '',
    custom_prompt: str = '',
) -> dict:
    """Assemble a complete generation config from creative dimensions.

    Returns dict with keys: prompt, negative_prompt, cfg_scale, sharpness,
    styles, aspect_ratio, loras
    """
    subject_parts = []

    if _is_set(skin_tone):
        subject_parts.append(f'{skin_tone} skin')

    subject_parts.append('woman')

    if _is_set(hair_color) and _is_set(hair_style):
        subject_parts.append(f'with {hair_color} {hair_style} hair')
    elif _is_set(hair_style):
        subject_parts.append(f'with {hair_style} hair')
    elif _is_set(hair_color):
        subject_parts.append(f'with {hair_color} hair')

    if _is_set(expression):
        subject_parts.append(f'{expression}')

    subject_description = ' '.join(subject_parts)

    prompt_template = shoot_type_config.get('prompt_template', '{subject}')
    prompt = prompt_template.replace('{subject}', subject_description)

    extra_parts = []

    if _is_set(outfit):
        extra_parts.append(f'wearing {outfit}')

    if _is_set(pose):
        extra_parts.append(pose)

    if _is_set(background):
        bkg = background
        if not bkg.lower().startswith('in '):
            bkg = f'in {bkg}'
        extra_parts.append(bkg)

    if _is_set(lighting):
        extra_parts.append(lighting)

    if _is_set(camera_angle):
        extra_parts.append(f'{camera_angle} shot')

    if _is_set(footwear):
        extra_parts.append(f'{footwear}')

    if extra_parts:
        prompt = prompt + ', ' + ', '.join(extra_parts)

    if _is_set(custom_prompt):
        prompt = prompt + ', ' + custom_prompt

    return {
        'prompt': prompt,
        'negative_prompt': shoot_type_config.get('negative_prompt', ''),
        'cfg_scale': shoot_type_config.get('cfg_scale', 4.5),
        'sharpness': shoot_type_config.get('sharpness', 2.0),
        'styles': shoot_type_config.get('styles', ['Freyra V2', 'SAI Photographic', 'Freyra Negative']),
        'aspect_ratio': shoot_type_config.get('aspect_ratio', '896*1152'),
        'loras': shoot_type_config.get('loras', []),
    }


def randomize_dimensions(options_cache: dict | None = None) -> dict:
    """Pick random values for each creative dimension. Useful for 'Surprise Me' feature."""
    result = {}
    for dim_key, filename in DIMENSION_FILES.items():
        if options_cache and dim_key in options_cache:
            opts = options_cache[dim_key]
        else:
            opts = load_options_no_none(filename)
        if opts:
            result[dim_key] = random.choice(opts)
        else:
            result[dim_key] = ''
    return result
