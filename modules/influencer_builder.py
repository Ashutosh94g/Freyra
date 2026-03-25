import os
import modules.config

BUILDER_CATEGORIES = [
    ('skin_tone', 'skin_tones.txt'),
    ('hair', 'influencer_hair.txt'),
    ('outfit', 'influencer_outfits.txt'),
    ('posture', 'influencer_poses.txt'),
    ('makeup', 'influencer_makeup.txt'),
    ('background', 'influencer_settings.txt'),
    ('lighting', 'influencer_lighting.txt'),
]

CATEGORY_LABELS = {
    'skin_tone': 'Skin Tone',
    'hair': 'Hair Style',
    'outfit': 'Outfit / Dress',
    'posture': 'Posture / Pose',
    'makeup': 'Makeup & Expression',
    'background': 'Background / Setting',
    'lighting': 'Lighting',
}

NONE_OPTION = 'None'


def load_dropdown_options(wildcard_filename):
    """Read a wildcard .txt file and return its lines as dropdown choices, prepending 'None'."""
    filepath = os.path.join(modules.config.path_wildcards, wildcard_filename)
    try:
        lines = open(filepath, encoding='utf-8').read().splitlines()
        lines = [x.strip() for x in lines if x.strip()]
    except FileNotFoundError:
        lines = []
    return [NONE_OPTION] + lines


def assemble_builder_prompt(**values):
    """Take category values and assemble a structured prompt string.

    Args:
        skin_tone, hair, outfit, posture, makeup, background, lighting:
            Each can be a string or empty/None to skip.

    Returns:
        Assembled prompt string with non-empty segments joined by commas.
    """
    segments = []
    for key, _filename in BUILDER_CATEGORIES:
        val = values.get(key, '')
        if not val or val == NONE_OPTION:
            continue
        if key == 'outfit':
            segments.append(f'wearing {val}')
        elif key == 'background':
            segments.append(f'in {val}')
        else:
            segments.append(val)
    return ', '.join(segments)


def get_effective_value(dropdown_val, text_override):
    """Return text_override if non-empty, otherwise dropdown_val."""
    if text_override and text_override.strip():
        return text_override.strip()
    return dropdown_val if dropdown_val != NONE_OPTION else ''
