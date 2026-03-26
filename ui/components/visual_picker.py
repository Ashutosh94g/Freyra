"""Freyra Visual Picker -- thumbnail card grid for option selection.

Replaces plain gr.Dropdown with a clickable visual grid of option cards.
Each card shows a color-coded placeholder (or real thumbnail when available)
plus the option text. JavaScript wires clicks to update a hidden gr.Textbox.
"""

import os
import hashlib
import gradio as gr

THUMBNAIL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'assets', 'thumbnails')

CATEGORY_PALETTES = {
    'influencer_outfits': [
        ('linear-gradient(135deg, #2d2d2d, #4a4a4a)', '#e0e0e0'),
        ('linear-gradient(135deg, #f5e6d3, #d4a574)', '#3a2a1a'),
        ('linear-gradient(135deg, #e8d5d5, #c4a0a0)', '#4a2a2a'),
        ('linear-gradient(135deg, #d5e8d5, #a0c4a0)', '#2a4a2a'),
        ('linear-gradient(135deg, #d5d5e8, #a0a0c4)', '#2a2a4a'),
    ],
    'influencer_poses': [
        ('linear-gradient(135deg, #3a3a4a, #5a5a7a)', '#c0c0e0'),
    ],
    'influencer_settings': [
        ('linear-gradient(135deg, #2a3a2a, #4a6a4a)', '#d0e0d0'),
        ('linear-gradient(135deg, #3a3a2a, #6a6a4a)', '#e0e0d0'),
        ('linear-gradient(135deg, #2a3a3a, #4a6a6a)', '#d0e0e0'),
    ],
    'influencer_lighting': [
        ('linear-gradient(135deg, #c4852e, #e8b86d)', '#1a1a1a'),
        ('linear-gradient(135deg, #4a5a6a, #7a8a9a)', '#1a1a2a'),
        ('linear-gradient(135deg, #6a4a6a, #9a7a9a)', '#1a1a1a'),
    ],
    'influencer_makeup': [
        ('linear-gradient(135deg, #e8c0c0, #d4a0a0)', '#4a2a2a'),
        ('linear-gradient(135deg, #c4852e, #e8b86d)', '#2a1a1a'),
    ],
    'influencer_expressions': [
        ('linear-gradient(135deg, #3a3a3a, #5a5a5a)', '#e0e0e0'),
    ],
    'influencer_hair': [
        ('linear-gradient(135deg, #3a2a1a, #6a4a2a)', '#e0d0c0'),
    ],
    'influencer_camera_angles': [
        ('linear-gradient(135deg, #2a2a3a, #4a4a5a)', '#c0c0d0'),
    ],
    'influencer_footwear': [
        ('linear-gradient(135deg, #3a3a3a, #5a5a5a)', '#d0d0d0'),
    ],
}

NONE_OPTION = 'None (auto)'


def _slug(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:10]


def _get_thumbnail_url(category: str, option: str) -> str | None:
    """Check if a real thumbnail exists for this option."""
    slug = _slug(option)
    for ext in ('jpg', 'png', 'webp'):
        path = os.path.join(THUMBNAIL_DIR, category, f'{slug}.{ext}')
        if os.path.isfile(path):
            return path
    return None


def _get_card_style(category: str, index: int) -> tuple[str, str]:
    """Return (background, text_color) for a card."""
    palettes = CATEGORY_PALETTES.get(category, [
        ('linear-gradient(135deg, #2a2a2a, #3a3a3a)', '#c0c0c0'),
    ])
    bg, fg = palettes[index % len(palettes)]
    return bg, fg


def _build_grid_html(picker_id: str, category: str, options: list[str], selected: str = '') -> str:
    """Build the HTML for the visual picker grid."""
    cards = []

    none_selected = 'selected' if (not selected or selected == NONE_OPTION) else ''
    cards.append(
        f'<div class="picker-card {none_selected}" '
        f'data-value="{NONE_OPTION}" '
        f'onclick="freyraPickerSelect(\'{picker_id}\', this)">'
        f'<div class="picker-card-label">Auto</div>'
        f'</div>'
    )

    for i, opt in enumerate(options):
        if opt == NONE_OPTION:
            continue

        is_selected = 'selected' if opt == selected else ''
        bg, fg = _get_card_style(category, i)

        thumb_path = _get_thumbnail_url(category, opt)
        if thumb_path:
            style = f'background:url(file={thumb_path}) center/cover; color:{fg};'
        else:
            style = f'background:{bg}; color:{fg};'

        short_label = opt if len(opt) <= 35 else opt[:32] + '...'

        cards.append(
            f'<div class="picker-card {is_selected}" '
            f'data-value="{opt}" '
            f'style="{style}" '
            f'title="{opt}" '
            f'onclick="freyraPickerSelect(\'{picker_id}\', this)">'
            f'<div class="picker-card-label">{short_label}</div>'
            f'</div>'
        )

    grid_html = f'<div id="{picker_id}" class="visual-picker-grid">{"".join(cards)}</div>'
    return grid_html


PICKER_JS = """
<script>
function freyraPickerSelect(pickerId, card) {
    var grid = document.getElementById(pickerId);
    if (!grid) return;
    var cards = grid.querySelectorAll('.picker-card');
    cards.forEach(function(c) { c.classList.remove('selected'); });
    card.classList.add('selected');

    var value = card.getAttribute('data-value');
    var hiddenId = pickerId + '_value';
    var hiddenEl = document.querySelector('#' + hiddenId + ' textarea, #' + hiddenId + ' input');
    if (hiddenEl) {
        var nativeInputValueSetter = Object.getOwnPropertyDescriptor(
            window.HTMLTextAreaElement ? window.HTMLTextAreaElement.prototype : window.HTMLInputElement.prototype,
            'value'
        );
        if (!nativeInputValueSetter) {
            nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value');
        }
        if (nativeInputValueSetter && nativeInputValueSetter.set) {
            nativeInputValueSetter.set.call(hiddenEl, value);
        } else {
            hiddenEl.value = value;
        }
        hiddenEl.dispatchEvent(new Event('input', { bubbles: true }));
        hiddenEl.dispatchEvent(new Event('change', { bubbles: true }));
    }
}
</script>
"""

_js_injected = False


def build_visual_picker(
    category: str,
    options: list[str],
    label: str = '',
    default: str = '',
) -> tuple[gr.HTML, gr.Textbox]:
    """Build a visual picker component.

    Returns (picker_html, value_textbox).
    The value_textbox holds the selected option string and triggers .change() events.
    """
    global _js_injected

    picker_id = f'picker_{category}'
    grid_html = _build_grid_html(picker_id, category, options, default)

    js_block = ''
    if not _js_injected:
        js_block = PICKER_JS
        _js_injected = True

    html_component = gr.HTML(
        value=js_block + grid_html,
        label=label,
    )

    value_textbox = gr.Textbox(
        value=default or NONE_OPTION,
        visible=False,
        elem_id=f'{picker_id}_value',
    )

    return html_component, value_textbox
