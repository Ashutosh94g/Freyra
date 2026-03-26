"""Freyra Smart Gallery Page -- browse, filter, and export generated images."""

import os
import gradio as gr

import modules.config
from modules.export_presets import (
    EXPORT_FORMATS, EXPORT_FORMAT_LABELS, crop_for_platform, export_with_exif,
)


def _format_key_from_label(label: str) -> str | None:
    for key, fmt in EXPORT_FORMATS.items():
        if fmt['label'] == label:
            return key
    return None


def _do_export(gallery_selection, export_label):
    """Export selected gallery image with platform cropping + EXIF."""
    if gallery_selection is None:
        return 'No image selected. Click an image in the gallery first.', None

    format_key = _format_key_from_label(export_label)
    if format_key is None:
        return f'Unknown format: {export_label}', None

    img_path = None
    if isinstance(gallery_selection, str):
        img_path = gallery_selection
    elif isinstance(gallery_selection, dict):
        img_path = gallery_selection.get('name') or gallery_selection.get('path')
    elif isinstance(gallery_selection, (list, tuple)) and len(gallery_selection) > 0:
        first = gallery_selection[0]
        if isinstance(first, str):
            img_path = first
        elif isinstance(first, dict):
            img_path = first.get('name') or first.get('path')

    if img_path is None or not os.path.isfile(str(img_path)):
        return 'Could not locate the selected image file.', None

    from PIL import Image
    src_img = Image.open(img_path)

    export_dir = os.path.join(modules.config.path_outputs, 'exports')
    os.makedirs(export_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(img_path))[0]
    out_name = f'{base}_{format_key}.jpg'
    out_path = os.path.join(export_dir, out_name)

    exported = export_with_exif(src_img, format_key, output_path=out_path)
    if not os.path.isfile(out_path):
        exported.save(out_path, quality=95)

    fmt = EXPORT_FORMATS[format_key]
    return (
        f'Exported {fmt["label"]} ({fmt["width"]}x{fmt["height"]}) to exports/{out_name}',
        out_path,
    )


def build_gallery_tab():
    """Build the Smart Gallery tab UI."""
    gr.Markdown('### Smart Gallery')
    gr.Markdown('Browse, compare, and export your generated images.')

    with gr.Row():
        gallery_view = gr.Gallery(
            label='All Generated Images',
            show_label=False,
            object_fit='contain',
            height=500,
            format='png',
            show_download_button=True,
        )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown('**Export**')
            export_format = gr.Dropdown(
                label='Export Format',
                choices=EXPORT_FORMAT_LABELS,
                value=EXPORT_FORMAT_LABELS[0],
                interactive=True,
            )
            export_btn = gr.Button('Export Selected', variant='primary', size='sm')
            export_status = gr.Textbox(label='Export Status', interactive=False, lines=1)
            export_download = gr.File(label='Download', visible=False)

        with gr.Column(scale=1):
            gr.Markdown('**Compare**')
            compare_btn = gr.Button('Compare Side-by-Side', variant='secondary', size='sm')
            with gr.Row():
                compare_img_1 = gr.Image(label='Image A', height=200)
                compare_img_2 = gr.Image(label='Image B', height=200)

    export_btn.click(
        fn=_do_export,
        inputs=[gallery_view, export_format],
        outputs=[export_status, export_download],
    ).then(
        fn=lambda path: gr.update(visible=path is not None, value=path),
        inputs=[export_download],
        outputs=[export_download],
    )

    def _compare(gallery_data):
        if gallery_data is None or len(gallery_data) < 2:
            return None, None
        first = gallery_data[0]
        second = gallery_data[1]
        path_a = first if isinstance(first, str) else (first.get('name') if isinstance(first, dict) else None)
        path_b = second if isinstance(second, str) else (second.get('name') if isinstance(second, dict) else None)
        return path_a, path_b

    compare_btn.click(
        fn=_compare,
        inputs=[gallery_view],
        outputs=[compare_img_1, compare_img_2],
        queue=False, show_progress='hidden',
    )

    return {
        'gallery': gallery_view,
        'export_format': export_format,
        'export_btn': export_btn,
        'export_status': export_status,
        'export_download': export_download,
        'compare_btn': compare_btn,
        'compare_img_1': compare_img_1,
        'compare_img_2': compare_img_2,
    }
