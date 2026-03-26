"""Freyra Smart Gallery Page -- browse, filter, and export generated images."""

import gradio as gr

from modules.export_presets import EXPORT_FORMAT_LABELS


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

        with gr.Column(scale=1):
            gr.Markdown('**Compare**')
            compare_btn = gr.Button('Compare Side-by-Side', variant='secondary', size='sm')
            with gr.Row():
                compare_img_1 = gr.Image(label='Image A', height=200)
                compare_img_2 = gr.Image(label='Image B', height=200)

    return {
        'gallery': gallery_view,
        'export_format': export_format,
        'export_btn': export_btn,
        'export_status': export_status,
        'compare_btn': compare_btn,
        'compare_img_1': compare_img_1,
        'compare_img_2': compare_img_2,
    }
