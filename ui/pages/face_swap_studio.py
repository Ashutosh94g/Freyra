"""Freyra Face Swap Studio -- swap your character's face onto any image.

Supports fetching target images from Instagram post URLs, direct image URLs,
or manual upload. Uses the enhanced face swap pipeline with parsing masks,
color matching, Poisson blending, and GFPGAN restoration.
"""

import numpy as np
import gradio as gr

from modules.character_profiles import list_profile_names, load_profile


def _fetch_url_image(url: str):
    """Fetch an image from an Instagram post URL or direct image URL."""
    if not url or not url.strip():
        return None, 'Please enter a URL.'

    from modules.instagram_fetcher import fetch_image_from_url
    image, message = fetch_image_from_url(url)
    return image, message


def _load_character_faces(character_name):
    """Load face images from a saved character profile."""
    if not character_name or character_name == 'None':
        return None, None, None
    images = load_profile(character_name)
    if images is None:
        return None, None, None
    result = [None, None, None]
    for i, img in enumerate(images[:3]):
        result[i] = img
    return result[0], result[1], result[2]


def _run_face_swap(
    target_image, face_1, face_2, face_3,
    do_color_match, do_face_restore, do_enhanced_blending,
):
    """Execute the face swap pipeline."""
    if target_image is None:
        return None, 'No target image. Upload one or fetch from a URL.'

    face_refs = [f for f in [face_1, face_2, face_3] if f is not None]
    if not face_refs:
        return None, 'No face reference provided. Upload face photos or select a character.'

    source_face = face_refs[0]

    from modules.face_swap import swap_face, is_available

    if not is_available():
        return None, ('Face swap model not available. '
                      'Ensure inswapper_128.onnx is in models/insightface/')

    result = swap_face(
        source_img=source_face,
        target_img=target_image,
        enhance_blend=True,
        color_match=bool(do_color_match),
        face_restore=bool(do_face_restore),
        enhanced_blending=bool(do_enhanced_blending),
    )

    if result is None:
        return None, (
            'Face swap failed. Check the terminal logs for details. '
            'Common causes: no face detected in source or target image.'
        )

    return result, 'Face swap completed successfully!'


def build_face_swap_tab():
    """Build the Face Swap Studio tab UI.

    Returns a dict of component references for external wiring.
    """
    gr.Markdown('### Face Swap Studio')
    gr.Markdown(
        'Swap your character\'s face onto any photo. '
        'Paste an Instagram link, a direct image URL, or upload a target image.'
    )

    with gr.Row():
        # -- Left: Target image input --
        with gr.Column(scale=1):
            gr.Markdown('**Target Image**')

            url_input = gr.Textbox(
                label='Image URL',
                placeholder='Instagram post URL or direct image URL (jpg/png)',
                lines=1, max_lines=1,
            )
            fetch_btn = gr.Button(
                'Fetch Image',
                variant='secondary', size='sm',
            )
            fetch_status = gr.Textbox(
                label='Status', interactive=False,
                lines=1, max_lines=4, visible=True,
            )

            gr.Markdown('**-- or upload directly --**')
            target_image = gr.Image(
                label='Target Image',
                type='numpy',
                height=300,
                sources=['upload', 'clipboard'],
            )

        # -- Right: Face reference --
        with gr.Column(scale=1):
            gr.Markdown('**Face Reference**')
            character_select = gr.Dropdown(
                label='Load Saved Character',
                choices=list_profile_names(),
                value='None',
                interactive=True,
            )
            gr.Markdown('Or upload face reference photos:')
            with gr.Row():
                swap_face_1 = gr.Image(
                    label='Face 1', type='numpy',
                    height=120, sources=['upload'],
                )
                swap_face_2 = gr.Image(
                    label='Face 2', type='numpy',
                    height=120, sources=['upload'],
                )
                swap_face_3 = gr.Image(
                    label='Face 3', type='numpy',
                    height=120, sources=['upload'],
                )

    with gr.Row():
        with gr.Column():
            gr.Markdown('**Quality Settings**')
            with gr.Row():
                do_color_match = gr.Checkbox(
                    label='Color Matching',
                    value=True,
                    info='Match swapped face lighting to target',
                )
                do_face_restore = gr.Checkbox(
                    label='Face Restoration (GFPGAN)',
                    value=True,
                    info='Restore facial details after swap (downloads ~340MB model on first use)',
                )
                do_enhanced_blending = gr.Checkbox(
                    label='Multi-Pass Swap',
                    value=True,
                    info='Run 2-pass swap for stronger identity transfer',
                )

    with gr.Row():
        swap_btn = gr.Button(
            'Swap Face',
            variant='primary',
            elem_classes=['freyra-generate-btn'],
            scale=2,
        )
        swap_status = gr.Textbox(
            label='Result', interactive=False,
            lines=1, max_lines=3, scale=3,
        )

    result_image = gr.Image(
        label='Result', type='numpy',
        height=500, interactive=False,
    )

    # -- Wire events --
    fetch_btn.click(
        fn=_fetch_url_image,
        inputs=[url_input],
        outputs=[target_image, fetch_status],
        show_progress='full',
    )

    character_select.change(
        fn=_load_character_faces,
        inputs=[character_select],
        outputs=[swap_face_1, swap_face_2, swap_face_3],
        queue=False, show_progress='hidden',
    )

    swap_btn.click(
        fn=_run_face_swap,
        inputs=[
            target_image,
            swap_face_1, swap_face_2, swap_face_3,
            do_color_match, do_face_restore, do_enhanced_blending,
        ],
        outputs=[result_image, swap_status],
        show_progress='full',
    )

    return {
        'url_input': url_input,
        'fetch_btn': fetch_btn,
        'fetch_status': fetch_status,
        'target_image': target_image,
        'character': character_select,
        'face_1': swap_face_1,
        'face_2': swap_face_2,
        'face_3': swap_face_3,
        'do_color_match': do_color_match,
        'do_face_restore': do_face_restore,
        'do_enhanced_blending': do_enhanced_blending,
        'swap_btn': swap_btn,
        'swap_status': swap_status,
        'result_image': result_image,
    }
