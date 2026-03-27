"""Freyra Face Swap Studio -- swap your character's face onto any image.

Supports fetching target images from Instagram post URLs or direct upload.
Uses the enhanced face swap pipeline with parsing masks, color matching,
Poisson blending, and GFPGAN restoration.
"""

import numpy as np
import gradio as gr

from modules.character_profiles import list_profile_names, load_profile


def _fetch_ig_image(url: str):
    """Fetch an image from an Instagram post URL."""
    if not url or not url.strip():
        return None, 'Please enter an Instagram post URL.'

    from modules.instagram_fetcher import is_instagram_url, fetch_instagram_image

    if not is_instagram_url(url):
        return None, ('Not a valid Instagram URL. '
                      'Expected format: https://www.instagram.com/p/XXXXX/ '
                      'or /reel/XXXXX/')

    image, message = fetch_instagram_image(url)
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
        return None, 'No target image. Upload one or fetch from Instagram.'

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
        return None, 'Face swap failed. Could not detect faces in one or both images.'

    return result, 'Face swap completed successfully.'


def build_face_swap_tab():
    """Build the Face Swap Studio tab UI.

    Returns a dict of component references for external wiring.
    """
    gr.Markdown('### Face Swap Studio')
    gr.Markdown(
        'Swap your character\'s face onto any photo. '
        'Paste an Instagram post link or upload a target image directly.'
    )

    with gr.Row():
        # -- Left: Target image input --
        with gr.Column(scale=1):
            gr.Markdown('**Target Image**')

            ig_url = gr.Textbox(
                label='Instagram Post URL',
                placeholder='https://www.instagram.com/p/XXXXX/',
                lines=1, max_lines=1,
            )
            ig_fetch_btn = gr.Button(
                'Fetch from Instagram',
                variant='secondary', size='sm',
            )
            ig_status = gr.Textbox(
                label='Status', interactive=False,
                lines=1, max_lines=2, visible=True,
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
                    info='Restore facial details after swap',
                )
                do_enhanced_blending = gr.Checkbox(
                    label='Enhanced Blending',
                    value=True,
                    info='Face-parsing mask + Poisson seamless clone',
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
            lines=1, max_lines=2, scale=3,
        )

    result_image = gr.Image(
        label='Result', type='numpy',
        height=500, interactive=False,
    )

    # -- Wire events --
    ig_fetch_btn.click(
        fn=_fetch_ig_image,
        inputs=[ig_url],
        outputs=[target_image, ig_status],
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
        'ig_url': ig_url,
        'ig_fetch_btn': ig_fetch_btn,
        'ig_status': ig_status,
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
