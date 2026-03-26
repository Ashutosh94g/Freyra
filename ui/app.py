"""Freyra v3.0 -- The Opinionated AI Photo Studio

Main UI application. Replaces the old monolithic webui.py with a clean,
opinionated interface built around creative dimensions.
"""

import gradio as gr
import random
import os
import time

import shared
import modules.config
import modules.html
import modules.async_worker as worker
import modules.constants as constants
import modules.flags as flags
import args_manager
import fooocus_version

from modules.auth import auth_enabled, check_auth
from modules.prompt_assembler import (
    load_options, load_options_no_none, NONE_OPTION, DIMENSION_FILES,
    assemble_prompt, randomize_dimensions,
)
from modules.shoot_types import (
    SHOOT_TYPES, SHOOT_TYPE_LABELS, QUALITY_MODES,
    QUALITY_MODE_LABELS, get_shoot_type, get_quality_mode,
)
from ui.theme import create_freyra_theme, FREYRA_CSS
from ui.constants import (
    FREYRA_TITLE, FREYRA_SUBTITLE, IMAGE_COUNT_MAX, IMAGE_COUNT_DEFAULT,
)
from ui.components.vram_indicator import create_vram_indicator
from modules.character_profiles import (
    list_profile_names, save_profile, load_profile, delete_profile,
)
from modules.face_engine import prepare_face_tasks, auto_select_method
from ui.pages.campaign import build_campaign_tab
from ui.pages.gallery import build_gallery_tab

try:
    from modules.ui_gradio_extensions import reload_javascript
    reload_javascript()
except Exception:
    pass


def _build_task_params(
    shoot_type_label, quality_label, image_number, image_seed, seed_random,
    skin_tone, hair_style, hair_color, outfit, pose, makeup, expression,
    background, lighting, camera_angle, footwear, custom_prompt,
    face_image_1, face_image_2, face_image_3,
    pro_enabled, pro_base_model, pro_seed_override, pro_resolution,
):
    """Build a params dict for AsyncTask.from_dict() from creative dimensions."""
    shoot = get_shoot_type(shoot_type_label)
    if shoot is None:
        shoot = list(SHOOT_TYPES.values())[0]

    quality = get_quality_mode(quality_label)
    if quality is None:
        quality = QUALITY_MODES["standard"]

    assembled = assemble_prompt(
        shoot_type_config=shoot,
        skin_tone=skin_tone,
        hair_style=hair_style,
        hair_color=hair_color,
        outfit=outfit,
        pose=pose,
        makeup=makeup,
        expression=expression,
        background=background,
        lighting=lighting,
        camera_angle=camera_angle,
        footwear=footwear,
        custom_prompt=custom_prompt,
    )

    if seed_random:
        seed = random.randint(constants.MIN_SEED, constants.MAX_SEED)
    else:
        try:
            seed = int(image_seed)
        except (ValueError, TypeError):
            seed = random.randint(constants.MIN_SEED, constants.MAX_SEED)

    loras_config = modules.config.default_loras[:2]
    shoot_loras = assembled.get('loras', [])
    if shoot_loras:
        loras_config = []
        for lora_name, lora_weight in shoot_loras[:2]:
            loras_config.append([True, lora_name, lora_weight])
        while len(loras_config) < modules.config.default_max_lora_number:
            loras_config.append([True, 'None', 1.0])
    else:
        while len(loras_config) < modules.config.default_max_lora_number:
            loras_config.append([True, 'None', 1.0])

    aspect_ratio = assembled.get('aspect_ratio', '896*1152')
    base_model = modules.config.default_base_model_name

    if pro_enabled:
        if pro_base_model and pro_base_model != 'None':
            base_model = pro_base_model
        if pro_resolution and pro_resolution != 'Default':
            aspect_ratio = pro_resolution
        if pro_seed_override and str(pro_seed_override).strip():
            try:
                seed = int(pro_seed_override)
            except (ValueError, TypeError):
                pass

    params = {
        'prompt': assembled['prompt'],
        'negative_prompt': assembled['negative_prompt'],
        'styles': assembled.get('styles', ['Fooocus V2', 'SAI Photographic', 'Fooocus Negative']),
        'performance': quality['performance'],
        'generation_steps': quality['steps'],
        'aspect_ratio': aspect_ratio,
        'image_number': int(image_number),
        'output_format': 'png',
        'seed': seed,
        'sharpness': assembled.get('sharpness', 2.0),
        'cfg_scale': assembled.get('cfg_scale', 4.5),
        'base_model': base_model,
        'refiner_model': 'None',
        'loras': loras_config,
        'sampler': 'dpmpp_2m_sde_gpu',
        'scheduler': 'karras',
        'disable_preview': False,
        'disable_intermediate_results': quality['performance'] in ('Lightning', 'Hyper-SD'),
        'builder_enabled': False,
    }

    face_images = [fi for fi in [face_image_1, face_image_2, face_image_3] if fi is not None]
    cn_tasks = prepare_face_tasks(face_images)
    params['_cn_tasks'] = cn_tasks
    params['_has_face'] = len(face_images) > 0

    return params, seed


def _create_task_from_params(params):
    """Create an AsyncTask from the params dict."""
    task = worker.AsyncTask.from_dict(params)

    cn_tasks = params.get('_cn_tasks')
    if cn_tasks:
        task.cn_tasks = cn_tasks

    if params.get('_has_face'):
        task.input_image_checkbox = True
        task.current_tab = 'ip'

    return task


def generate_clicked(task: worker.AsyncTask):
    import ldm_patched.modules.model_management as model_management

    with model_management.interrupt_processing_mutex:
        model_management.interrupt_processing = False

    if len(task.args) == 0 and not hasattr(task, 'prompt'):
        return

    execution_start_time = time.perf_counter()
    finished = False

    yield (
        gr.update(visible=True, value=modules.html.make_progress_html(1, 'Starting generation...')),
        gr.update(visible=True, value=None),
        gr.update(visible=False, value=None),
        gr.update(visible=False),
    )

    worker.async_tasks.append(task)

    while not finished:
        time.sleep(0.01)
        if len(task.yields) > 0:
            flag, product = task.yields.pop(0)
            if flag == 'preview':
                if len(task.yields) > 0 and task.yields[0][0] == 'preview':
                    continue
                percentage, title, image = product
                yield (
                    gr.update(visible=True, value=modules.html.make_progress_html(percentage, title)),
                    gr.update(visible=True, value=image) if image is not None else gr.update(),
                    gr.update(),
                    gr.update(visible=False),
                )
            if flag == 'results':
                yield (
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(visible=True, value=product),
                    gr.update(visible=False),
                )
            if flag == 'finish':
                yield (
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=True, value=product),
                )
                finished = True

    elapsed = time.perf_counter() - execution_start_time
    print(f'[Freyra] Generation complete: {elapsed:.2f}s')
    return


def build_ui():
    """Build the complete Freyra UI."""

    theme = create_freyra_theme()
    shared.gradio_root = gr.Blocks(
        title=f'{FREYRA_TITLE} {fooocus_version.version}',
        theme=theme,
        css=FREYRA_CSS,
    ).queue()

    with shared.gradio_root:
        current_task = gr.State(worker.AsyncTask(args=[]))
        state_is_generating = gr.State(False)

        # Header
        gr.HTML(
            f'<div class="freyra-header">'
            f'<h1>{FREYRA_TITLE}</h1>'
            f'<div class="subtitle">{FREYRA_SUBTITLE}</div>'
            f'</div>'
        )

        with gr.Row():
            # ── LEFT COLUMN: Creative Controls ──
            with gr.Column(scale=1):
                # Shoot Type
                shoot_type = gr.Radio(
                    label='Shoot Type',
                    choices=SHOOT_TYPE_LABELS,
                    value=SHOOT_TYPE_LABELS[0],
                    elem_classes=['shoot-type-radio'],
                )

                # Quality Mode
                quality_mode = gr.Radio(
                    label='Quality',
                    choices=QUALITY_MODE_LABELS,
                    value='Standard',
                    elem_classes=['quality-radio'],
                )

                # Image Count + Randomize
                image_number = gr.Slider(
                    label='Number of Images',
                    minimum=1,
                    maximum=IMAGE_COUNT_MAX,
                    step=1,
                    value=IMAGE_COUNT_DEFAULT,
                )

                randomize_btn = gr.Button(
                    value='Surprise Me',
                    variant='secondary',
                    size='sm',
                )

                # Character / Face
                with gr.Accordion('Character / Face', open=True, elem_classes=['dimension-section']):
                    saved_profiles = list_profile_names()
                    character_select = gr.Dropdown(
                        label='Saved Characters',
                        choices=saved_profiles,
                        value='None',
                        interactive=True,
                    )
                    gr.Markdown('Or upload 1-3 reference photos:')
                    with gr.Row():
                        face_image_1 = gr.Image(
                            label='Face 1', type='numpy',
                            height=120, sources=['upload'],
                            elem_classes=['face-upload-area'],
                        )
                        face_image_2 = gr.Image(
                            label='Face 2', type='numpy',
                            height=120, sources=['upload'],
                            elem_classes=['face-upload-area'],
                        )
                        face_image_3 = gr.Image(
                            label='Face 3', type='numpy',
                            height=120, sources=['upload'],
                            elem_classes=['face-upload-area'],
                        )
                    with gr.Row():
                        save_char_name = gr.Textbox(
                            label='Character Name', placeholder='Name this character...',
                            lines=1, max_lines=1, scale=3,
                        )
                        save_char_btn = gr.Button('Save Character', variant='secondary', scale=1, size='sm')
                        delete_char_btn = gr.Button('Delete', variant='stop', scale=1, size='sm')

                    def on_save_character(name, img1, img2, img3):
                        if not name or not name.strip():
                            return gr.update()
                        images = [i for i in [img1, img2, img3] if i is not None]
                        if not images:
                            return gr.update()
                        save_profile(name.strip(), images)
                        new_profiles = list_profile_names()
                        return gr.update(choices=new_profiles, value=name.strip())

                    save_char_btn.click(
                        on_save_character,
                        inputs=[save_char_name, face_image_1, face_image_2, face_image_3],
                        outputs=[character_select],
                        queue=False, show_progress='hidden',
                    )

                    def on_delete_character(name):
                        if not name or name == 'None':
                            return gr.update()
                        delete_profile(name)
                        new_profiles = list_profile_names()
                        return gr.update(choices=new_profiles, value='None')

                    delete_char_btn.click(
                        on_delete_character,
                        inputs=[character_select],
                        outputs=[character_select],
                        queue=False, show_progress='hidden',
                    )

                    def on_load_character(name):
                        images = load_profile(name)
                        if images is None:
                            return [None, None, None]
                        result = [None, None, None]
                        for i, img in enumerate(images[:3]):
                            result[i] = img
                        return result

                    character_select.change(
                        on_load_character,
                        inputs=[character_select],
                        outputs=[face_image_1, face_image_2, face_image_3],
                        queue=False, show_progress='hidden',
                    )

                # Skin Tone
                skin_tone_options = load_options('skin_tones.txt')
                with gr.Accordion('Skin Tone', open=False, elem_classes=['dimension-section']):
                    skin_tone = gr.Dropdown(
                        label='Skin Tone', choices=skin_tone_options,
                        value=NONE_OPTION, interactive=True,
                    )

                # Hair
                hair_style_options = load_options('influencer_hair.txt')
                hair_color_options = load_options('influencer_hair_colors.txt')
                with gr.Accordion('Hair', open=False, elem_classes=['dimension-section']):
                    hair_style = gr.Dropdown(
                        label='Hair Style', choices=hair_style_options,
                        value=NONE_OPTION, interactive=True,
                    )
                    hair_color = gr.Dropdown(
                        label='Hair Color', choices=hair_color_options,
                        value=NONE_OPTION, interactive=True,
                    )

                # Outfit
                outfit_options = load_options('influencer_outfits.txt')
                with gr.Accordion('Outfit', open=False, elem_classes=['dimension-section']):
                    outfit = gr.Dropdown(
                        label='Select Outfit', choices=outfit_options,
                        value=NONE_OPTION, interactive=True,
                    )
                    outfit_custom = gr.Textbox(
                        label='Custom Outfit',
                        placeholder='Or describe your own...',
                        lines=1, max_lines=1,
                    )

                # Pose
                pose_options = load_options('influencer_poses.txt')
                with gr.Accordion('Pose', open=False, elem_classes=['dimension-section']):
                    pose = gr.Dropdown(
                        label='Select Pose', choices=pose_options,
                        value=NONE_OPTION, interactive=True,
                    )
                    pose_custom = gr.Textbox(
                        label='Custom Pose',
                        placeholder='Or describe your own...',
                        lines=1, max_lines=1,
                    )

                # Makeup
                makeup_options = load_options('influencer_makeup.txt')
                with gr.Accordion('Makeup', open=False, elem_classes=['dimension-section']):
                    makeup = gr.Dropdown(
                        label='Select Makeup', choices=makeup_options,
                        value=NONE_OPTION, interactive=True,
                    )

                # Expression
                expression_options = load_options('influencer_expressions.txt')
                with gr.Accordion('Expression', open=False, elem_classes=['dimension-section']):
                    expression = gr.Dropdown(
                        label='Select Expression', choices=expression_options,
                        value=NONE_OPTION, interactive=True,
                    )

                # Background
                bg_options = load_options('influencer_settings.txt')
                with gr.Accordion('Background', open=False, elem_classes=['dimension-section']):
                    background = gr.Dropdown(
                        label='Select Background', choices=bg_options,
                        value=NONE_OPTION, interactive=True,
                    )
                    background_custom = gr.Textbox(
                        label='Custom Background',
                        placeholder='Or describe your own...',
                        lines=1, max_lines=1,
                    )

                # Lighting
                lighting_options = load_options('influencer_lighting.txt')
                with gr.Accordion('Lighting', open=False, elem_classes=['dimension-section']):
                    lighting = gr.Dropdown(
                        label='Select Lighting', choices=lighting_options,
                        value=NONE_OPTION, interactive=True,
                    )

                # Camera Angle
                camera_options = load_options('influencer_camera_angles.txt')
                with gr.Accordion('Camera Angle', open=False, elem_classes=['dimension-section']):
                    camera_angle = gr.Dropdown(
                        label='Select Camera Angle', choices=camera_options,
                        value=NONE_OPTION, interactive=True,
                    )

                # Footwear
                footwear_options = load_options('influencer_footwear.txt')
                with gr.Accordion('Footwear', open=False, elem_classes=['dimension-section']):
                    footwear = gr.Dropdown(
                        label='Select Footwear', choices=footwear_options,
                        value=NONE_OPTION, interactive=True,
                    )

                # Custom Prompt (optional extra)
                with gr.Accordion('Additional Details', open=False, elem_classes=['dimension-section']):
                    custom_prompt = gr.Textbox(
                        label='Extra Details',
                        placeholder='Add any additional details...',
                        lines=2, max_lines=3,
                    )

                # Pro Settings (hidden by default)
                with gr.Accordion('Pro Settings', open=False, elem_classes=['dimension-section', 'pro-settings-toggle']):
                    pro_enabled = gr.Checkbox(label='Enable Pro Overrides', value=False)
                    pro_base_model = gr.Dropdown(
                        label='Base Model',
                        choices=['None'] + modules.config.model_filenames,
                        value='None',
                        interactive=True,
                    )
                    seed_random = gr.Checkbox(label='Random Seed', value=True)
                    image_seed = gr.Textbox(label='Seed', value='0', visible=False)
                    seed_random.change(
                        lambda r: gr.update(visible=not r),
                        inputs=[seed_random], outputs=[image_seed],
                        queue=False, show_progress='hidden',
                    )
                    pro_resolution = gr.Dropdown(
                        label='Resolution Override',
                        choices=['Default'] + flags.sdxl_aspect_ratios,
                        value='Default',
                        interactive=True,
                    )
                    refresh_files = gr.Button(
                        value='Refresh Model Files',
                        variant='secondary',
                        size='sm',
                    )

                    def refresh_files_clicked():
                        modules.config.update_files()
                        return gr.update(choices=['None'] + modules.config.model_filenames)

                    refresh_files.click(
                        refresh_files_clicked, outputs=[pro_base_model],
                        queue=False, show_progress='hidden',
                    )

                # VRAM Monitor
                create_vram_indicator()

            # ── RIGHT COLUMN: Output ──
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab(label='Studio'):
                        with gr.Row():
                            progress_window = gr.Image(
                                label='Preview', show_label=True,
                                visible=False, height=512,
                            )
                            progress_gallery = gr.Gallery(
                                label='Generating...', show_label=True,
                                object_fit='contain', height=512,
                                visible=False, format='png',
                            )

                        progress_html = gr.HTML(
                            value=modules.html.make_progress_html(0, ''),
                            visible=False, elem_classes='progress-bar',
                        )

                        gallery = gr.Gallery(
                            label='Generated Images', show_label=False,
                            object_fit='contain', visible=True, height=500,
                            format='png', show_download_button=True,
                            elem_classes=['freyra-gallery'],
                        )

                        prompt_preview = gr.Textbox(
                            label='Assembled Prompt (read-only)',
                            interactive=False, lines=2, max_lines=4,
                            visible=True,
                        )

                        with gr.Row():
                            generate_button = gr.Button(
                                value='Generate',
                                variant='primary',
                                elem_classes=['freyra-generate-btn'],
                                scale=3,
                            )
                            stop_button = gr.Button(
                                value='Stop',
                                variant='stop',
                                visible=False,
                                elem_classes=['freyra-stop-btn'],
                                scale=1,
                            )
                            skip_button = gr.Button(
                                value='Skip',
                                variant='secondary',
                                visible=False,
                                scale=1,
                            )

                    with gr.Tab(label='Campaign'):
                        campaign_ui = build_campaign_tab()

                    with gr.Tab(label='Gallery & Export'):
                        gallery_ui = build_gallery_tab()

        # ── Wire: Surprise Me button ──
        randomize_outputs = [
            shoot_type, skin_tone, hair_style, hair_color,
            outfit, pose, makeup, expression,
            background, lighting, camera_angle, footwear,
        ]

        def on_randomize():
            dims = randomize_dimensions()
            rand_shoot = random.choice(SHOOT_TYPE_LABELS)
            return [
                rand_shoot,
                dims.get('skin_tone', NONE_OPTION),
                dims.get('hair_style', NONE_OPTION),
                dims.get('hair_color', NONE_OPTION),
                dims.get('outfit', NONE_OPTION),
                dims.get('pose', NONE_OPTION),
                dims.get('makeup', NONE_OPTION),
                dims.get('expression', NONE_OPTION),
                dims.get('background', NONE_OPTION),
                dims.get('lighting', NONE_OPTION),
                dims.get('camera_angle', NONE_OPTION),
                dims.get('footwear', NONE_OPTION),
            ]

        randomize_btn.click(
            on_randomize, outputs=randomize_outputs,
            queue=False, show_progress='hidden',
        )

        # ── Wire: prompt preview updates ──
        all_dimension_inputs = [
            shoot_type, skin_tone, hair_style, hair_color,
            outfit, outfit_custom, pose, pose_custom, makeup, expression,
            background, background_custom, lighting, camera_angle, footwear,
            custom_prompt,
        ]

        def update_prompt_preview(
            st, sk, hs, hc, ou, ou_c, po, po_c, mk, ex,
            bg, bg_c, lt, ca, fw, cp,
        ):
            shoot = get_shoot_type(st)
            if shoot is None:
                shoot = list(SHOOT_TYPES.values())[0]

            effective_outfit = ou_c.strip() if ou_c and ou_c.strip() else ou
            effective_pose = po_c.strip() if po_c and po_c.strip() else po
            effective_bg = bg_c.strip() if bg_c and bg_c.strip() else bg

            assembled = assemble_prompt(
                shoot_type_config=shoot,
                skin_tone=sk, hair_style=hs, hair_color=hc,
                outfit=effective_outfit, pose=effective_pose,
                makeup=mk, expression=ex,
                background=effective_bg, lighting=lt,
                camera_angle=ca, footwear=fw,
                custom_prompt=cp,
            )
            return assembled['prompt']

        for component in all_dimension_inputs:
            component.change(
                update_prompt_preview,
                inputs=all_dimension_inputs,
                outputs=[prompt_preview],
                queue=False, show_progress='hidden',
            )

        # ── Wire: generation ──
        def stop_clicked(task):
            import ldm_patched.modules.model_management as model_management
            task.last_stop = 'stop'
            if task.processing:
                model_management.interrupt_current_processing()
            return task

        def skip_clicked(task):
            import ldm_patched.modules.model_management as model_management
            task.last_stop = 'skip'
            if task.processing:
                model_management.interrupt_current_processing()
            return task

        stop_button.click(
            stop_clicked, inputs=current_task, outputs=current_task,
            queue=False, show_progress='hidden',
        )
        skip_button.click(
            skip_clicked, inputs=current_task, outputs=current_task,
            queue=False, show_progress='hidden',
        )

        generation_inputs = [
            shoot_type, quality_mode, image_number, image_seed, seed_random,
            skin_tone, hair_style, hair_color,
            outfit, pose, makeup, expression,
            background, lighting, camera_angle, footwear, custom_prompt,
            face_image_1, face_image_2, face_image_3,
            pro_enabled, pro_base_model, image_seed, pro_resolution,
        ]

        def prepare_generation(
            st, qm, img_num, img_seed, seed_rnd,
            sk, hs, hc, ou, po, mk, ex, bg, lt, ca, fw, cp,
            fi1, fi2, fi3,
            pro_en, pro_bm, pro_seed, pro_res,
        ):
            ou_val = ou
            po_val = po

            params, seed = _build_task_params(
                shoot_type_label=st, quality_label=qm,
                image_number=img_num, image_seed=img_seed, seed_random=seed_rnd,
                skin_tone=sk, hair_style=hs, hair_color=hc,
                outfit=ou_val, pose=po_val, makeup=mk, expression=ex,
                background=bg, lighting=lt, camera_angle=ca, footwear=fw,
                custom_prompt=cp,
                face_image_1=fi1, face_image_2=fi2, face_image_3=fi3,
                pro_enabled=pro_en, pro_base_model=pro_bm,
                pro_seed_override=pro_seed, pro_resolution=pro_res,
            )

            task = _create_task_from_params(params)
            return task

        generate_button.click(
            lambda: (
                gr.update(visible=True), gr.update(visible=True),
                gr.update(visible=False), [], True,
            ),
            outputs=[stop_button, skip_button, generate_button, gallery, state_is_generating],
        ).then(
            fn=prepare_generation,
            inputs=generation_inputs,
            outputs=current_task,
        ).then(
            fn=generate_clicked,
            inputs=current_task,
            outputs=[progress_html, progress_window, progress_gallery, gallery],
        ).then(
            lambda: (
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                False,
            ),
            outputs=[generate_button, stop_button, skip_button, state_is_generating],
        )

    return shared.gradio_root


def launch_app():
    """Build and launch the Freyra app."""
    app = build_ui()

    from modules.api import api_app
    app.app.mount("/api/v1", api_app)

    app.launch(
        inbrowser=args_manager.args.in_browser,
        server_name=args_manager.args.listen,
        server_port=args_manager.args.port,
        share=args_manager.args.share,
        auth=check_auth if (args_manager.args.share or args_manager.args.listen) and auth_enabled else None,
        allowed_paths=[
            modules.config.path_outputs,
            modules.config.temp_path,
            os.path.abspath('javascript'),
            os.path.abspath('css'),
            os.path.abspath('sdxl_styles'),
            os.path.abspath('assets'),
        ],
        blocked_paths=[constants.AUTH_FILENAME],
    )
