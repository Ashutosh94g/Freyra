"""Freyra Advanced Settings Panel -- full power-user controls.

Ported from the legacy webui.py advanced column. Provides Settings, Styles,
Models, and Advanced tabs behind an "Advanced" toggle. All controls return
as a dict that can be wired into the generation pipeline as overrides.
"""

import copy
import gradio as gr

import modules.config
import modules.flags as flags
import modules.style_sorter as style_sorter
import args_manager

from modules.sdxl_styles import legal_style_names


def build_advanced_panel() -> dict:
    """Build the full advanced settings panel.

    Returns a dict of all Gradio components keyed by name for wiring.
    """
    components = {}

    advanced_checkbox = gr.Checkbox(
        label='Advanced',
        value=False,
        container=False,
        elem_classes='min_check',
    )
    components['advanced_checkbox'] = advanced_checkbox

    with gr.Column(visible=False) as advanced_column:
        with gr.Tab(label='Settings'):
            if not args_manager.args.disable_preset_selection:
                preset_selection = gr.Dropdown(
                    label='Preset',
                    choices=modules.config.available_presets,
                    value=args_manager.args.preset if args_manager.args.preset else "initial",
                    interactive=True,
                )
                components['preset_selection'] = preset_selection

            performance_selection = gr.Radio(
                label='Performance',
                choices=flags.Performance.values(),
                value=modules.config.default_performance,
                elem_classes=['performance_selection'],
            )
            components['performance_selection'] = performance_selection

            generation_steps = gr.Slider(
                label='Generation Steps',
                minimum=10, maximum=200, step=1,
                value=modules.config.default_generation_steps,
                info='More steps = better detail but slower.',
                interactive=not flags.Performance.has_restricted_features(
                    modules.config.default_performance),
            )
            components['generation_steps'] = generation_steps

            with gr.Accordion(label='Aspect Ratios', open=False):
                aspect_ratios_selection = gr.Radio(
                    label='Aspect Ratios', show_label=False,
                    choices=modules.config.available_aspect_ratios_labels,
                    value=modules.config.default_aspect_ratio,
                    info='width x height',
                    elem_classes='aspect_ratios',
                )
                components['aspect_ratios_selection'] = aspect_ratios_selection

            output_format = gr.Radio(
                label='Output Format',
                choices=flags.OutputFormat.list(),
                value=modules.config.default_output_format,
            )
            components['output_format'] = output_format

            negative_prompt = gr.Textbox(
                label='Negative Prompt', show_label=True,
                placeholder="Describing what you do not want to see.",
                lines=2, elem_id='adv_negative_prompt',
                value=modules.config.default_prompt_negative,
            )
            components['negative_prompt'] = negative_prompt

        with gr.Tab(label='Styles', elem_classes=['style_selections_tab']):
            style_sorter.try_load_sorted_styles(
                style_names=legal_style_names,
                default_selected=modules.config.default_styles,
            )

            style_search_bar = gr.Textbox(
                show_label=False, container=False,
                placeholder="\U0001F50E Type here to search styles ...",
                value="", label='Search Styles',
            )
            components['style_search_bar'] = style_search_bar

            style_selections = gr.CheckboxGroup(
                show_label=False, container=False,
                choices=copy.deepcopy(style_sorter.all_styles),
                value=copy.deepcopy(modules.config.default_styles),
                label='Selected Styles',
                elem_classes=['style_selections'],
            )
            components['style_selections'] = style_selections

            gradio_receiver_style_selections = gr.Textbox(
                elem_id='gradio_receiver_style_selections',
                visible=False,
            )
            components['gradio_receiver_style_selections'] = gradio_receiver_style_selections

        with gr.Tab(label='Models'):
            with gr.Group():
                with gr.Row():
                    base_model = gr.Dropdown(
                        label='Base Model (SDXL only)',
                        choices=modules.config.model_filenames,
                        value=modules.config.default_base_model_name,
                        show_label=True,
                    )
                    components['base_model'] = base_model

                    refiner_model = gr.Dropdown(
                        label='Refiner (SDXL or SD 1.5)',
                        choices=['None'] + modules.config.model_filenames,
                        value=modules.config.default_refiner_model_name,
                        show_label=True,
                    )
                    components['refiner_model'] = refiner_model

                refiner_switch = gr.Slider(
                    label='Refiner Switch At',
                    minimum=0.1, maximum=1.0, step=0.0001,
                    value=modules.config.default_refiner_switch,
                    visible=modules.config.default_refiner_model_name != 'None',
                )
                components['refiner_switch'] = refiner_switch

                refiner_model.change(
                    lambda x: gr.update(visible=x != 'None'),
                    inputs=refiner_model, outputs=refiner_switch,
                    show_progress="hidden", queue=False,
                )

            with gr.Group():
                lora_ctrls = []
                for i, (enabled, filename, weight) in enumerate(
                    modules.config.default_loras
                ):
                    with gr.Row():
                        lora_enabled = gr.Checkbox(
                            label='Enable', value=enabled,
                            elem_classes=['lora_enable', 'min_check'], scale=1,
                        )
                        lora_model = gr.Dropdown(
                            label=f'LoRA {i + 1}',
                            choices=['None'] + modules.config.lora_filenames,
                            value=filename,
                            elem_classes='lora_model', scale=5,
                        )
                        lora_weight = gr.Slider(
                            label='Weight',
                            minimum=modules.config.default_loras_min_weight,
                            maximum=modules.config.default_loras_max_weight,
                            step=0.01, value=weight,
                            elem_classes='lora_weight', scale=5,
                        )
                        lora_ctrls += [lora_enabled, lora_model, lora_weight]

                components['lora_ctrls'] = lora_ctrls

            with gr.Row():
                refresh_files = gr.Button(
                    value='\U0001f504 Refresh All Files',
                    variant='secondary',
                    elem_classes='refresh_button',
                )
                components['refresh_files'] = refresh_files

        with gr.Tab(label='Advanced'):
            guidance_scale = gr.Slider(
                label='Guidance Scale',
                minimum=1.0, maximum=30.0, step=0.01,
                value=modules.config.default_cfg_scale,
                info='Higher value means style is cleaner, vivider, and more artistic.',
            )
            components['guidance_scale'] = guidance_scale

            sharpness = gr.Slider(
                label='Image Sharpness',
                minimum=0.0, maximum=30.0, step=0.001,
                value=modules.config.default_sample_sharpness,
                info='Higher value means image and texture are sharper.',
            )
            components['sharpness'] = sharpness

            dev_mode = gr.Checkbox(
                label='Developer Debug Mode',
                value=modules.config.default_developer_debug_mode_checkbox,
                container=False,
            )
            components['dev_mode'] = dev_mode

            with gr.Column(
                visible=modules.config.default_developer_debug_mode_checkbox
            ) as dev_tools:
                with gr.Tab(label='Debug Tools'):
                    adm_scaler_positive = gr.Slider(
                        label='Positive ADM Guidance Scaler',
                        minimum=0.1, maximum=3.0, step=0.001, value=1.5,
                    )
                    components['adm_scaler_positive'] = adm_scaler_positive

                    adm_scaler_negative = gr.Slider(
                        label='Negative ADM Guidance Scaler',
                        minimum=0.1, maximum=3.0, step=0.001, value=0.8,
                    )
                    components['adm_scaler_negative'] = adm_scaler_negative

                    adm_scaler_end = gr.Slider(
                        label='ADM Guidance End At Step',
                        minimum=0.0, maximum=1.0, step=0.001, value=0.3,
                    )
                    components['adm_scaler_end'] = adm_scaler_end

                    refiner_swap_method = gr.Dropdown(
                        label='Refiner swap method',
                        value=flags.refiner_swap_method,
                        choices=['joint', 'separate', 'vae'],
                    )
                    components['refiner_swap_method'] = refiner_swap_method

                    adaptive_cfg = gr.Slider(
                        label='CFG Mimicking from TSNR',
                        minimum=1.0, maximum=30.0, step=0.01,
                        value=modules.config.default_cfg_tsnr,
                    )
                    components['adaptive_cfg'] = adaptive_cfg

                    clip_skip = gr.Slider(
                        label='CLIP Skip',
                        minimum=1, maximum=flags.clip_skip_max, step=1,
                        value=modules.config.default_clip_skip,
                    )
                    components['clip_skip'] = clip_skip

                    sampler_name = gr.Dropdown(
                        label='Sampler',
                        choices=flags.sampler_list,
                        value=modules.config.default_sampler,
                    )
                    components['sampler_name'] = sampler_name

                    scheduler_name = gr.Dropdown(
                        label='Scheduler',
                        choices=flags.scheduler_list,
                        value=modules.config.default_scheduler,
                    )
                    components['scheduler_name'] = scheduler_name

                    vae_name = gr.Dropdown(
                        label='VAE',
                        choices=[flags.default_vae] + modules.config.vae_filenames,
                        value=modules.config.default_vae,
                        show_label=True,
                    )
                    components['vae_name'] = vae_name

                    generate_image_grid = gr.Checkbox(
                        label='Generate Image Grid for Each Batch',
                        value=False,
                    )
                    components['generate_image_grid'] = generate_image_grid

                    overwrite_step = gr.Slider(
                        label='Forced Overwrite of Sampling Step',
                        minimum=-1, maximum=200, step=1,
                        value=modules.config.default_overwrite_step,
                        info='Set as -1 to disable.',
                    )
                    components['overwrite_step'] = overwrite_step

                    overwrite_switch = gr.Slider(
                        label='Forced Overwrite of Refiner Switch Step',
                        minimum=-1, maximum=200, step=1,
                        value=modules.config.default_overwrite_switch,
                        info='Set as -1 to disable.',
                    )
                    components['overwrite_switch'] = overwrite_switch

                    overwrite_width = gr.Slider(
                        label='Forced Overwrite of Generating Width',
                        minimum=-1, maximum=2048, step=1, value=-1,
                        info='Set as -1 to disable.',
                    )
                    components['overwrite_width'] = overwrite_width

                    overwrite_height = gr.Slider(
                        label='Forced Overwrite of Generating Height',
                        minimum=-1, maximum=2048, step=1, value=-1,
                        info='Set as -1 to disable.',
                    )
                    components['overwrite_height'] = overwrite_height

                    overwrite_vary_strength = gr.Slider(
                        label='Forced Overwrite of Denoising Strength of "Vary"',
                        minimum=-1, maximum=1.0, step=0.001, value=-1,
                        info='Set as negative number to disable.',
                    )
                    components['overwrite_vary_strength'] = overwrite_vary_strength

                    overwrite_upscale_strength = gr.Slider(
                        label='Forced Overwrite of Denoising Strength of "Upscale"',
                        minimum=-1, maximum=1.0, step=0.001,
                        value=modules.config.default_overwrite_upscale,
                        info='Set as negative number to disable.',
                    )
                    components['overwrite_upscale_strength'] = overwrite_upscale_strength

                    disable_preview = gr.Checkbox(
                        label='Disable Preview',
                        value=modules.config.default_black_out_nsfw,
                        interactive=not modules.config.default_black_out_nsfw,
                    )
                    components['disable_preview'] = disable_preview

                    disable_intermediate_results = gr.Checkbox(
                        label='Disable Intermediate Results',
                        value=flags.Performance.has_restricted_features(
                            modules.config.default_performance),
                    )
                    components['disable_intermediate_results'] = disable_intermediate_results

                    disable_seed_increment = gr.Checkbox(
                        label='Disable seed increment',
                        value=False,
                    )
                    components['disable_seed_increment'] = disable_seed_increment

                    read_wildcards_in_order = gr.Checkbox(
                        label="Read wildcards in order",
                        value=False,
                    )
                    components['read_wildcards_in_order'] = read_wildcards_in_order

                    black_out_nsfw = gr.Checkbox(
                        label='Black Out NSFW',
                        value=modules.config.default_black_out_nsfw,
                        interactive=not modules.config.default_black_out_nsfw,
                    )
                    components['black_out_nsfw'] = black_out_nsfw

                    black_out_nsfw.change(
                        lambda x: gr.update(value=x, interactive=not x),
                        inputs=black_out_nsfw, outputs=disable_preview,
                        queue=False, show_progress="hidden",
                    )

                    if not args_manager.args.disable_image_log:
                        save_final_enhanced_image_only = gr.Checkbox(
                            label='Save only final enhanced image',
                            value=modules.config.default_save_only_final_enhanced_image,
                        )
                        components['save_final_enhanced_image_only'] = save_final_enhanced_image_only

                    if not args_manager.args.disable_metadata:
                        save_metadata_to_images = gr.Checkbox(
                            label='Save Metadata to Images',
                            value=modules.config.default_save_metadata_to_images,
                        )
                        components['save_metadata_to_images'] = save_metadata_to_images

                        metadata_scheme = gr.Radio(
                            label='Metadata Scheme',
                            choices=flags.metadata_scheme,
                            value=modules.config.default_metadata_scheme,
                            visible=modules.config.default_save_metadata_to_images,
                        )
                        components['metadata_scheme'] = metadata_scheme

                        save_metadata_to_images.change(
                            lambda x: gr.update(visible=x),
                            inputs=[save_metadata_to_images],
                            outputs=[metadata_scheme],
                            queue=False, show_progress="hidden",
                        )

                with gr.Tab(label='Control'):
                    debugging_cn_preprocessor = gr.Checkbox(
                        label='Debug Preprocessors', value=False,
                    )
                    components['debugging_cn_preprocessor'] = debugging_cn_preprocessor

                    skipping_cn_preprocessor = gr.Checkbox(
                        label='Skip Preprocessors', value=False,
                    )
                    components['skipping_cn_preprocessor'] = skipping_cn_preprocessor

                    mixing_image_prompt_and_vary_upscale = gr.Checkbox(
                        label='Mixing Image Prompt and Vary/Upscale',
                        value=False,
                    )
                    components['mixing_image_prompt_and_vary_upscale'] = mixing_image_prompt_and_vary_upscale

                    mixing_image_prompt_and_inpaint = gr.Checkbox(
                        label='Mixing Image Prompt and Inpaint',
                        value=False,
                    )
                    components['mixing_image_prompt_and_inpaint'] = mixing_image_prompt_and_inpaint

                    controlnet_softness = gr.Slider(
                        label='Softness of ControlNet',
                        minimum=0.0, maximum=1.0, step=0.001, value=0.25,
                    )
                    components['controlnet_softness'] = controlnet_softness

                    with gr.Tab(label='Canny'):
                        canny_low_threshold = gr.Slider(
                            label='Canny Low Threshold',
                            minimum=1, maximum=255, step=1, value=64,
                        )
                        components['canny_low_threshold'] = canny_low_threshold

                        canny_high_threshold = gr.Slider(
                            label='Canny High Threshold',
                            minimum=1, maximum=255, step=1, value=128,
                        )
                        components['canny_high_threshold'] = canny_high_threshold

                with gr.Tab(label='Inpaint'):
                    debugging_inpaint_preprocessor = gr.Checkbox(
                        label='Debug Inpaint Preprocessing', value=False,
                    )
                    components['debugging_inpaint_preprocessor'] = debugging_inpaint_preprocessor

                    debugging_enhance_masks_checkbox = gr.Checkbox(
                        label='Debug Enhance Masks', value=False,
                    )
                    components['debugging_enhance_masks_checkbox'] = debugging_enhance_masks_checkbox

                    debugging_dino = gr.Checkbox(
                        label='Debug GroundingDINO', value=False,
                    )
                    components['debugging_dino'] = debugging_dino

                    inpaint_disable_initial_latent = gr.Checkbox(
                        label='Disable initial latent in inpaint',
                        value=False,
                    )
                    components['inpaint_disable_initial_latent'] = inpaint_disable_initial_latent

                    inpaint_engine = gr.Dropdown(
                        label='Inpaint Engine',
                        value=modules.config.default_inpaint_engine_version,
                        choices=flags.inpaint_engine_versions,
                    )
                    components['inpaint_engine'] = inpaint_engine

                    inpaint_strength = gr.Slider(
                        label='Inpaint Denoising Strength',
                        minimum=0.0, maximum=1.0, step=0.001, value=1.0,
                    )
                    components['inpaint_strength'] = inpaint_strength

                    inpaint_respective_field = gr.Slider(
                        label='Inpaint Respective Field',
                        minimum=0.0, maximum=1.0, step=0.001, value=0.618,
                    )
                    components['inpaint_respective_field'] = inpaint_respective_field

                    inpaint_erode_or_dilate = gr.Slider(
                        label='Mask Erode or Dilate',
                        minimum=-64, maximum=64, step=1, value=0,
                    )
                    components['inpaint_erode_or_dilate'] = inpaint_erode_or_dilate

                    dino_erode_or_dilate = gr.Slider(
                        label='GroundingDINO Box Erode or Dilate',
                        minimum=-64, maximum=64, step=1, value=0,
                    )
                    components['dino_erode_or_dilate'] = dino_erode_or_dilate

                with gr.Tab(label='FreeU'):
                    freeu_enabled = gr.Checkbox(label='Enabled', value=False)
                    components['freeu_enabled'] = freeu_enabled

                    freeu_b1 = gr.Slider(
                        label='B1', minimum=0, maximum=2, step=0.01, value=1.01,
                    )
                    components['freeu_b1'] = freeu_b1

                    freeu_b2 = gr.Slider(
                        label='B2', minimum=0, maximum=2, step=0.01, value=1.02,
                    )
                    components['freeu_b2'] = freeu_b2

                    freeu_s1 = gr.Slider(
                        label='S1', minimum=0, maximum=4, step=0.01, value=0.99,
                    )
                    components['freeu_s1'] = freeu_s1

                    freeu_s2 = gr.Slider(
                        label='S2', minimum=0, maximum=4, step=0.01, value=0.95,
                    )
                    components['freeu_s2'] = freeu_s2

            dev_mode.change(
                lambda r: gr.update(visible=r),
                inputs=[dev_mode], outputs=[dev_tools],
                queue=False, show_progress="hidden",
            )

    components['advanced_column'] = advanced_column

    advanced_checkbox.change(
        lambda x: gr.update(visible=x),
        advanced_checkbox, advanced_column,
        queue=False, show_progress="hidden",
    )

    # Wire performance -> step count + interactivity
    def on_performance_change(perf):
        is_restricted = flags.Performance.has_restricted_features(perf)
        step_map = {
            'Quality': 60, 'Speed': 30, 'Extreme Speed': 8,
            'Lightning': 4, 'Hyper-SD': 4,
        }
        new_steps = step_map.get(perf, 60)
        return (
            gr.update(interactive=not is_restricted),
            gr.update(value=is_restricted),
            gr.update(value=new_steps, interactive=not is_restricted),
        )

    performance_selection.change(
        on_performance_change,
        inputs=performance_selection,
        outputs=[guidance_scale, disable_intermediate_results, generation_steps],
        queue=False, show_progress="hidden",
    )

    # Wire style search
    style_search_bar.change(
        style_sorter.search_styles,
        inputs=[style_selections, style_search_bar],
        outputs=style_selections,
        queue=False, show_progress="hidden",
    ).then(lambda: None, js='()=>{refresh_style_localization();}')

    gradio_receiver_style_selections.input(
        style_sorter.sort_styles,
        inputs=style_selections,
        outputs=style_selections,
        queue=False, show_progress="hidden",
    ).then(lambda: None, js='()=>{refresh_style_localization();}')

    # Wire refresh files
    def refresh_files_clicked():
        modules.config.update_files()
        results = [gr.update(choices=modules.config.model_filenames)]
        results += [gr.update(choices=['None'] + modules.config.model_filenames)]
        results += [gr.update(choices=[flags.default_vae] + modules.config.vae_filenames)]
        if not args_manager.args.disable_preset_selection:
            results += [gr.update(choices=modules.config.available_presets)]
        for _ in range(modules.config.default_max_lora_number):
            results += [
                gr.update(interactive=True),
                gr.update(choices=['None'] + modules.config.lora_filenames),
                gr.update(),
            ]
        return results

    refresh_outputs = [base_model, refiner_model, vae_name]
    if not args_manager.args.disable_preset_selection:
        refresh_outputs += [preset_selection]
    refresh_outputs += lora_ctrls
    refresh_files.click(
        refresh_files_clicked, [], refresh_outputs,
        queue=False, show_progress="hidden",
    )

    return components


def collect_advanced_overrides(components: dict) -> list:
    """Return the list of Gradio component inputs for advanced overrides.

    These are the components whose values should be passed to the generation
    function when the user has Advanced mode enabled.
    """
    keys = [
        'performance_selection', 'generation_steps', 'aspect_ratios_selection',
        'output_format', 'negative_prompt', 'style_selections',
        'base_model', 'refiner_model', 'refiner_switch',
        'guidance_scale', 'sharpness',
        'sampler_name', 'scheduler_name', 'vae_name',
        'overwrite_step', 'overwrite_switch', 'overwrite_width', 'overwrite_height',
        'overwrite_vary_strength', 'overwrite_upscale_strength',
        'disable_preview', 'disable_intermediate_results',
        'disable_seed_increment', 'read_wildcards_in_order',
        'black_out_nsfw',
        'adm_scaler_positive', 'adm_scaler_negative', 'adm_scaler_end',
        'adaptive_cfg', 'clip_skip', 'refiner_swap_method',
        'controlnet_softness',
        'debugging_cn_preprocessor', 'skipping_cn_preprocessor',
        'mixing_image_prompt_and_vary_upscale', 'mixing_image_prompt_and_inpaint',
        'canny_low_threshold', 'canny_high_threshold',
        'freeu_enabled', 'freeu_b1', 'freeu_b2', 'freeu_s1', 'freeu_s2',
        'generate_image_grid',
    ]
    result = []
    for k in keys:
        comp = components.get(k)
        if comp is not None:
            result.append(comp)
    return result
