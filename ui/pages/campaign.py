"""Freyra Campaign Page -- batch generation with character consistency."""

import random
import time
import gradio as gr

from modules.campaign import create_campaign, update_campaign_status
from modules.character_profiles import list_profile_names, load_profile
from modules.shoot_types import SHOOT_TYPE_LABELS, get_shoot_type, SHOOT_TYPES, QUALITY_MODES
from modules.prompt_assembler import load_options_no_none, assemble_prompt
from modules.face_engine import prepare_face_tasks
import modules.config
import modules.async_worker as worker


def _generate_campaign(
    name, character_name, shoot_label, count,
    vary_poses, vary_backgrounds, vary_outfits,
):
    """Run sequential generation for a campaign with character lock."""
    if not name or not name.strip():
        yield 'Please enter a campaign name.', []
        return

    if not character_name or character_name == 'None':
        yield 'Please select a saved character first.', []
        return

    face_images = load_profile(character_name)
    if face_images is None or len(face_images) == 0:
        yield 'Could not load character face images.', []
        return

    shoot = get_shoot_type(shoot_label)
    if shoot is None:
        shoot = list(SHOOT_TYPES.values())[0]

    quality = QUALITY_MODES['standard']

    poses = load_options_no_none('influencer_poses.txt') if vary_poses else []
    backgrounds = load_options_no_none('influencer_settings.txt') if vary_backgrounds else []
    outfits = load_options_no_none('influencer_outfits.txt') if vary_outfits else []

    count = int(count)
    variations = []
    for i in range(count):
        v = {}
        if poses:
            v['pose'] = poses[i % len(poses)] if i < len(poses) else random.choice(poses)
        if backgrounds:
            v['background'] = backgrounds[i % len(backgrounds)] if i < len(backgrounds) else random.choice(backgrounds)
        if outfits:
            v['outfit'] = outfits[i % len(outfits)] if i < len(outfits) else random.choice(outfits)
        variations.append(v)

    campaign_id = create_campaign(name.strip(), character_name, shoot_label, variations)
    update_campaign_status(campaign_id, 'running')

    all_results = []

    for i, var in enumerate(variations):
        yield f'Generating image {i + 1}/{count}...', all_results

        try:
            assembled = assemble_prompt(
                shoot_type_config=shoot,
                pose=var.get('pose', ''),
                background=var.get('background', ''),
                outfit=var.get('outfit', ''),
            )

            loras_config = []
            for lora_name, lora_weight in assembled.get('loras', [])[:2]:
                loras_config.append([True, lora_name, lora_weight])
            while len(loras_config) < modules.config.default_max_lora_number:
                loras_config.append([True, 'None', 1.0])

            params = {
                'prompt': assembled['prompt'],
                'negative_prompt': assembled['negative_prompt'],
                'styles': assembled.get('styles', ['Freyra V2', 'SAI Photographic', 'Freyra Negative']),
                'performance': quality['performance'],
                'generation_steps': quality['steps'],
                'aspect_ratio': assembled.get('aspect_ratio', '896*1152'),
                'image_number': 1,
                'output_format': 'png',
                'seed': random.randint(0, 2**31),
                'sharpness': assembled.get('sharpness', 2.0),
                'cfg_scale': assembled.get('cfg_scale', 4.5),
                'base_model': modules.config.default_base_model_name,
                'refiner_model': 'None',
                'loras': loras_config,
                'sampler': 'dpmpp_2m_sde_gpu',
                'scheduler': 'karras',
                'disable_preview': True,
                'disable_intermediate_results': True,
            }

            cn_tasks = prepare_face_tasks(face_images)
            params['_cn_tasks'] = cn_tasks
            params['_has_face'] = True

            task = worker.AsyncTask.from_dict(params)
            if cn_tasks:
                task.cn_tasks = cn_tasks
            task.input_image_checkbox = True
            task.current_tab = 'ip'

            worker.async_tasks.append(task)

            timeout = 300
            start = time.time()
            while time.time() - start < timeout:
                time.sleep(0.2)
                if len(task.yields) > 0:
                    flag, product = task.yields.pop(0)
                    if flag == 'finish':
                        if isinstance(product, list):
                            all_results.extend(product)
                        break
            else:
                yield f'Image {i + 1}/{count} timed out. Continuing...', all_results
                continue

        except Exception as e:
            yield f'Error on image {i + 1}/{count}: {e}', all_results
            continue

        yield f'Completed image {i + 1}/{count}', all_results

    update_campaign_status(campaign_id, 'completed', all_results)
    yield f'Campaign complete! {count} images generated.', all_results


def build_campaign_tab():
    """Build the Campaign tab UI."""
    gr.Markdown('### Campaign Mode')
    gr.Markdown('Generate a series of images with the same character in different scenarios.')

    campaign_name = gr.Textbox(
        label='Campaign Name',
        placeholder='e.g., Beach Summer Collection',
        lines=1, max_lines=1,
    )

    campaign_character = gr.Dropdown(
        label='Character',
        choices=list_profile_names(),
        value='None',
        interactive=True,
    )

    campaign_shoot_type = gr.Dropdown(
        label='Shoot Type',
        choices=SHOOT_TYPE_LABELS,
        value=SHOOT_TYPE_LABELS[0],
        interactive=True,
    )

    campaign_count = gr.Slider(
        label='Number of Variations',
        minimum=2, maximum=10, step=1, value=4,
    )

    gr.Markdown('**Per-image variation dimensions** (the system will cycle through these)')

    campaign_vary_poses = gr.Checkbox(label='Vary Poses', value=True)
    campaign_vary_backgrounds = gr.Checkbox(label='Vary Backgrounds', value=True)
    campaign_vary_outfits = gr.Checkbox(label='Vary Outfits', value=False)

    campaign_status = gr.Textbox(
        label='Status', interactive=False, lines=2,
        value='Configure your campaign above and click Generate Campaign.',
    )

    generate_campaign_btn = gr.Button(
        value='Generate Campaign',
        variant='primary',
    )

    campaign_gallery = gr.Gallery(
        label='Campaign Results',
        show_label=True,
        object_fit='contain',
        height=400,
        format='png',
        show_download_button=True,
    )

    generate_campaign_btn.click(
        fn=_generate_campaign,
        inputs=[
            campaign_name, campaign_character, campaign_shoot_type,
            campaign_count,
            campaign_vary_poses, campaign_vary_backgrounds, campaign_vary_outfits,
        ],
        outputs=[campaign_status, campaign_gallery],
    )

    return {
        'name': campaign_name,
        'character': campaign_character,
        'shoot_type': campaign_shoot_type,
        'count': campaign_count,
        'vary_poses': campaign_vary_poses,
        'vary_backgrounds': campaign_vary_backgrounds,
        'vary_outfits': campaign_vary_outfits,
        'status': campaign_status,
        'generate_btn': generate_campaign_btn,
        'gallery': campaign_gallery,
    }
