"""Freyra Campaign Page -- batch generation with character consistency."""

import gradio as gr

from modules.campaign import list_campaigns, create_campaign
from modules.character_profiles import list_profile_names
from modules.shoot_types import SHOOT_TYPE_LABELS
from modules.prompt_assembler import load_options, NONE_OPTION


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

    pose_options = load_options('influencer_poses.txt')
    bg_options = load_options('influencer_settings.txt')
    outfit_options = load_options('influencer_outfits.txt')

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
