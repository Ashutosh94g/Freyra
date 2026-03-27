"""Freyra Dimension Input -- reusable tri-mode input component.

Each creative dimension (outfit, pose, lighting, etc.) gets three input modes:
1. Dropdown with curated presets
2. Custom text override
3. Reference image upload (BLIP extracts a description)

A hidden textbox holds the resolved effective value.
Priority: image description > custom text > dropdown.
"""

import gradio as gr


NONE_OPTION = 'None'


def build_dimension_input(
    dimension_key: str,
    label: str,
    options: list[str],
    allow_image: bool = True,
    open_accordion: bool = False,
) -> dict:
    """Build a tri-mode dimension input component.

    Returns a dict with keys:
        'dropdown': gr.Dropdown
        'custom': gr.Textbox
        'image': gr.Image (or None if allow_image=False)
        'image_desc': gr.Textbox (hidden, holds BLIP-extracted text)
        'resolved': gr.Textbox (hidden, holds the final effective value)
    """
    components = {}

    with gr.Accordion(label, open=open_accordion, elem_classes=['dimension-section']):
        dropdown = gr.Dropdown(
            label=f'Select {label}',
            choices=options,
            value=NONE_OPTION,
            interactive=True,
        )
        components['dropdown'] = dropdown

        custom = gr.Textbox(
            label=f'Custom {label}',
            placeholder='Or describe your own...',
            lines=1, max_lines=2,
        )
        components['custom'] = custom

        if allow_image:
            ref_image = gr.Image(
                label=f'{label} Reference Image',
                type='numpy',
                height=120,
                sources=['upload'],
                elem_classes=['dimension-ref-image'],
            )
            components['image'] = ref_image

            image_desc = gr.Textbox(
                label=f'{label} (extracted)',
                visible=False,
                interactive=False,
            )
            components['image_desc'] = image_desc
        else:
            components['image'] = None
            components['image_desc'] = None

        resolved = gr.Textbox(
            visible=False,
            elem_id=f'dim_resolved_{dimension_key}',
        )
        components['resolved'] = resolved

    return components


def wire_dimension_resolver(components: dict):
    """Wire the priority resolution: image_desc > custom > dropdown -> resolved.

    Call this after build_dimension_input. Sets up .change() events so the
    resolved textbox always holds the effective value.
    """
    dropdown = components['dropdown']
    custom = components['custom']
    image_desc = components.get('image_desc')
    resolved = components['resolved']

    inputs = [dropdown, custom]
    if image_desc is not None:
        inputs.append(image_desc)

    def _resolve(*args):
        dd_val = args[0] if len(args) > 0 else ''
        cust_val = args[1] if len(args) > 1 else ''
        img_val = args[2] if len(args) > 2 else ''

        if img_val and img_val.strip() and img_val.strip() != NONE_OPTION:
            return img_val.strip()
        if cust_val and cust_val.strip():
            return cust_val.strip()
        if dd_val and dd_val != NONE_OPTION:
            return dd_val
        return ''

    for comp in inputs:
        if comp is not None:
            comp.change(
                _resolve,
                inputs=inputs,
                outputs=[resolved],
                queue=False, show_progress='hidden',
            )


def wire_image_interrogation(components: dict, dimension_key: str):
    """Wire the image upload to run BLIP interrogation for this dimension.

    When a user uploads a reference image, BLIP runs and the result is
    written into image_desc, which then triggers the resolver chain.
    """
    ref_image = components.get('image')
    image_desc = components.get('image_desc')

    if ref_image is None or image_desc is None:
        return

    def _on_image_upload(img):
        if img is None:
            return gr.update(value='')
        try:
            from modules.image_interrogator import describe_for_dimension
            desc = describe_for_dimension(img, dimension_key)
            return gr.update(value=desc)
        except Exception as e:
            print(f'[Freyra] Image interrogation for {dimension_key} failed: {e}')
            return gr.update(value='')

    ref_image.change(
        _on_image_upload,
        inputs=[ref_image],
        outputs=[image_desc],
        queue=True, show_progress='hidden',
    )
