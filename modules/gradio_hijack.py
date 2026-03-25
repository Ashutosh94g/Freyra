"""Gradio 4.x compatibility module for Freyra.

Replaces the Gradio 3.x custom Image class with thin wrappers
around Gradio 4's gr.Image and gr.ImageEditor.

Keeps Block.__init__ component tracking and Gradio internals patches.
"""

from __future__ import annotations

import importlib
import inspect as _inspect

import numpy as np
import gradio as gr
import gradio.routes

from gradio.blocks import Block


# ─── Component tracking ──────────────────────────────────────────────
# all_components is populated by the Block.__init__ override below.
# webui.py uses it for dump_english_config().
all_components = []

if not hasattr(Block, 'original_init'):
    Block.original_init = Block.__init__


def _blk_init(self, *args, **kwargs):
    all_components.append(self)
    return Block.original_init(self, *args, **kwargs)


Block.__init__ = _blk_init


# ─── Image compatibility wrapper ─────────────────────────────────────
# Maps old gr.Image(source=..., tool=..., brush_color=...) API
# to Gradio 4 gr.Image / gr.ImageEditor.

def Image(
    value=None,
    *,
    label=None,
    show_label=None,
    height=None,
    width=None,
    source='upload',
    sources=None,
    tool=None,
    type='numpy',
    image_mode='RGB',
    brush_color='#000000',
    brush_radius=None,
    mask_opacity=0.7,
    interactive=None,
    visible=True,
    elem_id=None,
    elem_classes=None,
    show_download_button=True,
    show_share_button=None,
    container=True,
    scale=None,
    min_width=160,
    every=None,
    streaming=False,
    mirror_webcam=True,
    invert_colors=False,
    shape=None,
    **kwargs,
):
    """Compatibility wrapper: returns gr.Image or gr.ImageEditor depending on tool."""
    # Map old source= string to new sources= list
    if sources is None:
        source_map = {
            'upload': ['upload'],
            'webcam': ['webcam'],
            'canvas': ['upload'],
        }
        sources = source_map.get(source, ['upload'])

    if tool in ('sketch', 'color-sketch'):
        # Use ImageEditor for brush/sketch tools
        brush_kwargs = {
            'default_color': brush_color,
            'color_mode': 'fixed' if tool == 'sketch' else 'defaults',
        }
        if brush_color:
            brush_kwargs['colors'] = [brush_color]
        if brush_radius:
            brush_kwargs['default_size'] = int(brush_radius)

        return gr.ImageEditor(
            value=value,
            label=label,
            show_label=show_label,
            height=height,
            width=width,
            type=type,
            sources=sources,
            brush=gr.Brush(**brush_kwargs),
            eraser=gr.Eraser(default_size=int(brush_radius) if brush_radius else 3),
            layers=False,
            interactive=interactive,
            visible=visible,
            elem_id=elem_id,
            elem_classes=elem_classes,
            container=container,
            scale=scale,
            min_width=min_width,
            every=every,
        )
    else:
        # Standard image display/upload
        return gr.Image(
            value=value,
            label=label,
            show_label=show_label,
            height=height,
            width=width,
            type=type,
            image_mode=image_mode,
            sources=sources,
            interactive=interactive,
            visible=visible,
            elem_id=elem_id,
            elem_classes=elem_classes,
            show_download_button=show_download_button,
            show_share_button=show_share_button,
            container=container,
            scale=scale,
            min_width=min_width,
            every=every,
            streaming=streaming,
            mirror_webcam=mirror_webcam,
        )


# ─── ImageEditor → legacy dict conversion ────────────────────────────

def convert_image_editor_to_legacy(value):
    """Convert gr.ImageEditor output to legacy gr.Image(tool='sketch') format.

    Old format: {'image': numpy(H,W,3), 'mask': numpy(H,W,3)}
    New format: {'background': numpy(H,W,3), 'layers': [numpy(H,W,4)], 'composite': numpy(H,W,3)}
    """
    if value is None:
        return None
    if isinstance(value, dict) and 'background' in value:
        image = value.get('background')
        layers = value.get('layers', [])

        mask = None
        if layers:
            layer = layers[0] if isinstance(layers[0], np.ndarray) else None
            if layer is not None:
                if layer.ndim == 3 and layer.shape[2] == 4:
                    # RGBA layer → extract alpha as grayscale mask → stack to 3-channel
                    alpha = layer[:, :, 3]
                    mask = np.stack([alpha, alpha, alpha], axis=-1)
                elif layer.ndim == 3:
                    mask = layer
                else:
                    mask = layer

        return {'image': image, 'mask': mask}
    return value


# ─── Gradio internals patches ────────────────────────────────────────

# Patch asyncio.wait_for timeout — prevents Gradio task timeout on
# long-running generation (e.g. IP-Adapter model loading).
try:
    gradio.routes.asyncio = importlib.reload(gradio.routes.asyncio)

    if not hasattr(gradio.routes.asyncio, 'original_wait_for'):
        gradio.routes.asyncio.original_wait_for = gradio.routes.asyncio.wait_for

    def _patched_wait_for(fut, timeout):
        del timeout
        return gradio.routes.asyncio.original_wait_for(fut, timeout=65535)

    gradio.routes.asyncio.wait_for = _patched_wait_for
except Exception:
    pass

# Guard: AsyncRequest was removed in Gradio 4+ (queue uses WebSockets).
import gradio.utils

if hasattr(gradio.utils, 'AsyncRequest') and not hasattr(gradio.utils.AsyncRequest, '_json_response_data'):
    gradio.utils.AsyncRequest._json_response_data = {}

# Guard: Queue httpx patch — only applies to Gradio 3.x with HTTP polling.
try:
    import gradio.queueing
    import httpx as _httpx

    _original_queue_start = gradio.queueing.Queue.start

    if _inspect.iscoroutinefunction(_original_queue_start):
        async def _patched_queue_start(self, ssl_verify=True):
            await _original_queue_start(self, ssl_verify=ssl_verify)
            try:
                await self.queue_client.aclose()
            except Exception:
                pass
            self.queue_client = _httpx.AsyncClient(
                verify=ssl_verify,
                timeout=_httpx.Timeout(None),
            )

        gradio.queueing.Queue.start = _patched_queue_start
except Exception:
    pass

