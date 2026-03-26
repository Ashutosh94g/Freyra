"""VRAM usage indicator component for Freyra UI."""

import gradio as gr


def get_vram_html() -> str:
    """Generate HTML for VRAM usage bar."""
    try:
        import ldm_patched.modules.model_management as mm
        info = mm.get_vram_info()
        if info['total'] == 0:
            return '<div style="color:#666;font-size:11px;">GPU not detected</div>'

        pct = info['percent']
        color = '#c4852e' if pct < 80 else '#e74c3c' if pct < 95 else '#ff0000'

        return (
            f'<div style="font-size:11px;color:#888;margin:4px 0;">'
            f'VRAM: {info["used_gb"]}GB / {info["total_gb"]}GB ({pct}%)'
            f'<div class="vram-bar">'
            f'<div class="vram-bar-fill" style="width:{pct}%;background:{color};"></div>'
            f'</div></div>'
        )
    except Exception:
        return ''


def create_vram_indicator():
    """Create a VRAM indicator with manual refresh.

    The `every=N` parameter causes 504 Gateway Timeout on Gradio share links
    because the scheduled event blocks the queue during initial page load.
    Use a button to refresh on demand instead.
    """
    initial_html = ''
    try:
        initial_html = get_vram_html()
    except Exception:
        pass

    vram_display = gr.HTML(value=initial_html)
    refresh_btn = gr.Button('Refresh VRAM', variant='secondary', size='sm')
    refresh_btn.click(
        fn=get_vram_html,
        outputs=[vram_display],
        queue=False, show_progress='hidden',
    )
    return vram_display
