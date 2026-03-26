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
    """Create a VRAM indicator component that auto-refreshes."""
    vram_display = gr.HTML(value=get_vram_html, every=5)
    return vram_display
