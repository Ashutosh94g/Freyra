"""Freyra custom Gradio theme -- dark luxury creative aesthetic."""

import gradio as gr


def create_freyra_theme():
    return gr.themes.Base(
        primary_hue=gr.themes.Color(
            c50="#fdf8f0",
            c100="#faecd8",
            c200="#f5d5a8",
            c300="#e8b86d",
            c400="#d4993e",
            c500="#c4852e",
            c600="#a96a22",
            c700="#8a5220",
            c800="#6e4120",
            c900="#5a361d",
            c950="#331c0e",
            name="amber",
        ),
        secondary_hue=gr.themes.Color(
            c50="#f8f8f8",
            c100="#f0f0f0",
            c200="#e4e4e4",
            c300="#d1d1d1",
            c400="#b4b4b4",
            c500="#9a9a9a",
            c600="#6e6e6e",
            c700="#4e4e4e",
            c800="#3a3a3a",
            c900="#2a2a2a",
            c950="#1a1a1a",
            name="neutral",
        ),
        neutral_hue=gr.themes.Color(
            c50="#fafafa",
            c100="#f5f5f5",
            c200="#e8e8e8",
            c300="#d4d4d4",
            c400="#a8a8a8",
            c500="#787878",
            c600="#585858",
            c700="#404040",
            c800="#2d2d2d",
            c900="#1e1e1e",
            c950="#141414",
            name="zinc",
        ),
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
    ).set(
        body_background_fill="#0f0f0f",
        body_background_fill_dark="#0f0f0f",
        body_text_color="#e0e0e0",
        body_text_color_dark="#e0e0e0",
        body_text_color_subdued="#999999",
        body_text_color_subdued_dark="#999999",
        background_fill_primary="#1a1a1a",
        background_fill_primary_dark="#1a1a1a",
        background_fill_secondary="#141414",
        background_fill_secondary_dark="#141414",
        border_color_primary="#2a2a2a",
        border_color_primary_dark="#2a2a2a",
        block_background_fill="#1a1a1a",
        block_background_fill_dark="#1a1a1a",
        block_border_color="#2a2a2a",
        block_border_color_dark="#2a2a2a",
        block_label_background_fill="#1e1e1e",
        block_label_background_fill_dark="#1e1e1e",
        block_label_text_color="#c4852e",
        block_label_text_color_dark="#c4852e",
        block_title_text_color="#e0e0e0",
        block_title_text_color_dark="#e0e0e0",
        panel_background_fill="#141414",
        panel_background_fill_dark="#141414",
        panel_border_color="#2a2a2a",
        panel_border_color_dark="#2a2a2a",
        input_background_fill="#1e1e1e",
        input_background_fill_dark="#1e1e1e",
        input_border_color="#333333",
        input_border_color_dark="#333333",
        input_border_color_focus="#c4852e",
        input_border_color_focus_dark="#c4852e",
        button_primary_background_fill="#c4852e",
        button_primary_background_fill_dark="#c4852e",
        button_primary_background_fill_hover="#d4993e",
        button_primary_background_fill_hover_dark="#d4993e",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
        button_secondary_background_fill="#2a2a2a",
        button_secondary_background_fill_dark="#2a2a2a",
        button_secondary_background_fill_hover="#333333",
        button_secondary_background_fill_hover_dark="#333333",
        button_secondary_text_color="#e0e0e0",
        button_secondary_text_color_dark="#e0e0e0",
        slider_color="#c4852e",
        slider_color_dark="#c4852e",
        checkbox_background_color="#1e1e1e",
        checkbox_background_color_dark="#1e1e1e",
        checkbox_background_color_selected="#c4852e",
        checkbox_background_color_selected_dark="#c4852e",
        checkbox_border_color="#444444",
        checkbox_border_color_dark="#444444",
        checkbox_label_text_color="#e0e0e0",
        checkbox_label_text_color_dark="#e0e0e0",
        shadow_spread="0px",
        block_shadow="none",
        block_shadow_dark="none",
    )


FREYRA_CSS = """
/* Freyra v3.0 -- Opinionated AI Photo Studio */

.freyra-header {
    text-align: center;
    padding: 12px 0 8px 0;
    border-bottom: 1px solid #2a2a2a;
    margin-bottom: 12px;
}

.freyra-header h1 {
    font-size: 28px;
    font-weight: 700;
    color: #c4852e;
    margin: 0;
    letter-spacing: 2px;
}

.freyra-header .subtitle {
    font-size: 12px;
    color: #666;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-top: 2px;
}

.freyra-generate-btn {
    min-height: 56px !important;
    font-size: 18px !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    border-radius: 8px !important;
}

.freyra-stop-btn {
    min-height: 42px !important;
    background: #8b0000 !important;
    border-color: #8b0000 !important;
    color: white !important;
}

.freyra-stop-btn:hover {
    background: #a00000 !important;
}

.dimension-section {
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    margin-bottom: 4px;
    background: #161616;
}

.dimension-section .label-wrap {
    padding: 8px 12px !important;
}

.quality-radio .wrap {
    gap: 8px;
}

.quality-radio label {
    border-radius: 6px !important;
    padding: 8px 16px !important;
    font-size: 13px !important;
}

.shoot-type-radio label {
    border-radius: 6px !important;
    padding: 6px 12px !important;
    font-size: 12px !important;
}

.freyra-gallery {
    min-height: 500px;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    background: #141414;
}

.pro-settings-toggle {
    opacity: 0.5;
    font-size: 11px !important;
}

.pro-settings-toggle:hover {
    opacity: 0.8;
}

.vram-bar {
    height: 6px;
    background: #1e1e1e;
    border-radius: 3px;
    overflow: hidden;
    margin: 4px 0;
}

.vram-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #c4852e, #d4993e);
    border-radius: 3px;
    transition: width 0.3s ease;
}

.face-upload-area {
    border: 2px dashed #333;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
    min-height: 120px;
}

.face-upload-area:hover {
    border-color: #c4852e;
}

footer {
    display: none !important;
}

.progress-bar {
    margin: 4px 0;
}

.loader-container {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px;
}

.loader-container .progress-container {
    flex: 1;
}

.loader-container progress {
    width: 100%;
    height: 8px;
    border-radius: 4px;
    appearance: none;
}

.loader-container progress::-webkit-progress-bar {
    background: #1e1e1e;
    border-radius: 4px;
}

.loader-container progress::-webkit-progress-value {
    background: linear-gradient(90deg, #c4852e, #d4993e);
    border-radius: 4px;
}

.loader-container span {
    color: #999;
    font-size: 12px;
    white-space: nowrap;
}

.loader {
    border: 3px solid #2a2a2a;
    border-top: 3px solid #c4852e;
    border-radius: 50%;
    width: 20px;
    height: 20px;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* ── Visual Picker ── */
.visual-picker-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
    gap: 6px;
    padding: 6px 0;
    max-height: 320px;
    overflow-y: auto;
}

.picker-card {
    border: 2px solid #2a2a2a;
    border-radius: 8px;
    padding: 10px 8px;
    cursor: pointer;
    transition: border-color 0.15s, transform 0.1s, box-shadow 0.15s;
    min-height: 56px;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    background: #1e1e1e;
    user-select: none;
}

.picker-card:hover {
    border-color: #c4852e;
    transform: translateY(-1px);
    box-shadow: 0 2px 8px rgba(196, 133, 46, 0.2);
}

.picker-card.selected {
    border-color: #c4852e;
    box-shadow: 0 0 0 2px rgba(196, 133, 46, 0.4), 0 2px 8px rgba(196, 133, 46, 0.3);
}

.picker-card-label {
    font-size: 11px;
    font-weight: 500;
    line-height: 1.3;
    word-break: break-word;
}

/* ── Pose Editor ── */
.pose-reference-upload {
    border: 2px dashed #333;
    border-radius: 8px;
    margin-top: 8px;
}

.pose-reference-upload:hover {
    border-color: #c4852e;
}

/* ── History Panel ── */
.freyra-history-controls {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0 16px 0;
    border-bottom: 1px solid #2a2a2a;
    margin-bottom: 12px;
}

.freyra-hist-btn {
    background: #2a2a2a;
    color: #ccc;
    border: 1px solid #444;
    border-radius: 4px;
    padding: 6px 14px;
    cursor: pointer;
    font-size: 12px;
    display: inline-block;
    text-align: center;
}

.freyra-hist-btn:hover {
    background: #3a3a3a;
    color: #fff;
}

.freyra-hist-btn-danger {
    border-color: #8b0000;
    color: #ff6b6b;
}

.freyra-hist-btn-danger:hover {
    background: #4a0000;
}

.freyra-history-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
    gap: 12px;
}

.freyra-history-card {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    overflow: hidden;
    transition: border-color 0.15s, box-shadow 0.15s;
    cursor: default;
}

.freyra-history-card:hover {
    border-color: #c4852e;
    box-shadow: 0 2px 12px rgba(196, 133, 46, 0.15);
}

.freyra-history-thumb {
    width: 100%;
    height: 140px;
    object-fit: cover;
    display: block;
    background: #111;
}

.freyra-history-no-thumb {
    display: flex;
    align-items: center;
    justify-content: center;
    color: #444;
    font-size: 13px;
}

.freyra-history-meta {
    padding: 10px;
}

.freyra-history-date {
    font-size: 11px;
    color: #888;
}

.freyra-history-seed {
    font-size: 11px;
    color: #c4852e;
    margin-top: 2px;
}

.freyra-history-prompt {
    font-size: 12px;
    color: #aaa;
    margin-top: 4px;
    line-height: 1.3;
    max-height: 48px;
    overflow: hidden;
}

.freyra-history-count {
    font-size: 11px;
    color: #666;
    margin-top: 4px;
}

/* ── Responsive layout ── */
@media (max-width: 768px) {
    .freyra-header h1 {
        font-size: 22px;
        letter-spacing: 1px;
    }

    .freyra-header .subtitle {
        font-size: 10px;
    }

    .gradio-container {
        padding: 4px !important;
    }

    /* Stack columns vertically on mobile */
    .gradio-row {
        flex-direction: column !important;
    }

    .gradio-column {
        max-width: 100% !important;
        flex: 1 1 100% !important;
    }

    /* Gallery fills full width */
    .freyra-gallery {
        min-height: 300px;
        height: auto !important;
    }

    .freyra-gallery .gallery-item img {
        max-width: 100%;
    }

    /* Accordion compact mode */
    .dimension-section {
        margin-bottom: 2px;
    }

    .dimension-section .label-wrap {
        padding: 10px 12px !important;
        min-height: 44px;
        display: flex;
        align-items: center;
    }

    .dimension-section .label-wrap span {
        font-size: 14px !important;
    }

    /* Touch-friendly inputs */
    .gradio-dropdown,
    .gradio-textbox textarea,
    .gradio-textbox input,
    .gradio-slider input,
    .gradio-checkbox input {
        min-height: 44px !important;
        font-size: 16px !important;
    }

    /* Sticky generate button */
    .freyra-generate-btn {
        position: sticky;
        bottom: 8px;
        z-index: 100;
        min-height: 52px !important;
        font-size: 16px !important;
        box-shadow: 0 -4px 16px rgba(0,0,0,0.6);
    }

    .freyra-stop-btn {
        min-height: 44px !important;
    }

    /* Picker grid compact */
    .visual-picker-grid {
        grid-template-columns: repeat(auto-fill, minmax(90px, 1fr));
        gap: 4px;
        max-height: 240px;
    }

    .picker-card {
        min-height: 48px;
        padding: 8px 6px;
    }

    .picker-card-label {
        font-size: 10px;
    }

    /* Face upload compact */
    .face-upload-area {
        min-height: 90px;
        padding: 8px;
    }

    /* History compact */
    .freyra-history-grid {
        grid-template-columns: 1fr;
        gap: 8px;
    }

    .freyra-history-thumb {
        height: 100px;
    }

    .freyra-history-controls {
        flex-direction: column;
        gap: 8px;
        align-items: flex-start;
    }

    /* Pose editor responsive */
    .pose-reference-upload img {
        max-height: 150px;
    }
}

/* Extra small screens (phones in portrait) */
@media (max-width: 480px) {
    .freyra-header h1 {
        font-size: 18px;
    }

    .visual-picker-grid {
        grid-template-columns: repeat(3, 1fr);
    }

    .face-upload-area {
        min-height: 70px;
    }

    .freyra-generate-btn {
        min-height: 48px !important;
        font-size: 14px !important;
    }
}
"""
