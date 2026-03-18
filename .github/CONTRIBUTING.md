# Coding Guidelines for Freyra

This document contains conventions to follow when contributing to this fork.

## Scope

**This fork has exactly one use case:** photorealistic fashion/fitness female influencer images
on a free Google Colab T4 GPU. Do not add features outside this scope.

## Python Style

- Python 3.12 compatible
- No type annotations on code you didn't author (upstream Fooocus has none)
- Keep patches minimal — prefer wrapping over rewriting upstream code
- Every new module goes in `modules/`

## Memory Rules (T4, 15 GB VRAM)

- Always test with base model + 2 LoRAs loaded simultaneously
- Max resolution: `896×1152` (portrait) or `1152×896` (landscape)
- The UNet **must** run fp8 (`--unet-in-fp8-e4m3fn`) — do not make fp16 the default
- Never enable the SDXL refiner in the influencer preset
- After adding a pipeline stage, check if a `soft_empty_cache()` call is appropriate

## Adding New Styles

Style entries live in `sdxl_styles/sdxl_styles_influencer.json`.
Use `{prompt}` as the placeholder. Negative prompts are separate fields.
Run `python -c "import modules.sdxl_styles"` to verify the file parses.

## Adding New Presets

Presets live in `presets/`. They are plain JSON — schema matches `presets/realistic.json`.
Required keys: `default_model`, `default_loras`, `default_styles`, `default_aspect_ratio`.

## Metadata Spoofing

- Controlled by `modules/metadata_spoof.py`
- Hooked into `modules/private_logger.py` — do **not** call piexif anywhere else
- The feature must be opt-in (default off) until it's well tested
- Never hardcode GPS coordinates — always make them configurable

## Git Workflow

```
main          — stable, always Colab-runnable
feature/xxx   — experimental work
```

- Do not commit model weights (.safetensors, .ckpt, .pt)
- Do not commit `.env` files or auth tokens
- `fooocus_colab.ipynb` is the upstream notebook (kept for reference, broken)
- `fooocus_influencer_colab.ipynb` is our working notebook

## Do NOT

- Use `entry_with_update.py` in Colab (overwrites fork changes)
- Pin `gradio==3.41.2` (has CVEs)
- Import cupy at module level (not available in all Colab runtimes)
- Store credentials or API keys in any tracked file
