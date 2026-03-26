# Freyra v3.0 — The Opinionated AI Photo Studio

## What Is This

**Freyra** is a fork of [Fooocus](https://github.com/lllyasviel/Fooocus) (v2.5.5, GPL-3.0) transformed into
an **opinionated virtual photo studio** for generating photorealistic **female influencer AI model images**.

Unlike generic AI image tools that expose dozens of technical knobs, Freyra hides diffusion complexity
and exposes only what a creative director cares about: character identity, shoot type, pose, outfit,
background, lighting, hair, makeup, expression, camera angle, and footwear.

> Original Fooocus was effectively abandoned August 2024 (226 open issues, 59 open PRs, no maintainer activity).
> We fork it, fix it, and narrow its focus.

---

## Hard Constraints

| Constraint | Value |
|---|---|
| Target hardware | Google Colab free-tier **T4 GPU** |
| VRAM budget | 15 GB |
| System RAM budget | ~12.7 GB |
| Max config | Base model + 2 LoRAs simultaneously |
| Python | 3.12 (Colab default) |
| Use case | Photorealistic influencer images **only** |

---

## Known Bugs in Upstream Fooocus (must fix before adding features)

| # | Issue | Root cause | Fix |
|---|---|---|---|
| 1 | numpy / cupy crash | Colab Python 3.12 + numpy 2.x breaks cupy → pymatting → rembg | Pin `numpy<2.0`; catch cupy import failure gracefully |
| 2 | pygit2 version conflict | Fooocus pins 1.15.1; Colab ships 1.19.x | Change pin to `>=1.15.0` |
| 3 | PyTorch CUDA mismatch | Newer GPU arch (sm_120) unsupported by pinned torch | Let Colab manage torch; don't over-pin |
| 4 | Gradio 3.41.2 | Security CVEs, broken on newer Python | Upgrade to 4.x series |
| 5 | No maintenance | All PRs/issues stale | This fork owns it |

---

## Repository File Map (key files)

```
Freyra/
├── webui.py                  # 81 KB — entire Gradio UI
├── launch.py                 # Entry point, dependency installer
├── entry_with_update.py      # Git-pull + launch (do NOT use in Colab)
├── args_manager.py           # CLI argument definitions
├── fooocus_version.py        # Version string
├── modules/
│   ├── async_worker.py       # 79 KB — core generation pipeline orchestrator
│   ├── config.py             # 36 KB — config management, model paths, defaults
│   ├── default_pipeline.py   # 17 KB — model loading, LoRA, sampling pipeline
│   ├── core.py               # 13 KB — low-level diffusion ops
│   ├── patch.py              # 22 KB — sampling monkey-patches
│   ├── meta_parser.py        # 25 KB — metadata parsing (A1111/Fooocus)
│   ├── private_logger.py     # Image saving + HTML log + metadata spoofing hook
│   ├── metadata_spoof.py     # [NEW] EXIF spoofing to mimic real cameras
│   ├── inpaint_worker.py     # Inpainting algorithm
│   ├── lora.py               # LoRA loading
│   ├── flags.py              # Enums, constants
│   ├── sdxl_styles.py        # Style system
│   └── model_loader.py       # Model file discovery
├── ldm_patched/
│   └── modules/
│       └── model_management.py  # VRAM management — edited for T4 budget
├── presets/
│   ├── influencer.json       # [NEW] influencer preset
│   └── realistic.json        # upstream realistic preset (reference)
├── sdxl_styles/
│   └── sdxl_styles_influencer.json  # [NEW] fashion/fitness style prompts
├── wildcards/
│   ├── influencer_poses.txt       # [NEW]
│   ├── influencer_outfits.txt     # [NEW]
│   ├── influencer_settings.txt    # [NEW]
│   └── influencer_lighting.txt    # [NEW]
├── fooocus_influencer_colab.ipynb  # [NEW] T4-optimized notebook
└── run_influencer.bat              # [NEW] Windows launcher
```

---

## T4 Memory Budget

### Before optimizations (crashes)

| Component | VRAM |
|---|---|
| SDXL UNet fp16 | ~6.5 GB |
| CLIP encoders | ~1.5 GB |
| VAE | ~0.3 GB |
| GPT-2 prompt expansion | ~0.3 GB |
| 2 LoRAs | ~0.4–1.0 GB |
| Inference workspace | ~3–5 GB |
| **Total** | **12–14.6 GB → OOM at higher resolutions** |

### After optimizations (stable)

| Component | VRAM |
|---|---|
| SDXL UNet **fp8** | ~3.3 GB |
| CLIP encoders fp16 | ~1.5 GB |
| VAE fp16 | ~0.3 GB |
| GPT-2 | ~0.3 GB |
| 2 LoRAs | ~0.5 GB |
| **Available for inference** | **~9.1 GB ✓** |

---

## Memory Optimisation Checklist

- [x] Pin `numpy<2.0` in `requirements_versions.txt`
- [x] Raise `minimum_inference_memory()` → 2 GB in `ldm_patched/modules/model_management.py`
- [x] Default launch flags include `--unet-in-fp8-e4m3fn --always-high-vram`
- [x] Influencer preset disables refiner (`"default_refiner": "None"`)
- [x] Resolution capped at `896*1152` in influencer preset
- [ ] VAE tiling (future — add `--vae-in-cpu` flag if OOM persists)
- [ ] Aggressive `torch.cuda.empty_cache()` calls between pipeline stages (future)

---

## Development Workflow

```
Edit code locally (no GPU needed)
        ↓
git push origin main
        ↓
Google Colab free T4 — pull latest, run fooocus_influencer_colab.ipynb
        ↓
Test generation, iterate
```

**DO NOT** use `entry_with_update.py` in Colab — it git-pulls and may overwrite your changes.
Use `launch.py` directly.

---

## Phases

### Phase 1 — Fix & Fork ✅ (in progress)
1. Fork upstream to our GitHub
2. Fix numpy/cupy/pygit2 compatibility
3. Update Colab notebook
4. Verify basic txt2img works on T4

### Phase 2 — Influencer Preset & Styles ✅ (in progress)
1. `presets/influencer.json`
2. `sdxl_styles/sdxl_styles_influencer.json`
3. Wildcard files
4. `run_influencer.bat`

### Phase 3 — Metadata Spoofing Module ✅ (in progress)
- `modules/metadata_spoof.py` — strip AI metadata, inject real-camera EXIF
- Hook into `modules/private_logger.py`
- UI controls in `webui.py` (camera, GPS, photographer name)
- Requires `piexif` library

### Phase 4 — T4-Optimized Colab Notebook ✅ (in progress)
- `fooocus_influencer_colab.ipynb`
- Fix deps, clone fork, download models, launch with T4 flags

### Phase 5 — Face Consistency (later)
- Enhance InsightFace / FaceSwap in `extras/`
- "Character Lock" — save reference face, auto-apply across batch
- IP-Adapter FaceID integration

---

## Recommended Models

| Type | Model | Source |
|---|---|---|
| Base checkpoint | `RealVisXL_V5.0.safetensors` | CivitAI / HuggingFace |
| Photography LoRA | `SDXL_FILM_PHOTOGRAPHY_STYLE_V1.safetensors` | HuggingFace mashb1t |
| Face detail LoRA | `add-detail-xl.safetensors` | CivitAI |

---

## Launch Flags (T4 Colab)

```bash
python launch.py \
  --share \
  --preset influencer \
  --always-high-vram \
  --unet-in-fp8-e4m3fn \
  --disable-image-log
```

---

## Glossary

| Term | Meaning |
|---|---|
| `ldm_patched/` | Forked ComfyUI backend bundled inside Fooocus |
| FP8 | 8-bit float — half the VRAM of FP16 for UNet weights |
| LoRA | Low-Rank Adaptation — lightweight style/character layer |
| IP-Adapter | Image-Prompt adapter for face consistency |
| SDXL | Stable Diffusion XL — the base architecture used |
| T4 | Google Colab free-tier NVIDIA Tesla T4 GPU (16 GB, 15 GB usable) |
