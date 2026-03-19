# Freyra — GitHub Copilot Instructions

## What This Repo Is

**Freyra** is a focused fork of Fooocus v2.5.5 (GPL-3.0) for generating photorealistic
**female influencer / fashion / fitness model images** on a free Google Colab T4 GPU.
The original upstream was abandoned in August 2024. We own all maintenance and features.

One use case only: **photorealistic female influencer images**. Do not add features outside this scope.

---

## Hard Constraints — Never Violate

| Constraint | Value |
|---|---|
| Target hardware | Google Colab free-tier **T4 GPU, 15 GB VRAM** |
| System RAM | ~12.7 GB |
| Max simultaneous models | Base checkpoint + 2 LoRAs |
| Python | 3.12 (Colab default) |
| Preferred UNet precision | fp8 (`--unet-in-fp8-e4m3fn`) — saves ~3 GB vs fp16 |
| Max safe resolution | `896×1152` (portrait) or `1152×896` (landscape) |
| Refiner | Always disabled in influencer preset |

---

## Architecture: How Generation Works

```
launch.py
  └── webui.py (Gradio UI)
        ├── get_task(*args) → AsyncTask(args=[...])  # packs all UI controls in order
        ├── generate_clicked(task) → yields progress/previews/results to Gradio
        └── worker thread (modules/async_worker.py)
              ├── handler(async_task) → @torch.no_grad @torch.inference_mode
              │     ├── performance mode selection (LCM / Lightning / HyperSD / Speed / Quality)
              │     ├── apply_image_input() → goals = ['vary'|'upscale'|'inpaint'|'cn'|'enhance']
              │     ├── process_prompt() → pipeline.refresh_everything() → CLIP encode
              │     ├── process_task() → pipeline.process_diffusion() → VAE decode → save
              │     └── async_task.yields.append(['finish', img_paths])
              └── modules/default_pipeline.py
                    ├── refresh_everything() → load base + refiner + LoRAs
                    ├── process_diffusion() → ldm_patched sampling loop
                    └── final_unet, final_clip, final_vae (module-level singletons)
```

### AsyncTask argument order

`AsyncTask.__init__` pops from a **reversed** list built by `get_task()` in `webui.py`.
The order is defined once in `webui.py:ctrls` — if you add a new UI control you MUST add it
to `ctrls` and pop it in `AsyncTask.__init__` in the matching position.

### Key module-level singletons in default_pipeline.py

- `final_unet` — currently loaded UNet (may be patched by FreeU / IP-Adapter / LoRA)
- `final_clip` — CLIP text encoder
- `final_vae` — VAE
- `final_expansion` — GPT-2 prompt expansion model
- `loaded_ControlNets` — dict of path → ControlNet model

Never hold references to these across generation calls; they are replaced by `refresh_everything()`.

---

## Key Files Reference

| File | Size | Role |
|---|---|---|
| `webui.py` | 81 KB | Entire Gradio UI — all components and callbacks |
| `modules/async_worker.py` | 79 KB | Core generation pipeline orchestrator |
| `modules/config.py` | 36 KB | Config management, model paths, defaults |
| `modules/default_pipeline.py` | 17 KB | Model loading, LoRA patching, sampling |
| `modules/patch.py` | 22 KB | Sampling monkey-patches (PatchSettings) |
| `modules/meta_parser.py` | 25 KB | Metadata parsing (A1111 / Fooocus schemes) |
| `modules/gradio_hijack.py` | ~500 lines | Gradio fixes: asyncio timeout, AsyncRequest patch, Queue httpx timeout |
| `modules/patch_clip.py` | ~200 lines | CLIPVision / CLIPText patching for Kohya compatibility |
| `modules/private_logger.py` | — | Image saving, HTML log, metadata spoofing hook |
| `modules/metadata_spoof.py` | — | EXIF spoofing — strip AI metadata, inject camera EXIF |
| `modules/flags.py` | — | All enums and valid value lists |
| `modules/lora.py` | — | LoRA loading into UNet and CLIP |
| `ldm_patched/modules/model_management.py` | — | VRAM management — edited for T4 budget |
| `extras/ip_adapter.py` | — | IP-Adapter patching of UNet attention |
| `extras/face_crop.py` | — | Face detection and cropping for FaceSwap tab |
| `presets/influencer.json` | — | Our primary preset |
| `presets/realistic.json` | — | Upstream reference preset |
| `sdxl_styles/sdxl_styles_influencer.json` | — | Fashion/fitness style prompts |
| `wildcards/` | — | influencer_poses, outfits, settings, lighting |
| `fooocus_influencer_colab.ipynb` | — | T4-optimized Colab notebook (use this, not fooocus_colab.ipynb) |
| `args_manager.py` | — | Freyra-specific CLI args (use this for new args) |
| `ldm_patched/modules/args_parser.py` | — | Backend CLI args (VRAM mode, dtype flags) |

---

## All Valid Parameter Values

### Samplers (`modules/flags.py:KSAMPLER`)
```
euler, euler_ancestral, heun, heunpp2, dpm_2, dpm_2_ancestral, lms, dpm_fast,
dpm_adaptive, dpmpp_2s_ancestral, dpmpp_sde, dpmpp_sde_gpu, dpmpp_2m,
dpmpp_2m_sde, dpmpp_2m_sde_gpu, dpmpp_3m_sde, dpmpp_3m_sde_gpu, ddpm, lcm, tcd,
restart, ddim, uni_pc, uni_pc_bh2
```
**Best for photorealism:** `dpmpp_2m_sde_gpu` (default), `dpmpp_2m_sde`, `euler_ancestral`

### Schedulers
```
normal, karras, exponential, sgm_uniform, simple, ddim_uniform, lcm, turbo,
align_your_steps, tcd, edm_playground_v2.5
```
**Best for photorealism:** `karras` (default)

### Aspect Ratios (`modules/flags.py:sdxl_aspect_ratios`)
Portrait modes for influencer: `896*1152` (default), `832*1216`, `832*1152`, `768*1344`
Square: `1024*1024`
Landscape: `1152*896`, `1216*832`

### Performance Modes (`modules/flags.py:Performance`)
- `Speed` — 30 steps (default, use this)
- `Quality` — 60 steps
- `Extreme Speed` — LCM, ~8 steps
- `Lightning` — ~4 steps
- `Hyper-SD` — ~4 steps

### ControlNet Types (`modules/flags.py:ip_list`)
```
ImagePrompt, FaceSwap, PyraCanny, CPDS
```

### Output Formats
```
png, jpeg, webp
```

### Metadata Schemes
```
fooocus (JSON), a1111 (plain text)
```

### Inpaint Engines
```
None, v1, v2.5, v2.6
```

### Inpaint Mask Models
```
u2net, u2netp, u2net_human_seg, u2net_cloth_seg, silueta,
isnet-general-use, isnet-anime, sam
```

---

## Influencer Image Prompt Framework

### Subject Construction Template
```
[ethnicity] female, [age range], [body type], [skin tone],
[hair: length, color, style], [eye color],
[clothing: type, color, fabric, fit],
[accessories],
[pose/body posture],
[head position], [facial expression],
[setting/background],
[lighting], [camera angle], [lens style]
```

### Valid Values by Category

**Ethnicity / Look**
```
caucasian, latina, east asian, south asian, mixed race, middle eastern,
african american, scandinavian, mediterranean
```

**Body Type**
```
athletic, slim, curvy, petite, tall slender, hourglass, fit, toned
```

**Skin Tone**
```
fair, porcelain, light, olive, tan, bronze, caramel, dark brown, ebony
```

**Hair Length**
```
buzzcut, pixie, chin length bob, shoulder length, collarbone length,
mid-back, waist length, extra long
```

**Hair Color**
```
jet black, dark brown, chestnut brown, warm brunette, dirty blonde,
golden blonde, platinum blonde, ash blonde, auburn, copper red,
burgundy, rose gold, silver, salt and pepper highlights, ombre
```

**Hair Style**
```
straight sleek, loose waves, beach waves, tight curls, coily natural,
high ponytail, low ponytail, messy bun, top knot, half up half down,
braided, side swept, windswept, slicked back, voluminous blowout
```

**Clothing Type**
```
sports bra and leggings, athleisure set, mini dress, bodycon dress,
maxi dress, wrap dress, crop top and high waist jeans, blazer and shorts,
tailored suit, bikini, one-piece swimsuit, lingerie set, streetwear hoodie,
off-shoulder top, fitted turtleneck, satin slip dress, linen co-ord set
```

**Clothing Color**
```
white, off-white, cream, ivory, beige, nude, blush pink, hot pink,
fuchsia, coral, terracotta, burnt orange, mustard yellow, olive green,
forest green, sage, teal, sky blue, royal blue, navy, cobalt, lavender,
mauve, burgundy, wine red, caramel, tan, chocolate brown, charcoal, black
```

**Clothing Fabric / Texture**
```
silk, satin, cotton, linen, denim, leather, faux leather, velvet, lace,
mesh, sheer, ribbed knit, cable knit, sequin, metallic, suede
```

**Body Posture**
```
standing straight, standing with weight on one leg, contrapposto,
sitting on edge, sitting cross-legged, leaning against wall,
leaning on railing, arms crossed, hands on hips, arms raised overhead,
one hand on chin, walking forward, mid-stride walk, dynamic action pose,
side profile standing, three-quarter turn, back turned looking over shoulder,
crouching, kneeling, jumping
```

**Head Position**
```
facing camera directly, slight left turn, slight right turn,
strong left profile, strong right profile, chin up, chin down,
head tilted left, head tilted right, looking up, looking down,
looking over shoulder
```

**Facial Expression**
```
neutral confident, slight smile, open smile, big laugh, smirking,
pouty lips, serious editorial, sultry half-lidded eyes, surprised,
eyes closed serene, biting lip, blowing kiss, fierce direct gaze
```

**Background / Setting**
```
white studio, grey seamless, black studio, luxury apartment interior,
modern minimalist bedroom, marble bathroom, rooftop city view,
tropical beach, poolside, lush garden, urban street, alley graffiti wall,
coffee shop window, boutique fashion store, desert landscape, snowy mountain,
golden wheat field, hotel lobby, penthouse terrace at sunset
```

**Lighting**
```
softbox studio, ring light, rembrandt, split lighting, butterfly lighting,
backlit rim light, golden hour sunlight, overcast diffused, hard dramatic shadows,
neon accent light, candle warm low-key, high key bright white, cinematic moody
```

**Camera / Lens Style**
```
full body shot, three-quarter shot, waist up portrait, tight headshot,
over-the-shoulder, low angle looking up, high angle looking down,
eye-level portrait, extreme close-up face, detail shot hands,
85mm portrait lens bokeh, 35mm environmental, 50mm natural, wide angle 24mm
```

---

## Recommended Models (Colab Download)

### Checkpoints (place in `models/checkpoints/`)
| Model | HuggingFace Path | Notes |
|---|---|---|
| `realisticStockPhoto_v20.safetensors` | `lllyasviel/fav_models` | Current default |
| `RealVisXL_V5.0.safetensors` | CivitAI download | Best general photorealism |
| `juggernautXL_v9Rdphoto2Lightning.safetensors` | CivitAI download | Fast photorealism |
| `dreamshaper_xl_v21TurboDPMSDE_v21.safetensors` | CivitAI download | Creative editorial |

### LoRAs (place in `models/loras/`)
| LoRA | Source | Weight | Purpose |
|---|---|---|---|
| `SDXL_FILM_PHOTOGRAPHY_STYLE_V1.safetensors` | `mashb1t/fav_models` | 0.25 | Film grain, photography feel |
| `add-detail-xl.safetensors` | CivitAI | 0.5–0.8 | Skin detail, pore detail |
| `epiCPhoto.safetensors` | CivitAI | 0.4–0.6 | Photorealistic enhancement |
| `DetailTweaker_xl.safetensors` | CivitAI | 0.5 | Fine detail pass |
| `xl_more_art-full_v1.safetensors` | CivitAI | 0.2–0.4 | Composition improvement |

### VAE (place in `models/vae/`)
| VAE | Source | Notes |
|---|---|---|
| `sdxl_vae.safetensors` | `stabilityai/sdxl-vae` | Better color accuracy than model-default |
| `sdxl-vae-fp16-fix.safetensors` | `madebyollin/sdxl-vae-fp16-fix` | Stable at fp16 (no NaNs) |

### Preset LoRA Loading Pattern
```json
"default_loras": [
  [true, "SDXL_FILM_PHOTOGRAPHY_STYLE_V1.safetensors", 0.25],
  [true, "add-detail-xl.safetensors", 0.5],
  [true, "epiCPhoto.safetensors", 0.4],
  [true, "None", 1.0],
  [true, "None", 1.0]
]
```

---

## Launch Flags Reference

### Freyra-level flags (`args_manager.py`)
```bash
--share                     # Gradio public URL (required for Colab)
--preset influencer         # Load influencer.json preset
--disable-image-log         # Don't write files to outputs/ (saves Colab disk)
--disable-metadata          # Skip metadata embedding
--language default          # UI language
--theme dark|light
--enable-auto-describe-image
--always-download-new-model
```

### Backend flags (`ldm_patched/modules/args_parser.py`)
```bash
# VRAM mode (pick one):
--always-gpu                # Keep everything on GPU (T4 default)
--always-high-vram          # Alias
--always-low-vram           # Force aggressive offloading

# UNet precision (pick one):
--unet-in-fp8-e4m3fn        # RECOMMENDED for T4 — saves ~3 GB
--unet-in-fp16
--unet-in-bf16

# VAE precision (pick one):
--vae-in-fp16               # Stable with fp16-fix VAE
--vae-in-fp32               # Default
--vae-in-cpu                # Offload VAE to CPU if OOM

# Attention:
--attention-pytorch         # Use PyTorch SDPA (T4 default, no xformers needed)
--attention-split           # Memory-split attention (fallback)

# Precision shortcuts:
--all-in-fp16               # Everything fp16 (risky without fp16-fix VAE)
```

### T4 Recommended Launch Command
```bash
python launch.py \
  --share \
  --preset influencer \
  --always-gpu \
  --unet-in-fp8-e4m3fn \
  --attention-pytorch \
  --disable-image-log
```

---

## Roadmap — Planned Features

### Phase 5 — Full CLI / API Control (next)
**Goal:** Every generation parameter controllable via CLI arg or REST API call without touching the UI.

**Design:**
- `args_manager.py` — add generation default overrides:
  `--default-prompt`, `--default-negative`, `--default-steps`, `--default-cfg`,
  `--default-sampler`, `--default-scheduler`, `--default-loras` (JSON string),
  `--default-resolution`, `--image-number`, `--seed`
- `modules/config.py` — read those args in `try_eval_env_var` fallbacks
- `modules/api.py` (NEW) — FastAPI app mounted at `/api/v1/` alongside Gradio
  - `POST /api/v1/generate` — accepts full generation spec JSON, returns job ID
  - `GET  /api/v1/jobs/{id}` — returns status + image URLs when done
  - `GET  /api/v1/models` — list available checkpoints / LoRAs / VAEs
  - `GET  /api/v1/presets` — list presets
  - Request body mirrors `AsyncTask` fields exactly
  - Mount via `shared.gradio_root.app.mount("/api/v1", api_app)`
- `modules/async_worker.py` — `AsyncTask` must accept keyword construction (not just positional args list)

**Key files to modify:** `webui.py`, `args_manager.py`, `modules/config.py`, `modules/async_worker.py`
**New file:** `modules/api.py`

---

### Phase 6 — Enhanced LoRA Stack (quality improvement)
**Goal:** Load up to 5 LoRAs simultaneously from the influencer preset with correct T4 memory budget.

**Research needed:**
- Current T4 budget with fp8 UNet leaves ~9 GB for inference workspace
- Each SDXL LoRA at rank 64 costs ~200–400 MB merged into UNet weights
- 5 LoRAs at 0.3–0.5 weight each: estimated ~0.8–1.5 GB total → fits within budget
- Test: load base + 5 LoRAs, generate at `896×1152`, monitor `nvidia-smi` peak

**LoRA stack for influencer quality:**
1. `SDXL_FILM_PHOTOGRAPHY_STYLE_V1` @ 0.25 — film texture
2. `add-detail-xl` @ 0.5 — skin/face detail
3. `epiCPhoto` @ 0.4 — photorealistic color
4. `DetailTweaker_xl` @ 0.35 — micro detail
5. (slot 5 free for character LoRA — see Phase 8)

**Files to modify:** `presets/influencer.json` (add loras 3–5 with HuggingFace download URLs)

---

### Phase 7 — Style & Prompt Wildcards Expansion
**Goal:** Rich, varied influencer prompts generated automatically so every output is unique.

**Wildcard files to expand/create:**
- `wildcards/influencer_poses.txt` — 50+ poses from the valid values list above
- `wildcards/influencer_outfits.txt` — 50+ outfit combinations
- `wildcards/influencer_settings.txt` — 30+ backgrounds/locations
- `wildcards/influencer_lighting.txt` — 20+ lighting setups
- `wildcards/influencer_faces.txt` (NEW) — face expressions + head positions
- `wildcards/influencer_hair.txt` (NEW) — hair style + color combinations
- `wildcards/skin_tones.txt` (NEW) — ethnicity + skin tone combos

**Usage in prompt:** `__influencer_poses__, __influencer_outfits__, __influencer_lighting__`
Wildcard syntax is already supported by `modules/util.py:apply_wildcards()`.

---

### Phase 8 — Character LoRA Training Pipeline (face consistency)
**Goal:** Train a custom LoRA on 10–20 reference images of one character to lock face identity across all generations.

**Architecture:**
- Use `kohya_ss` / `SimpleTuner` training scripts (external tool, run on Colab A100 or locally)
- Training data: 10 photos, face-cropped + captioned using `extras/face_crop.py` + BLIP-2
- LoRA rank: 32 (balance quality vs size)
- Training steps: 1000–1500 on SDXL
- Output: `models/loras/character_{name}.safetensors`
- Integration: load as LoRA slot 5 at weight 0.7–0.9

**Helper module to build:** `modules/character_prep.py`
- `crop_training_set(image_dir, output_dir)` — batch face crop using `extras/face_crop.py`
- `generate_captions(image_dir)` — BLIP-2 caption each image
- `write_kohya_config(character_name, image_dir)` — write kohya dataset config JSON

**Trigger word convention:** `<name>_character` e.g. `<luna_character>`

---

### Phase 9 — API Batch Generation Workflow
**Goal:** Generate 50–100 influencer images per Colab session without touching the UI.

**Design:**
```python
# Example batch spec (JSON file passed to API)
{
  "jobs": [
    {
      "prompt": "__influencer_hair__ female, __skin_tones__, __influencer_outfits__",
      "image_number": 4,
      "aspect_ratio": "896*1152",
      "loras": [["SDXL_FILM_PHOTOGRAPHY_STYLE_V1.safetensors", 0.25],
                ["add-detail-xl.safetensors", 0.5]],
      "seed": -1
    }
  ]
}
```

**Colab notebook cell:**
```python
import requests, json
spec = json.load(open('batch_spec.json'))
for job in spec['jobs']:
    r = requests.post('http://127.0.0.1:7865/api/v1/generate', json=job)
    job_id = r.json()['job_id']
    # poll /api/v1/jobs/{job_id} until done
```

---

## Known Bugs & Fixed Issues

| # | Bug | Status | Fix Location |
|---|---|---|---|
| 1 | numpy/cupy crash on Python 3.12 | Fixed | `requirements_versions.txt` — `numpy<2.0` |
| 2 | pygit2 version conflict | Fixed | `requirements_versions.txt` — `pygit2>=1.15.0` |
| 3 | PyTorch CUDA mismatch | Fixed | Don't over-pin torch; let Colab manage |
| 4 | Gradio 3.41.2 CVEs | Deferred (separate branch) | N/A |
| 5 | `no_init_weights` removed in transformers ≥4.45 | **Fixed** | `modules/patch_clip.py:163` — getattr fallback to `contextlib.nullcontext` |
| 6 | UI freezes after generation when IP-Adapter loads | **Fixed** | `modules/gradio_hijack.py` — AsyncRequest default + Queue httpx Timeout(None) |

---

## Coding Rules

1. **Python 3.12 only** — no 3.13+ syntax
2. **No type annotations** on code you didn't author — upstream has none
3. **Patches over rewrites** — wrap upstream functions, never rewrite them inline
4. **Every new module** goes in `modules/`
5. **New args** go in `args_manager.py` (Freyra-level) or read from `config.txt` / preset JSON
6. **Do not commit** model weights (`.safetensors`, `.ckpt`, `.pt`, `.bin` >1 MB)
7. **Memory discipline** — after any new pipeline stage, check if `torch.cuda.empty_cache()` is needed
8. **Wildcard files** — plain text, one value per line, no trailing spaces
9. **Style files** — valid JSON matching schema of `sdxl_styles_influencer.json`
10. **Preset files** — valid JSON, always include `checkpoint_downloads` and `lora_downloads` keys

## Git Workflow
```
main          — stable, always Colab-runnable
feature/xxx   — experimental work, merge to main only after Colab test
```
Never force-push main. Test generation in Colab before merging any feature branch.

## Testing a Change in Colab
```python
# In a notebook cell:
!cd /content/Freyra && git pull
# Restart runtime kernel
# Re-run the launch cell
# Trigger one generation with an IP-Adapter face reference to exercise the full path
```
