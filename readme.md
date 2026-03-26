# Freyra

Freyra is an opinionated virtual photo studio designed exclusively for generating AI influencer images. 

*Freyra is a heavily customized fork of the excellent [Fooocus](https://github.com/lllyasviel/Fooocus) project created by `lllyasviel` and `mashb1t`. We deeply thank the original authors and their contributors for laying the incredible foundation.*

## The Freyra Philosophy

While Fooocus is a powerful, general-purpose image generator, Freyra has been fundamentally re-engineered for a single specific purpose: **AI Influencer Photography**.

1. **The User is a Creative Director, Not a Diffusion Engineer**
   We never expose underlying diffusion parameters. There are no sampler dropdowns, CFG sliders, scheduler pickers, or LoRA weight controls. All technical diffusion settings are decided internally by "shoot type" configurations.

2. **The 12 Creative Dimensions**
   Instead of writing complex prompts, you build images by combining intuitive creative dimensions:
   * Character/Face
   * Shoot Type
   * Pose
   * Outfit
   * Background
   * Lighting
   * Hair Style
   * Hair Color
   * Makeup
   * Expression
   * Camera Angle
   * Footwear
   * Skin Tone
   * Quality Mode

3. **Strict Hardware Constraints**
   Freyra's target hardware is explicitly the **Colab Free T4 (15GB VRAM)**. Every feature, from the fp8 UNet loader to the dual-LoRA limit and face adapter workspace, is engineered to fit within this strict limit without crashing.

## Advanced Capabilities

* **Campaign Mode:** Generate large-scale batch runs while maintaining rock-solid character lock and face consistency across dozens of images.
* **Persistent Character Profiles:** Integrated storage for character IDs, skin tones, and facial features.
* **Opinionated Pipeline:** Optimized internal defaults using `dpmpp_2m_sde_gpu`, `karras` scheduling, and maximum portrait resolution caps (`896*1152`) that guarantee high-fidelity results without tuning.

## Installation

```bash
git clone https://github.com/Ashutosh94g/Freyra.git
cd Freyra
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux
source venv/bin/activate
pip install -r requirements_versions.txt
python launch.py
```

## Original Credits
Freyra wouldn't exist without [Fooocus](https://github.com/lllyasviel/Fooocus). Please visit the original repository if you are looking for a general-purpose, simple-to-use Midjourney alternative.
