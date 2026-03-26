"""Generate visual picker thumbnails for Freyra wildcard options.

Run this script ON Colab (with GPU) after the app is set up:
    cd /content/Freyra
    python tools/generate_thumbnails.py

It uses the Freyra pipeline to generate a 256x256 preview for each
wildcard option, saved to assets/thumbnails/{category}/{hash}.jpg.

Uses Lightning mode (4 steps) for speed -- ~3 seconds per thumbnail on T4.
"""

import os
import sys
import hashlib
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

THUMBNAIL_SIZE = (256, 256)
THUMBNAIL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              'assets', 'thumbnails')

CATEGORIES = [
    'influencer_outfits',
    'influencer_poses',
    'influencer_settings',
    'influencer_lighting',
    'influencer_makeup',
    'influencer_expressions',
    'influencer_hair',
    'influencer_camera_angles',
    'influencer_footwear',
]

CATEGORY_PROMPT_TEMPLATES = {
    'influencer_outfits': 'fashion model wearing {option}, studio photo, clean background, 1girl',
    'influencer_poses': 'fashion model {option}, studio photo, clean white background, 1girl',
    'influencer_settings': 'empty scene of {option}, no people, establishing shot, cinematic',
    'influencer_lighting': 'portrait of woman, {option}, close-up face, professional photography',
    'influencer_makeup': 'close-up portrait of woman with {option}, beauty photography',
    'influencer_expressions': 'close-up portrait of woman, {option}, headshot',
    'influencer_hair': 'portrait of woman with {option} hair, headshot, studio lighting',
    'influencer_camera_angles': 'fashion model, {option}, studio photo, 1girl',
    'influencer_footwear': 'fashion model wearing {option}, full body shot, clean background',
}


def slug(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:10]


def load_options(filename: str) -> list[str]:
    wildcard_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'wildcards')
    filepath = os.path.join(wildcard_dir, filename)
    if not os.path.isfile(filepath):
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def generate_thumbnail_pil(prompt: str, size=(256, 256)):
    """Generate a simple PIL placeholder thumbnail (no GPU needed)."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new('RGB', size, color=(30, 30, 30))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except (OSError, IOError):
        font = ImageFont.load_default()

    words = prompt.split()
    lines = []
    current = ''
    for w in words:
        test = f'{current} {w}'.strip()
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] > size[0] - 20:
            lines.append(current)
            current = w
        else:
            current = test
    if current:
        lines.append(current)

    y = max(10, (size[1] - len(lines) * 16) // 2)
    for line in lines[:8]:
        bbox = draw.textbbox((0, 0), line, font=font)
        x = (size[0] - (bbox[2] - bbox[0])) // 2
        draw.text((x, y), line, fill=(200, 180, 140), font=font)
        y += 16

    return img


def generate_with_pipeline(prompt: str, size=(256, 256)):
    """Generate a thumbnail using the actual Freyra/Freyra pipeline."""
    try:
        import modules.async_worker as worker
        import modules.config
        import random

        params = {
            'prompt': prompt,
            'negative_prompt': 'ugly, blurry, low quality, text, watermark',
            'styles': ['Freyra V2', 'SAI Photographic'],
            'performance': 'Lightning',
            'generation_steps': 4,
            'aspect_ratio': '1024*1024',
            'image_number': 1,
            'output_format': 'png',
            'seed': random.randint(0, 2**31),
            'sharpness': 1.0,
            'cfg_scale': 1.0,
            'base_model': modules.config.default_base_model_name,
            'refiner_model': 'None',
            'loras': [[True, 'None', 1.0]] * modules.config.default_max_lora_number,
            'sampler': 'dpmpp_2m_sde_gpu',
            'scheduler': 'karras',
            'disable_preview': True,
            'disable_intermediate_results': True,
        }

        task = worker.AsyncTask.from_dict(params)
        worker.async_tasks.append(task)

        start = time.time()
        while time.time() - start < 120:
            time.sleep(0.5)
            if task.yields:
                flag, product = task.yields.pop(0)
                if flag == 'finish':
                    if isinstance(product, list) and len(product) > 0:
                        from PIL import Image
                        img = Image.open(product[0]) if isinstance(product[0], str) else Image.fromarray(product[0])
                        return img.resize(size, Image.LANCZOS)
                    break

    except Exception as e:
        print(f'  Pipeline error: {e}')

    return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate thumbnails for Freyra visual picker')
    parser.add_argument('--mode', choices=['placeholder', 'pipeline'], default='placeholder',
                        help='placeholder = fast PIL text cards, pipeline = real AI thumbnails (needs GPU)')
    parser.add_argument('--categories', nargs='*', default=None,
                        help='Specific categories to generate (default: all)')
    parser.add_argument('--force', action='store_true', help='Regenerate existing thumbnails')
    args = parser.parse_args()

    categories = args.categories or CATEGORIES
    total = 0
    skipped = 0

    for cat in categories:
        filename = f'{cat}.txt'
        options = load_options(filename)
        if not options:
            print(f'[skip] {cat}: no options found')
            continue

        out_dir = os.path.join(THUMBNAIL_DIR, cat)
        os.makedirs(out_dir, exist_ok=True)

        print(f'\n[{cat}] {len(options)} options')

        for opt in options:
            s = slug(opt)
            out_path = os.path.join(out_dir, f'{s}.jpg')

            if os.path.isfile(out_path) and not args.force:
                skipped += 1
                continue

            template = CATEGORY_PROMPT_TEMPLATES.get(cat, '{option}')
            prompt = template.format(option=opt)

            if args.mode == 'pipeline':
                img = generate_with_pipeline(prompt)
                if img is None:
                    img = generate_thumbnail_pil(opt)
            else:
                img = generate_thumbnail_pil(opt)

            img.save(out_path, quality=85)
            total += 1
            print(f'  [{total}] {opt[:50]}')

    print(f'\nDone. Generated: {total}, Skipped: {skipped}')


if __name__ == '__main__':
    main()
