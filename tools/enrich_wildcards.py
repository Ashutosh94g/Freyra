"""Enrich Freyra wildcard files with more creative options.

Reads existing wildcard .txt files from the wildcards/ directory,
generates additional entries using curated category-specific expansions,
deduplicates, and writes the expanded files.

Usage:
    python tools/enrich_wildcards.py [--dry-run]

The script can be run repeatedly; it always deduplicates against existing entries.
"""

import os
import sys
import argparse

WILDCARDS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'wildcards')

EXPANSIONS = {
    'influencer_poses.txt': [
        'standing with one hand behind head',
        'sitting on floor with knees drawn up',
        'walking with arms swinging naturally',
        'leaning against a railing looking into distance',
        'sitting on a high stool legs crossed',
        'standing with weight shifted to one hip',
        'stretching one arm across chest',
        'kneeling on one knee',
        'standing with hands clasped behind back',
        'sitting sideways on a chair',
        'standing with feet shoulder-width apart arms relaxed',
        'twisting torso looking over left shoulder',
        'reaching up adjusting sunglasses on head',
        'walking away from camera with natural stride',
        'seated on ground leaning back on both palms',
        'standing on tiptoes stretching upward',
        'crossed ankles leaning back against wall',
        'standing with arms loosely folded',
        'sitting with one leg tucked under',
        'hands in pockets walking casually',
        'seated with legs extended and ankles crossed',
        'lunging forward in athletic stance',
        'standing with one foot resting on step',
        'twisting at waist with arms extended',
        'bending slightly forward at waist hands on knees',
        'standing in contrapposto classical pose',
        'sitting on edge of table legs dangling',
        'crouching down with one hand touching ground',
        'standing arms akimbo both fists on hips',
        'reclining on side head propped on hand',
        'standing with back to camera looking over shoulder',
        'seated in lotus position on cushion',
        'standing with hand resting on doorframe',
        'walking up stairs looking at camera',
        'sitting on swing legs extended forward',
        'standing in warrior yoga pose',
        'leaning forward elbows on knees while seated',
        'mid-stride frozen in motion',
        'standing with one arm raised overhead',
        'sitting on windowsill legs drawn up',
        'standing with arms outstretched to sides',
        'perched on bar stool one foot on footrest',
        'kneeling both knees hands on thighs',
        'standing pigeon-toed playful stance',
        'seated on ground cross-legged leaning forward',
        'standing with arms behind head elbows out',
        'walking on tiptoes playfully',
        'sitting with chin resting on hand elbow on knee',
        'standing holding jacket over one shoulder',
        'leaning against car door with arms crossed',
        'standing with one hand touching necklace',
        'seated with one leg pulled up on chair',
        'standing facing away with head turned right',
        'running in place with knees high',
        'lying on stomach chin on hands',
        'standing with gentle S-curve body line',
        'arms behind back holding opposite elbows',
        'stepping forward with purpose',
        'seated on floor legs spread in V shape',
        'standing with hands on lower back stretching',
    ],

    'influencer_outfits.txt': [
        'oversized blazer with matching wide-leg trousers',
        'white cropped tank top with pleated midi skirt',
        'denim-on-denim jacket and jeans combo',
        'sleek black catsuit with statement belt',
        'cashmere turtleneck with leather trousers',
        'high-waisted paper bag shorts with tucked blouse',
        'pastel tie-dye co-ord set',
        'structured peplum top with pencil skirt',
        'boho maxi dress with embroidered details',
        'athletic crop top with matching bike shorts',
        'velvet blazer over silk slip dress',
        'classic white button-down with tailored chinos',
        'cable-knit sweater with plaid mini skirt',
        'sheer mesh top layered over bralette with cargo pants',
        'vintage band tee knotted at waist with jeans',
        'sleeveless turtleneck with high-waisted leather shorts',
        'flowing kaftan with gold belt cinched at waist',
        'pinstripe suit with cropped trousers and corset belt',
        'crochet halter top with linen palazzo pants',
        'utility jumpsuit in olive drab',
        'satin wrap top with sequin mini skirt',
        'oversized denim shirt worn as dress with belt',
        'wool cape coat over turtleneck and jeans',
        'ribbed unitard with oversized cardigan',
        'asymmetric one-shoulder dress in cobalt blue',
        'matching sweatshirt and jogger set in pastel pink',
        'tailored vest over white t-shirt with trousers',
        'flowy bohemian printed wrap skirt with crop top',
        'high-neck sleeveless bodysuit with wide-leg trousers',
        'leather moto jacket over floral sundress',
        'monochrome all-black turtleneck and skirt combo',
        'off-shoulder ruched bodycon dress in emerald',
        'oversized flannel shirt tied at waist over shorts',
        'silk pajama-style shirt and trousers in burgundy',
        'tweed cropped jacket with matching mini skirt',
        'cut-out detail bodycon dress in white',
        'mesh overlay dress with nude underlay',
        'oversized hoodie dress with knee-high boots',
        'structured blazer dress in camel',
        'color-block sweater with contrast trim jeans',
        'flowing chiffon maxi dress in sunset ombre',
        'quilted vest over long-sleeve henley with leggings',
        'backless halter top with high-waisted tailored pants',
        'oversized trench coat worn as dress',
        'knit crop cardigan with matching wide-leg pants',
        'latex pencil skirt with fitted turtleneck',
        'distressed boyfriend jeans with crisp white blazer',
        'tiered ruffle midi skirt with fitted tank top',
        'fishnet tights under denim cutoffs with band tee',
        'structured corset top with flowing palazzo pants',
        'minimalist tank dress in sand beige',
        'vintage high-waisted mom jeans with tucked polo',
        'iridescent holographic mini dress',
        'longline linen vest over bralet with wide pants',
        'color-pop neon green sports bra with black leggings',
        'classic little black dress with pearl necklace',
        'chunky cable knit jumper dress in cream',
        'sequin blazer with plain black trousers',
        'tennis-core pleated white skirt and polo',
        'sheer floral blouse with high-waisted leather pants',
    ],

    'influencer_lighting.txt': [
        'Rembrandt lighting with shadow triangle on cheek',
        'butterfly lighting creating shadow under nose',
        'split lighting half face illuminated half in shadow',
        'broad lighting with main face toward camera brightly lit',
        'short lighting with main face away from camera in shadow',
        'loop lighting with small nose shadow on opposite cheek',
        'high key bright even lighting minimal shadows',
        'low key dramatic lighting with deep shadows',
        'clamshell lighting softbox above and reflector below',
        'hair light behind subject creating glow on edges',
        'practical lighting from visible lamp or screen in frame',
        'cross lighting two sources on opposite sides',
        'beauty dish centered overhead with fill below',
        'tungsten warm interior lighting with orange cast',
        'flash fill outdoors balancing sun with strobe',
        'silhouette backlighting subject entirely in shadow',
        'colored gel lighting red and blue bi-color',
        'natural window light diffused through sheer curtain',
        'street lamp warm pool of light at night',
        'overhead skylight soft directional from above',
        'late afternoon warm side light through blinds',
        'reflected light from water surface below',
        'overcast flat lighting for even skin tones',
        'morning mist soft diffused atmospheric light',
        'bright direct midday sun with hard shadows',
        'mixed color temperature warm and cool in same frame',
        'campfire flickering warm orange glow from below',
        'stadium or concert spotlighting from above',
        'car headlights dramatic low angle front light',
        'fairy lights soft warm bokeh background dots',
        'fluorescent greenish overhead indoor light',
        'sunrise pink and orange directional light',
        'moonlight cool blue ambient night tone',
        'studio strobe with barn doors creating strip of light',
        'umbrella bounce soft wrap-around fill',
        'spotlight single hard beam on face',
        'softbox with grid creating contained soft light',
        'projection gobo creating pattern shadows',
        'natural fire pit warm flickering ambient',
        'LED panel adjustable color temperature',
    ],

    'influencer_settings.txt': [
        'industrial warehouse with exposed steel beams',
        'cozy cabin interior with stone fireplace',
        'colorful street mural as backdrop',
        'lavender field in full bloom at sunset',
        'modern kitchen with marble island countertop',
        'library with floor-to-ceiling bookshelves',
        'cherry blossom tree-lined path in spring',
        'underground parking garage with cinematic lighting',
        'sunflower field at golden hour',
        'rooftop bar with city skyline behind',
        'vintage record shop with vinyl displays',
        'zen garden with raked sand and bonsai',
        'underground subway platform with arriving train',
        'crystal clear lake with mountain reflection',
        'luxury car interior leather seats',
        'open air market with colorful fabric stalls',
        'old town square with fountain and pigeons',
        'bamboo forest with shafts of sunlight',
        'neon arcade interior with game machines',
        'art deco hotel lobby with chandelier',
        'seaside cliff with crashing waves below',
        'ivy-covered stone wall cottage exterior',
        'modern art museum with geometric sculptures',
        'tropical waterfall with mist and rocks',
        'chic restaurant with dim ambient lighting',
        'autumn park with red and gold fallen leaves',
        'vintage train car interior',
        'skyscraper observation deck with panoramic glass',
        'outdoor cinema setup at twilight',
        'traditional Japanese temple garden',
        'industrial brick alley with fire escape stairs',
        'snow-covered pine forest trail',
        'palm-lined boulevard with classic cars',
        'minimalist concrete gallery space',
        'floating dock on calm lake at dawn',
    ],

    'influencer_makeup.txt': [
        'K-beauty inspired glass skin with gradient lip tint',
        'matte no-makeup makeup with fluffy brows',
        'retro 60s mod look with bold white liner and pale lip',
        'grunge aesthetic smudged liner and dark berry lip',
        'glowy sunkissed look with freckle tint and peach blush',
        'vampire aesthetic dark red lip and porcelain skin',
        'tropical fruit-inspired coral blush and tangerine lip',
        'monochromatic mauve look on eyes cheeks and lips',
        'ice queen aesthetic silver highlight and pale mauve lip',
        'warm terracotta tones on eyes and lips earthy glow',
        'peachy keen soft peach eyeshadow and matching lip',
        'sultry bronze smoky eye with deep brown lip',
        'fresh lavender eyeshadow with clear gloss and pink blush',
        'golden goddess highlighted skin with gold eyeshadow',
        'punk rock spiked liner with black lip',
        'soft goth dark plum eye with black cherry lip',
        'radiant luminous skin with cream highlight and berry tint',
        'barely-there makeup dewy SPF base and tinted lip oil',
        'copper and rose gold eye look with nude lip',
        'classic Hollywood red carpet red lip and subtle eye',
        'pastel fairy shimmer eyeshadow with glossy pink lip',
        'tribal-inspired artistic face paint accents',
        'sapphire blue eyeshadow with matching blue liner and nude lip',
        'honey-toned warm eyeshadow with caramel lip color',
        'elf-core ethereal shimmer with pointed highlight and blush',
    ],

    'influencer_expressions.txt': [
        'sly knowing smile like sharing a secret',
        'wide-eyed wonder and amazement',
        'brooding introspective gaze',
        'hearty belly laugh with eyes squeezed shut',
        'cool detached model stare',
        'dreamy faraway look',
        'coy smile with lowered gaze',
        'triumphant victorious expression',
        'soft tender loving expression',
        'challenging defiant stare',
        'relaxed closed-eyes peaceful face',
        'excited anticipation with bright eyes',
        'mischievous grin with raised eyebrow',
        'quiet contentment with subtle smile',
        'intense passionate gaze',
        'amused smirk trying not to laugh',
        'elegant composed resting face',
        'energetic enthusiastic smile',
        'pensive slight frown of concentration',
        'flirty side glance with half smile',
        'powerful commanding presence',
        'innocent doe-eyed expression',
        'wistful nostalgic expression',
        'fierce runway model sharp cheekbones expression',
        'genuine candid mid-laugh expression',
        'regal dignified slight chin raise',
        'vulnerable emotional glistening eyes',
        'carefree wind-in-hair blissful smile',
        'stoic emotionless editorial expression',
        'warm motherly nurturing smile',
    ],

    'influencer_hair.txt': [
        'loose romantic updo with face-framing tendrils',
        'textured lob with beachy waves',
        'sleek center-part low bun',
        'voluminous 70s feathered layers',
        'double Dutch braids into low ponytail',
        'vintage pin curls framing face',
        'crown braid wrapped around head',
        'choppy layered bob with side bangs',
        'high bun with loose strands',
        'micro braids pulled back',
        'retro finger waves side-parted',
        'loose fishtail braid over one shoulder',
        'straight with middle part face-framing layers',
        'bubble ponytail with hair cuffs',
        'twisted half updo with pins',
        'messy textured low ponytail',
        'blunt cut shoulder length with volume',
        'naturally curly wash-and-go defined coils',
        'cornrows into high ponytail',
        'soft curled Hollywood waves side-swept',
        'textured pixie with tapered sides',
        'waterfall braid with loose curls',
        'high sleek ponytail with wrapped base',
        'natural afro shaped and picked out',
        'two strand twists pinned up',
    ],

    'influencer_footwear.txt': [
        'classic white leather sneakers',
        'clear perspex heels',
        'cowboy boots with pointed toe',
        'velvet mules in jewel tone',
        'lace-up block heel sandals',
        'platform mary janes',
        'suede over-the-knee boots',
        'minimalist leather slide sandals',
        'embellished crystal-studded heels',
        'retro platform sneakers with chunky sole',
        'classic black pumps with red sole',
        'woven leather huarache sandals',
        'hiking boots with thick lug sole',
        'satin kitten heel mules',
        'two-strap Birkenstock sandals',
        'high-top canvas sneakers',
        'gold metallic strappy heels',
        'fur-lined slip-on loafers',
        'patent leather ankle boots',
        'athletic training shoes with neon accents',
    ],

    'influencer_camera_angles.txt': [
        'bird eye view from directly above',
        'ground level looking straight up',
        'dramatic worm eye angle from floor',
        'hip level candid street shot angle',
        'overhead 45 degree downward angle',
        'level eye contact straight-on composition',
        'shot through foreground object creating depth',
        'panning motion blur with sharp subject',
        'reflected in mirror creating double image',
        'shot from behind through doorway framing subject',
        'tilted 15 degrees for dynamic energy',
        'extreme wide establishing shot with subject small',
        'tight crop face filling entire frame',
        'shot from adjacent room through window',
        'captured from waist level upward power angle',
    ],
}


def read_existing(filepath: str) -> set[str]:
    if not os.path.exists(filepath):
        return set()
    with open(filepath, encoding='utf-8') as f:
        return {line.strip().lower() for line in f if line.strip()}


def enrich_file(filename: str, new_entries: list[str], dry_run: bool = False) -> int:
    filepath = os.path.join(WILDCARDS_DIR, filename)
    existing = read_existing(filepath)

    with open(filepath, encoding='utf-8') as f:
        original_lines = [line.rstrip('\n') for line in f if line.strip()]

    added = 0
    for entry in new_entries:
        if entry.strip().lower() not in existing:
            original_lines.append(entry.strip())
            existing.add(entry.strip().lower())
            added += 1

    if not dry_run and added > 0:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(original_lines) + '\n')

    return added


def main():
    parser = argparse.ArgumentParser(description='Enrich Freyra wildcard files')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be added without writing')
    args = parser.parse_args()

    total_added = 0
    for filename, entries in EXPANSIONS.items():
        filepath = os.path.join(WILDCARDS_DIR, filename)
        if not os.path.exists(filepath):
            print(f'  SKIP {filename} (file not found)')
            continue

        before = len(read_existing(filepath))
        added = enrich_file(filename, entries, dry_run=args.dry_run)
        total_added += added

        action = 'would add' if args.dry_run else 'added'
        print(f'  {filename}: {before} existing, {action} {added} new -> {before + added} total')

    print(f'\nTotal entries {"would be " if args.dry_run else ""}added: {total_added}')


if __name__ == '__main__':
    main()
