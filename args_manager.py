import ldm_patched.modules.args_parser as args_parser

args_parser.parser.add_argument("--share", action='store_true', help="Set whether to share on Gradio.")

args_parser.parser.add_argument("--preset", type=str, default=None, help="Apply specified UI preset.")
args_parser.parser.add_argument("--disable-preset-selection", action='store_true',
                                help="Disables preset selection in Gradio.")

args_parser.parser.add_argument("--language", type=str, default='default',
                                help="Translate UI using json files in [language] folder. "
                                  "For example, [--language example] will use [language/example.json] for translation.")

# For example, https://github.com/lllyasviel/Freyra/issues/849
args_parser.parser.add_argument("--disable-offload-from-vram", action="store_true",
                                help="Force loading models to vram when the unload can be avoided. "
                                  "Some Mac users may need this.")

args_parser.parser.add_argument("--theme", type=str, help="launches the UI with light or dark theme", default=None)
args_parser.parser.add_argument("--disable-image-log", action='store_true',
                                help="Prevent writing images and logs to the outputs folder.")

args_parser.parser.add_argument("--disable-analytics", action='store_true',
                                help="Disables analytics for Gradio.")

args_parser.parser.add_argument("--disable-metadata", action='store_true',
                                help="Disables saving metadata to images.")

args_parser.parser.add_argument("--disable-preset-download", action='store_true',
                                help="Disables downloading models for presets", default=False)

args_parser.parser.add_argument("--disable-enhance-output-sorting", action='store_true',
                                help="Disables enhance output sorting for final image gallery.")

args_parser.parser.add_argument("--enable-auto-describe-image", action='store_true',
                                help="Enables automatic description of uov and enhance image when prompt is empty", default=False)

args_parser.parser.add_argument("--always-download-new-model", action='store_true',
                                help="Always download newer models", default=False)

args_parser.parser.add_argument("--rebuild-hash-cache", help="Generates missing model and LoRA hashes.",
                                type=int, nargs="?", metavar="CPU_NUM_THREADS", const=-1)

args_parser.parser.add_argument("--tunnel", type=str, default='gradio',
                                choices=['gradio', 'cloudflared', 'both'],
                                help="Tunnel method for public URL: gradio (default), cloudflared, or both.")

args_parser.parser.add_argument("--legacy-ui", action='store_true',
                                help="Launch the original general-purpose UI instead of the influencer studio.")

# --- Phase 5: Generation default overrides via CLI ---
args_parser.parser.add_argument("--default-prompt", type=str, default=None,
                                help="Override default prompt text.")
args_parser.parser.add_argument("--default-negative", type=str, default=None,
                                help="Override default negative prompt text.")
args_parser.parser.add_argument("--default-steps", type=int, default=None,
                                help="Override default step count (ignores performance mode steps).")
args_parser.parser.add_argument("--default-cfg", type=float, default=None,
                                help="Override default CFG scale.")
args_parser.parser.add_argument("--default-sampler", type=str, default=None,
                                help="Override default sampler name.")
args_parser.parser.add_argument("--default-scheduler", type=str, default=None,
                                help="Override default scheduler name.")
args_parser.parser.add_argument("--default-resolution", type=str, default=None,
                                help="Override default aspect ratio, e.g. '896*1152'.")
args_parser.parser.add_argument("--image-number", type=int, default=None,
                                help="Override default number of images per generation.")
args_parser.parser.add_argument("--seed", type=int, default=None,
                                help="Override default seed (-1 for random).")
args_parser.parser.add_argument("--default-loras", type=str, default=None,
                                help='Override LoRAs as JSON, e.g. \'[[true,"lora.safetensors",0.5]]\'.')

args_parser.parser.set_defaults(
    disable_cuda_malloc=True,
    in_browser=True,
    port=None
)

args_parser.args = args_parser.parser.parse_args()

# (Disable by default because of issues like https://github.com/lllyasviel/Freyra/issues/724)
args_parser.args.always_offload_from_vram = not args_parser.args.disable_offload_from_vram

if args_parser.args.disable_analytics:
    import os
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

if args_parser.args.disable_in_browser:
    args_parser.args.in_browser = False

args = args_parser.args
