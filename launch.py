import os
import ssl
import sys
import warnings

# ── huggingface_hub compatibility shim ──────────────────────────────
# HfFolder was removed in huggingface_hub >=0.27 but Gradio 4.44.1 still imports it.
# Provide a lightweight stand-in so Gradio can load without forcing a hf_hub downgrade.
import huggingface_hub as _hf_hub
if not hasattr(_hf_hub, 'HfFolder'):
    class _HfFolder:
        @staticmethod
        def get_token():
            try:
                from huggingface_hub import get_token
                return get_token()
            except Exception:
                return None
        @staticmethod
        def save_token(token):
            try:
                _hf_hub.login(token=token)
            except Exception:
                pass
    _hf_hub.HfFolder = _HfFolder

# ── gradio_client JSON schema bug fix ───────────────────────────────
# gradio_client's _json_schema_to_python_type crashes when a schema value
# is a bool (e.g. additionalProperties: true) because it does `"const" in True`.
# We must patch the functions inside the module's own global namespace so that
# recursive calls within the module also go through the patched versions.
try:
    import gradio_client.utils as _gc_utils

    _original_get_type = _gc_utils.get_type

    def _safe_get_type(schema):
        if not isinstance(schema, dict):
            return "Any"
        return _original_get_type(schema)

    _original_jstpt = _gc_utils._json_schema_to_python_type

    def _safe_jstpt(schema, defs=None):
        if not isinstance(schema, dict):
            return "Any"
        return _original_jstpt(schema, defs)

    # Replace at module level so external callers use patched version
    _gc_utils.get_type = _safe_get_type
    _gc_utils._json_schema_to_python_type = _safe_jstpt

    # Also patch the module's own global dict so *internal* recursive calls
    # (which resolve names through __globals__) see the patched functions.
    _original_jstpt.__globals__['get_type'] = _safe_get_type
    _original_jstpt.__globals__['_json_schema_to_python_type'] = _safe_jstpt
except Exception:
    pass

# Suppress known harmless warnings that clutter Colab output
warnings.filterwarnings('ignore', message='You have unused kwarg parameters.*', category=UserWarning)
warnings.filterwarnings('ignore', message='Using the update method is deprecated.*', category=UserWarning)
warnings.filterwarnings('ignore', message='.*please upgrade.*', category=UserWarning)
warnings.filterwarnings('ignore', message='.*tie_word_embeddings.*')
warnings.filterwarnings('ignore', category=SyntaxWarning)

# Suppress noisy transformers logging (GPT-2 LOAD REPORT, UNEXPECTED keys)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

print('[System ARGV] ' + str(sys.argv))

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)
os.chdir(root)

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["GRADIO_CHECK_UPDATE"] = "0"
if "GRADIO_SERVER_PORT" not in os.environ:
    os.environ["GRADIO_SERVER_PORT"] = "7865"

ssl._create_default_https_context = ssl._create_unverified_context

import platform
import freyra_version

from build_launcher import build_launcher
from modules.launch_util import is_installed, run, python, run_pip, requirements_met, delete_folder_content
from modules.model_loader import load_file_from_url

REINSTALL_ALL = False
TRY_INSTALL_XFORMERS = False


def prepare_environment():
    torch_index_url = os.environ.get('TORCH_INDEX_URL', "https://download.pytorch.org/whl/cu121")
    torch_command = os.environ.get('TORCH_COMMAND',
                                   f"pip install torch==2.4.1 torchvision==0.19.1 --extra-index-url {torch_index_url}")
    requirements_file = os.environ.get('REQS_FILE', "requirements_versions.txt")

    print(f"Python {sys.version}")
    print(f"Freyra version: {freyra_version.version}")

    if REINSTALL_ALL or not is_installed("torch") or not is_installed("torchvision"):
        run(f'"{python}" -m {torch_command}', "Installing torch and torchvision", "Couldn't install torch", live=True)

    if TRY_INSTALL_XFORMERS:
        if REINSTALL_ALL or not is_installed("xformers"):
            xformers_package = os.environ.get('XFORMERS_PACKAGE', 'xformers==0.0.23')
            if platform.system() == "Windows":
                if platform.python_version().startswith("3.10"):
                    run_pip(f"install -U -I --no-deps {xformers_package}", "xformers", live=True)
                else:
                    print("Installation of xformers is not supported in this version of Python.")
                    print(
                        "You can also check this and build manually: https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Xformers#building-xformers-on-windows-by-duckness")
                    if not is_installed("xformers"):
                        exit(0)
            elif platform.system() == "Linux":
                run_pip(f"install -U -I --no-deps {xformers_package}", "xformers")

    if REINSTALL_ALL or not requirements_met(requirements_file):
        run_pip(f"install -r \"{requirements_file}\"", "requirements")

    return


vae_approx_filenames = [
    ('xlvaeapp.pth', 'https://huggingface.co/lllyasviel/misc/resolve/main/xlvaeapp.pth'),
    ('vaeapp_sd15.pth', 'https://huggingface.co/lllyasviel/misc/resolve/main/vaeapp_sd15.pt'),
    ('xl-to-v1_interposer-v4.0.safetensors',
     'https://huggingface.co/mashb1t/misc/resolve/main/xl-to-v1_interposer-v4.0.safetensors')
]


def ini_args():
    from args_manager import args
    return args


prepare_environment()
build_launcher()
args = ini_args()

if args.gpu_device_id is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_device_id)
    print("Set device to:", args.gpu_device_id)

if args.hf_mirror is not None:
    os.environ['HF_MIRROR'] = str(args.hf_mirror)
    print("Set hf_mirror to:", args.hf_mirror)

from modules import config
from modules.hash_cache import init_cache

os.environ["U2NET_HOME"] = config.path_inpaint

os.environ['GRADIO_TEMP_DIR'] = config.temp_path

if config.temp_path_cleanup_on_launch:
    print(f'[Cleanup] Attempting to delete content of temp dir {config.temp_path}')
    result = delete_folder_content(config.temp_path, '[Cleanup] ')
    if result:
        print("[Cleanup] Cleanup successful")
    else:
        print(f"[Cleanup] Failed to delete content of temp dir.")


def download_models(default_model, previous_default_models, checkpoint_downloads, embeddings_downloads, lora_downloads, vae_downloads):
    from modules.util import get_file_from_folder_list

    for file_name, url in vae_approx_filenames:
        load_file_from_url(url=url, model_dir=config.path_vae_approx, file_name=file_name)

    load_file_from_url(
        url='https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_expansion.bin',
        model_dir=config.path_freyra_expansion,
        file_name='pytorch_model.bin'
    )

    if args.disable_preset_download:
        print('Skipped model download.')
        return default_model, checkpoint_downloads

    if not args.always_download_new_model:
        if not os.path.isfile(get_file_from_folder_list(default_model, config.paths_checkpoints)):
            for alternative_model_name in previous_default_models:
                if os.path.isfile(get_file_from_folder_list(alternative_model_name, config.paths_checkpoints)):
                    print(f'You do not have [{default_model}] but you have [{alternative_model_name}].')
                    print(f'Freyra will use [{alternative_model_name}] to avoid downloading new models, '
                          f'but you are not using the latest models.')
                    print('Use --always-download-new-model to avoid fallback and always get new models.')
                    checkpoint_downloads = {}
                    default_model = alternative_model_name
                    break

    for file_name, url in checkpoint_downloads.items():
        model_dir = os.path.dirname(get_file_from_folder_list(file_name, config.paths_checkpoints))
        load_file_from_url(url=url, model_dir=model_dir, file_name=file_name)
    for file_name, url in embeddings_downloads.items():
        load_file_from_url(url=url, model_dir=config.path_embeddings, file_name=file_name)
    for file_name, url in lora_downloads.items():
        model_dir = os.path.dirname(get_file_from_folder_list(file_name, config.paths_loras))
        load_file_from_url(url=url, model_dir=model_dir, file_name=file_name)
    for file_name, url in vae_downloads.items():
        load_file_from_url(url=url, model_dir=config.path_vae, file_name=file_name)

    return default_model, checkpoint_downloads


config.default_base_model_name, config.checkpoint_downloads = download_models(
    config.default_base_model_name, config.previous_default_models, config.checkpoint_downloads,
    config.embeddings_downloads, config.lora_downloads, config.vae_downloads)

config.update_files()
init_cache(config.model_filenames, config.paths_checkpoints, config.lora_filenames, config.paths_loras)

# Download InsightFace inswapper model for face swap (non-fatal)
inswapper_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'insightface')
os.makedirs(inswapper_dir, exist_ok=True)
inswapper_path = os.path.join(inswapper_dir, 'inswapper_128.onnx')
if not os.path.isfile(inswapper_path):
    _inswapper_urls = [
        'https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx',
        'https://huggingface.co/thebiglaskowski/inswapper_128.onnx/resolve/main/inswapper_128.onnx',
    ]
    for _url in _inswapper_urls:
        try:
            load_file_from_url(url=_url, model_dir=inswapper_dir, file_name='inswapper_128.onnx')
            break
        except Exception as _e:
            print(f'[FaceSwap] Download failed from {_url}: {_e}')
    else:
        print('[FaceSwap] Could not download inswapper_128.onnx -- face swap will be disabled.')

if args.legacy_ui:
    import webui
else:
    from ui.app import launch_app
    launch_app()
