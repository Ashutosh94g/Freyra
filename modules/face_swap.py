"""Freyra Face Swap -- high-quality face replacement pipeline.

Pipeline: source face + target image -> detect -> swap -> parse mask ->
          color match -> seamless blend -> GFPGAN restore.

All processing runs on CPU to preserve GPU VRAM for diffusion.

Requires: insightface>=0.7.3, onnxruntime>=1.16.0
Model: inswapper_128.onnx (~540MB) in models/insightface/
Optional: GFPGANv1.4.pth (~340MB) auto-downloaded to models/gfpgan/
"""

import os
import cv2
import numpy as np
import torch

import modules.config

_analyser = None
_swapper = None
_face_parser = None
_gfpgan_model = None

INSWAPPER_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 'models', 'insightface', 'inswapper_128.onnx')
)

GFPGAN_MODEL_URL = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
GFPGAN_MODEL_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 'models', 'gfpgan')
)

# BiSeNet face parsing class indices (19-class CelebAMask-HQ labels):
# 1=skin, 2=l_brow, 3=r_brow, 4=l_eye, 5=r_eye, 6=eye_g(lasses),
# 7=l_ear, 8=r_ear, 9=ear_r(ing), 10=nose, 11=mouth, 12=u_lip, 13=l_lip
_FACE_PARSE_CLASSES = {1, 2, 3, 4, 5, 6, 10, 11, 12, 13}


def is_available() -> bool:
    """Check if face swap dependencies and model are available."""
    try:
        import insightface  # noqa: F401
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return os.path.isfile(INSWAPPER_PATH)


def get_face_analyser():
    """Get or create a cached InsightFace face analyser (CPU)."""
    global _analyser
    if _analyser is not None:
        return _analyser

    import insightface
    _analyser = insightface.app.FaceAnalysis(
        name='buffalo_l',
        root=os.path.dirname(INSWAPPER_PATH),
        providers=['CPUExecutionProvider'],
    )
    _analyser.prepare(ctx_id=-1, det_size=(640, 640))
    return _analyser


def get_face_swapper():
    """Get or create a cached inswapper model (CPU)."""
    global _swapper
    if _swapper is not None:
        return _swapper

    import insightface
    _swapper = insightface.model_zoo.get_model(
        INSWAPPER_PATH,
        providers=['CPUExecutionProvider'],
    )
    return _swapper


def _get_largest_face(faces):
    """Return the face with the largest bounding box area."""
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


# ---------------------------------------------------------------------------
# Face parsing mask (BiSeNet)
# ---------------------------------------------------------------------------

def _get_face_parser():
    """Load and cache the BiSeNet face parsing model on CPU."""
    global _face_parser
    if _face_parser is not None:
        return _face_parser

    try:
        from extras.facexlib.parsing import init_parsing_model
        _face_parser = init_parsing_model(model_name='bisenet', device='cpu')
        return _face_parser
    except Exception as e:
        print(f'[FaceSwap] Could not load face parser: {e}')
        return None


def _get_face_parse_mask(image_bgr: np.ndarray, face_bbox: np.ndarray,
                         expand_ratio: float = 0.2) -> np.ndarray | None:
    """Generate a precise face-region mask using BiSeNet parsing.

    Returns a float32 mask (0-1) the same size as image_bgr, or None on failure.
    """
    parser = _get_face_parser()
    if parser is None:
        return None

    try:
        h, w = image_bgr.shape[:2]
        bbox = face_bbox.astype(int)
        x1, y1, x2, y2 = bbox[:4]

        pad_w = int((x2 - x1) * expand_ratio)
        pad_h = int((y2 - y1) * expand_ratio)
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)

        face_crop = image_bgr[y1:y2, x1:x2]
        if face_crop.size == 0:
            return None

        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)

        input_tensor = torch.from_numpy(face_resized.transpose(2, 0, 1)).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        input_tensor = (input_tensor - mean) / std
        input_tensor = input_tensor.unsqueeze(0)

        with torch.no_grad():
            out = parser(input_tensor)
            if isinstance(out, (tuple, list)):
                out = out[0]
            parsing = out.squeeze(0).argmax(0).cpu().numpy()

        face_mask_512 = np.zeros((512, 512), dtype=np.float32)
        for cls_id in _FACE_PARSE_CLASSES:
            face_mask_512[parsing == cls_id] = 1.0

        face_mask_crop = cv2.resize(face_mask_512, (x2 - x1, y2 - y1),
                                    interpolation=cv2.INTER_LINEAR)

        ksize = max(3, int(min(x2 - x1, y2 - y1) * 0.04) | 1)
        face_mask_crop = cv2.GaussianBlur(face_mask_crop, (ksize, ksize), 0)

        full_mask = np.zeros((h, w), dtype=np.float32)
        full_mask[y1:y2, x1:x2] = face_mask_crop

        return full_mask

    except Exception as e:
        print(f'[FaceSwap] Face parsing failed: {e}')
        return None


# ---------------------------------------------------------------------------
# Color matching
# ---------------------------------------------------------------------------

def _match_face_color(swapped_bgr: np.ndarray, original_bgr: np.ndarray,
                      mask: np.ndarray) -> np.ndarray:
    """Match the color/lighting of the swapped face region to the original.

    Uses per-channel mean/std transfer within the masked face region.
    """
    mask_bool = mask > 0.5
    if not np.any(mask_bool):
        return swapped_bgr

    result = swapped_bgr.copy().astype(np.float32)

    for c in range(3):
        src_pixels = result[:, :, c][mask_bool]
        tgt_pixels = original_bgr[:, :, c][mask_bool].astype(np.float32)

        src_mean, src_std = src_pixels.mean(), max(src_pixels.std(), 1e-6)
        tgt_mean, tgt_std = tgt_pixels.mean(), max(tgt_pixels.std(), 1e-6)

        normalized = (result[:, :, c] - src_mean) / src_std
        transferred = normalized * tgt_std + tgt_mean

        blend_factor = 0.6
        blended = result[:, :, c] * (1 - blend_factor) + transferred * blend_factor

        mask_float = mask[:, :, np.newaxis] if mask.ndim < 3 else mask
        result[:, :, c] = np.where(mask_bool, blended, result[:, :, c])

    return np.clip(result, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Seamless blending (Poisson)
# ---------------------------------------------------------------------------

def _seamless_blend(swapped_bgr: np.ndarray, original_bgr: np.ndarray,
                    face_bbox: np.ndarray, parse_mask: np.ndarray | None) -> np.ndarray:
    """Blend the swapped face into the original using cv2.seamlessClone.

    Falls back to feathered alpha blending if seamlessClone fails.
    """
    h, w = original_bgr.shape[:2]
    bbox = face_bbox.astype(int)
    cx = int((bbox[0] + bbox[2]) / 2)
    cy = int((bbox[1] + bbox[3]) / 2)
    cx = max(1, min(w - 2, cx))
    cy = max(1, min(h - 2, cy))

    if parse_mask is not None:
        mask_uint8 = (parse_mask * 255).astype(np.uint8)
    else:
        mask_uint8 = np.zeros((h, w), dtype=np.uint8)
        x1 = max(0, bbox[0])
        y1 = max(0, bbox[1])
        x2 = min(w, bbox[2])
        y2 = min(h, bbox[3])
        mask_uint8[y1:y2, x1:x2] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_uint8 = cv2.dilate(mask_uint8, kernel, iterations=1)

    if np.sum(mask_uint8 > 0) < 100:
        return swapped_bgr

    try:
        result = cv2.seamlessClone(
            swapped_bgr, original_bgr, mask_uint8,
            (cx, cy), cv2.NORMAL_CLONE
        )
        return result
    except Exception:
        return _fallback_feather_blend(swapped_bgr, original_bgr, mask_uint8)


def _fallback_feather_blend(swapped: np.ndarray, original: np.ndarray,
                            mask_uint8: np.ndarray) -> np.ndarray:
    """Alpha blending fallback with Gaussian feathering."""
    mask = mask_uint8.astype(np.float32) / 255.0
    ksize = 21
    mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)
    mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)
    mask_3ch = mask[:, :, np.newaxis]
    result = (swapped.astype(np.float32) * mask_3ch +
              original.astype(np.float32) * (1 - mask_3ch))
    return np.clip(result, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Legacy blending (kept for backward compat when enhanced pipeline is off)
# ---------------------------------------------------------------------------

def _smooth_face_boundary(swapped, original, face, feather_amount=11):
    """Blend the swapped face region back into the original with feathered edges."""
    bbox = face.bbox.astype(int)
    x1, y1, x2, y2 = bbox
    h, w = original.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    pad = int(min(x2 - x1, y2 - y1) * 0.15)
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    mask = np.zeros((h, w), dtype=np.float32)
    mask[y1:y2, x1:x2] = 1.0

    ksize = feather_amount * 2 + 1
    mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)
    mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)

    mask_3ch = mask[:, :, np.newaxis]
    result = (swapped.astype(np.float32) * mask_3ch +
              original.astype(np.float32) * (1 - mask_3ch))
    return np.clip(result, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# GFPGAN face restoration
# ---------------------------------------------------------------------------

def _load_gfpgan():
    """Load and cache the GFPGAN v1.4 model on CPU."""
    global _gfpgan_model
    if _gfpgan_model is not None:
        return _gfpgan_model

    model_path = os.path.join(GFPGAN_MODEL_DIR, 'GFPGANv1.4.pth')

    if not os.path.isfile(model_path):
        try:
            from modules.model_loader import load_file_from_url
            print('[FaceSwap] Downloading GFPGAN v1.4 model...')
            load_file_from_url(
                url=GFPGAN_MODEL_URL,
                model_dir=GFPGAN_MODEL_DIR,
                file_name='GFPGANv1.4.pth',
            )
        except Exception as e:
            print(f'[FaceSwap] Failed to download GFPGAN: {e}')
            return None

    if not os.path.isfile(model_path):
        return None

    try:
        from ldm_patched.pfn.architecture.face.gfpganv1_clean_arch import GFPGANv1Clean

        state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        if 'params_ema' in state_dict:
            state_dict = state_dict['params_ema']
        elif 'params' in state_dict:
            state_dict = state_dict['params']

        model = GFPGANv1Clean(state_dict)
        model.eval()
        model = model.to('cpu')
        _gfpgan_model = model
        print('[FaceSwap] GFPGAN v1.4 loaded on CPU')
        return _gfpgan_model
    except Exception as e:
        print(f'[FaceSwap] Failed to load GFPGAN model: {e}')
        return None


def _restore_face_gfpgan(image_bgr: np.ndarray, face_bbox: np.ndarray) -> np.ndarray:
    """Apply GFPGAN face restoration to the face region in the image.

    Crops the face, runs GFPGAN at 512x512, and pastes back.
    """
    model = _load_gfpgan()
    if model is None:
        return image_bgr

    h, w = image_bgr.shape[:2]
    bbox = face_bbox.astype(int)
    x1, y1, x2, y2 = bbox[:4]

    pad = int(max(x2 - x1, y2 - y1) * 0.3)
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    face_crop = image_bgr[y1:y2, x1:x2].copy()
    if face_crop.size == 0:
        return image_bgr

    crop_h, crop_w = face_crop.shape[:2]

    face_input = cv2.resize(face_crop, (512, 512), interpolation=cv2.INTER_LINEAR)
    face_input = face_input.astype(np.float32) / 255.0
    face_input = (face_input - 0.5) / 0.5

    input_tensor = torch.from_numpy(face_input.transpose(2, 0, 1)).unsqueeze(0).float()

    try:
        with torch.no_grad():
            output, _ = model(input_tensor, return_rgb=False)
            output = output.squeeze(0).clamp(-1, 1)
            output = (output + 1) / 2.0
            restored_face = output.permute(1, 2, 0).cpu().numpy()
            restored_face = (restored_face * 255).astype(np.uint8)
    except Exception as e:
        print(f'[FaceSwap] GFPGAN inference failed: {e}')
        return image_bgr

    restored_face = cv2.resize(restored_face, (crop_w, crop_h),
                               interpolation=cv2.INTER_LANCZOS4)

    # Blend restored face back using a soft elliptical mask
    blend_mask = np.zeros((crop_h, crop_w), dtype=np.float32)
    center = (crop_w // 2, crop_h // 2)
    axes = (int(crop_w * 0.38), int(crop_h * 0.42))
    cv2.ellipse(blend_mask, center, axes, 0, 0, 360, 1.0, -1)
    blend_mask = cv2.GaussianBlur(blend_mask, (31, 31), 0)
    blend_mask_3ch = blend_mask[:, :, np.newaxis]

    blended = (restored_face.astype(np.float32) * blend_mask_3ch +
               face_crop.astype(np.float32) * (1 - blend_mask_3ch))

    result = image_bgr.copy()
    result[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    return result


# ---------------------------------------------------------------------------
# Main face swap entry points
# ---------------------------------------------------------------------------

def swap_face(
    source_img: np.ndarray,
    target_img: np.ndarray,
    enhance_blend: bool = True,
    color_match: bool = True,
    face_restore: bool = True,
    enhanced_blending: bool = True,
) -> np.ndarray | None:
    """Swap the face from source_img onto target_img.

    Parameters
    ----------
    source_img : np.ndarray
        Reference face image (RGB, HWC).
    target_img : np.ndarray
        Image to receive the face (RGB, HWC).
    enhance_blend : bool
        Legacy parameter. When True and enhanced_blending is False, uses
        Gaussian feathering. Kept for backward compatibility.
    color_match : bool
        Match swapped face color/lighting to the target face region.
    face_restore : bool
        Apply GFPGAN face restoration after swapping.
    enhanced_blending : bool
        Use face-parsing mask + Poisson seamless clone instead of
        basic Gaussian feathering.

    Returns
    -------
    np.ndarray or None
        Modified target with swapped face, or None on failure.
    """
    if not is_available():
        return None

    try:
        analyser = get_face_analyser()
        swapper = get_face_swapper()

        src_bgr = cv2.cvtColor(source_img, cv2.COLOR_RGB2BGR)
        tgt_bgr = cv2.cvtColor(target_img, cv2.COLOR_RGB2BGR)

        src_faces = analyser.get(src_bgr)
        tgt_faces = analyser.get(tgt_bgr)

        source_face = _get_largest_face(src_faces)
        target_face = _get_largest_face(tgt_faces)

        if source_face is None or target_face is None:
            return None

        # Step 1: InsightFace swap
        result_bgr = swapper.get(tgt_bgr, target_face, source_face, paste_back=True)

        # Step 2-5: Enhanced pipeline
        if enhanced_blending:
            parse_mask = _get_face_parse_mask(result_bgr, target_face.bbox)

            if color_match:
                result_bgr = _match_face_color(result_bgr, tgt_bgr, parse_mask
                                               if parse_mask is not None
                                               else np.zeros(tgt_bgr.shape[:2], dtype=np.float32))

            result_bgr = _seamless_blend(result_bgr, tgt_bgr, target_face.bbox, parse_mask)
        elif enhance_blend:
            result_bgr = _smooth_face_boundary(result_bgr, tgt_bgr, target_face)

        # Step 6: GFPGAN restoration
        if face_restore:
            result_bgr = _restore_face_gfpgan(result_bgr, target_face.bbox)

        return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    except Exception as e:
        print(f'[FaceSwap] Error: {e}')
        return None


def swap_face_batch(
    source_img: np.ndarray,
    target_imgs: list[np.ndarray],
    enhance_blend: bool = True,
    color_match: bool = True,
    face_restore: bool = True,
    enhanced_blending: bool = True,
) -> list[np.ndarray]:
    """Apply face swap to a batch of target images."""
    results = []
    for tgt in target_imgs:
        swapped = swap_face(
            source_img, tgt,
            enhance_blend=enhance_blend,
            color_match=color_match,
            face_restore=face_restore,
            enhanced_blending=enhanced_blending,
        )
        results.append(swapped if swapped is not None else tgt)
    return results
