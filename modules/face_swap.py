"""Freyra Face Swap -- high-quality face replacement pipeline.

Pipeline: source face + target image -> detect -> swap -> parse mask ->
          color match -> soft-mask blend -> GFPGAN restore.

Each enhanced step is individually protected: if it fails, the pipeline
continues with the result from the previous step instead of returning None.

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
# Robust face detection with fallbacks for close-ups
# ---------------------------------------------------------------------------

def _detect_face_robust(analyser, image_bgr: np.ndarray) -> list:
    """Detect faces with fallbacks for close-up images.

    When the standard detection fails (face fills the entire frame),
    adds padding around the image and retries at multiple scales.
    """
    faces = analyser.get(image_bgr)
    if faces:
        return faces

    h, w = image_bgr.shape[:2]
    print(f'[FaceSwap] No face at default scale ({w}x{h}), trying padded detection...')

    # Strategy 1: pad the image so the face is ~50% of the canvas
    for pad_ratio in (0.5, 1.0, 1.5):
        pad_x = int(w * pad_ratio)
        pad_y = int(h * pad_ratio)
        padded = cv2.copyMakeBorder(
            image_bgr, pad_y, pad_y, pad_x, pad_x,
            cv2.BORDER_CONSTANT, value=(128, 128, 128)
        )
        faces = analyser.get(padded)
        if faces:
            print(f'[FaceSwap] Found {len(faces)} face(s) with {pad_ratio}x padding')
            # Shift bboxes and landmarks back to the original coordinate space
            for face in faces:
                face.bbox[0] -= pad_x
                face.bbox[1] -= pad_y
                face.bbox[2] -= pad_x
                face.bbox[3] -= pad_y
                if hasattr(face, 'kps') and face.kps is not None:
                    face.kps[:, 0] -= pad_x
                    face.kps[:, 1] -= pad_y
                if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
                    face.landmark_2d_106[:, 0] -= pad_x
                    face.landmark_2d_106[:, 1] -= pad_y
                if hasattr(face, 'landmark_3d_68') and face.landmark_3d_68 is not None:
                    face.landmark_3d_68[:, 0] -= pad_x
                    face.landmark_3d_68[:, 1] -= pad_y
            return faces

    # Strategy 2: downscale the image so the face is smaller relative to det_size
    for scale in (0.5, 0.35, 0.25):
        small = cv2.resize(image_bgr, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
        faces = analyser.get(small)
        if faces:
            print(f'[FaceSwap] Found {len(faces)} face(s) at {scale}x scale')
            inv = 1.0 / scale
            for face in faces:
                face.bbox *= inv
                if hasattr(face, 'kps') and face.kps is not None:
                    face.kps *= inv
                if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
                    face.landmark_2d_106 *= inv
                if hasattr(face, 'landmark_3d_68') and face.landmark_3d_68 is not None:
                    face.landmark_3d_68 *= inv
            return faces

    print('[FaceSwap] All detection strategies exhausted, no face found')
    return []


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
                         expand_ratio: float = 0.3) -> np.ndarray | None:
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

        # Feather the EDGES only: dilate first to expand the mask slightly,
        # then blur. This keeps the face core at 1.0 while creating a
        # gradual falloff at the boundary.
        edge_k = max(3, int(min(x2 - x1, y2 - y1) * 0.04) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_k, edge_k))
        face_mask_crop = cv2.dilate(face_mask_crop, kernel, iterations=1)
        blur_size = max(3, int(min(x2 - x1, y2 - y1) * 0.08) | 1)
        face_mask_crop = cv2.GaussianBlur(face_mask_crop, (blur_size, blur_size), 0)
        face_mask_crop = np.clip(face_mask_crop, 0, 1.0)

        full_mask = np.zeros((h, w), dtype=np.float32)
        full_mask[y1:y2, x1:x2] = face_mask_crop

        return full_mask

    except Exception as e:
        print(f'[FaceSwap] Face parsing failed: {e}')
        return None


def _make_bbox_mask(h: int, w: int, face_bbox: np.ndarray) -> np.ndarray:
    """Create a soft elliptical mask from a face bounding box."""
    bbox = face_bbox.astype(int)
    x1 = max(0, int(bbox[0]))
    y1 = max(0, int(bbox[1]))
    x2 = min(w, int(bbox[2]))
    y2 = min(h, int(bbox[3]))

    mask = np.zeros((h, w), dtype=np.float32)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    ax = max(1, (x2 - x1) // 2)
    ay = max(1, (y2 - y1) // 2)
    cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 1.0, -1)

    ksize = max(5, int(min(ax, ay) * 0.3) | 1)
    mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)
    return mask


# ---------------------------------------------------------------------------
# Color matching
# ---------------------------------------------------------------------------

def _match_face_color(swapped_bgr: np.ndarray, original_bgr: np.ndarray,
                      mask: np.ndarray) -> np.ndarray:
    """Match only the LIGHTING of the swapped face to the target environment.

    Works in LAB color space: transfers only the L (luminance) channel
    to match the scene lighting, while preserving A and B channels which
    carry the swapped face's skin tone identity.
    """
    if mask is None:
        return swapped_bgr

    mask_bool = mask > 0.5
    if not np.any(mask_bool):
        return swapped_bgr

    swapped_lab = cv2.cvtColor(swapped_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    original_lab = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Only transfer L channel (lighting), leave A/B (color/identity) alone
    src_L = swapped_lab[:, :, 0][mask_bool]
    tgt_L = original_lab[:, :, 0][mask_bool]

    src_mean, src_std = src_L.mean(), max(src_L.std(), 1e-6)
    tgt_mean, tgt_std = tgt_L.mean(), max(tgt_L.std(), 1e-6)

    normalized = (swapped_lab[:, :, 0] - src_mean) / src_std
    transferred = normalized * tgt_std + tgt_mean

    blend_factor = 0.4
    blended_L = swapped_lab[:, :, 0] * (1 - blend_factor) + transferred * blend_factor
    swapped_lab[:, :, 0] = np.where(mask_bool, blended_L, swapped_lab[:, :, 0])

    result = cv2.cvtColor(np.clip(swapped_lab, 0, 255).astype(np.uint8),
                           cv2.COLOR_LAB2BGR)
    return result


# ---------------------------------------------------------------------------
# Soft-mask alpha blending (primary method -- no Poisson artifacts)
# ---------------------------------------------------------------------------

def _soft_mask_blend(swapped_bgr: np.ndarray, original_bgr: np.ndarray,
                     mask: np.ndarray) -> np.ndarray:
    """Blend the swapped face into the original using a soft alpha mask.

    The mask already has feathered edges from parsing/bbox generation.
    We only do a very light blur to prevent jagged edges, but keep the
    face core at full opacity so the swap is clearly visible.
    """
    # Very light single-pass blur just to anti-alias mask edges
    ksize = max(3, int(min(original_bgr.shape[:2]) * 0.008) | 1)
    soft = cv2.GaussianBlur(mask, (ksize, ksize), 0)

    peak = soft.max()
    if peak > 0:
        print(f'[FaceSwap] Blend mask: peak={peak:.3f}, '
              f'mean_face={soft[soft > 0.1].mean():.3f}, '
              f'coverage={np.sum(soft > 0.5)} px')

    mask_3ch = soft[:, :, np.newaxis]
    result = (swapped_bgr.astype(np.float64) * mask_3ch +
              original_bgr.astype(np.float64) * (1.0 - mask_3ch))
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

        # GFPGAN checkpoints use pickle-serialized objects; weights_only must be False
        state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
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

    Crops the face, runs GFPGAN at 512x512, and pastes back with soft blending.
    Returns the original image unchanged if restoration fails.
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

    blend_mask = np.zeros((crop_h, crop_w), dtype=np.float32)
    center = (crop_w // 2, crop_h // 2)
    axes = (max(1, int(crop_w * 0.40)), max(1, int(crop_h * 0.44)))
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

    The pipeline is:
      1. Detect faces in source and target
      2. Multi-pass inswapper (2 passes to reinforce identity)
      3. Optional luminance matching (LAB L-channel only)
      4. Optional GFPGAN face restoration

    The inswapper already handles its own blending via an internal mask
    built from the warped face template + 106-point landmarks + pixel
    difference thresholding. We do NOT apply a second blend pass on top
    because that erases the swap from the outer face region.

    Parameters
    ----------
    source_img : np.ndarray
        Reference face image (RGB, HWC).
    target_img : np.ndarray
        Image to receive the face (RGB, HWC).
    enhance_blend : bool
        Legacy parameter kept for backward compatibility.
    color_match : bool
        Match swapped face lighting to the target scene.
    face_restore : bool
        Apply GFPGAN face restoration after swapping.
    enhanced_blending : bool
        When True, enables multi-pass swap for stronger identity.

    Returns
    -------
    np.ndarray or None
        Modified target with swapped face, or None only if face detection
        or the core swap itself fails.
    """
    if not is_available():
        print('[FaceSwap] Not available: missing insightface/onnxruntime or inswapper model')
        return None

    # -- Step 0: load models --
    try:
        analyser = get_face_analyser()
        swapper = get_face_swapper()
    except Exception as e:
        print(f'[FaceSwap] Failed to load models: {e}')
        return None

    try:
        src_bgr = cv2.cvtColor(source_img, cv2.COLOR_RGB2BGR)
        tgt_bgr = cv2.cvtColor(target_img, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f'[FaceSwap] Color conversion failed: {e}')
        return None

    # Keep a pristine copy -- swapper returns a new array but we need
    # the original for color matching reference.
    original_bgr = tgt_bgr.copy()
    h, w = tgt_bgr.shape[:2]

    # -- Step 1: detect faces (with robust fallbacks for close-ups) --
    try:
        src_faces = _detect_face_robust(analyser, src_bgr)
        tgt_faces = _detect_face_robust(analyser, tgt_bgr)
    except Exception as e:
        print(f'[FaceSwap] Face detection failed: {e}')
        return None

    source_face = _get_largest_face(src_faces)
    target_face = _get_largest_face(tgt_faces)

    if source_face is None:
        print(f'[FaceSwap] No face detected in source image '
              f'({source_img.shape[1]}x{source_img.shape[0]})')
        return None
    if target_face is None:
        print(f'[FaceSwap] No face detected in target image '
              f'({target_img.shape[1]}x{target_img.shape[0]})')
        return None

    # Log face info and embedding health
    src_emb = getattr(source_face, 'normed_embedding', None)
    emb_status = 'MISSING'
    if src_emb is not None:
        emb_status = f'OK  norm={np.linalg.norm(src_emb):.3f}'
    print(f'[FaceSwap] Source face bbox: {source_face.bbox.astype(int).tolist()}'
          f'  embedding: {emb_status}')
    has_106 = hasattr(target_face, 'landmark_2d_106') and target_face.landmark_2d_106 is not None
    print(f'[FaceSwap] Target face bbox: {target_face.bbox.astype(int).tolist()}'
          f'  has_106lm: {has_106}')

    # -- Step 2: multi-pass inswapper for stronger identity transfer --
    num_passes = 2 if enhanced_blending else 1
    result_bgr = tgt_bgr.copy()

    for pass_idx in range(num_passes):
        try:
            if pass_idx == 0:
                current_target = target_face
            else:
                # Re-detect face in the intermediate result so the
                # inswapper aligns to the already-partially-swapped face.
                redet = analyser.get(result_bgr)
                current_target = _get_largest_face(redet) if redet else target_face

            result_bgr = swapper.get(
                result_bgr, current_target, source_face, paste_back=True,
            )
            # Measure how much this pass changed
            diff = np.abs(result_bgr.astype(float) - original_bgr.astype(float))
            bbox = target_face.bbox.astype(int)
            bx1, by1 = max(0, bbox[0]), max(0, bbox[1])
            bx2, by2 = min(w, bbox[2]), min(h, bbox[3])
            face_diff = diff[by1:by2, bx1:bx2]
            print(f'[FaceSwap] Pass {pass_idx + 1}/{num_passes} - '
                  f'face delta vs original: mean={face_diff.mean():.1f}, '
                  f'max={face_diff.max():.1f}, '
                  f'changed_px={np.sum(face_diff.max(axis=2) > 10)}')
        except Exception as e:
            print(f'[FaceSwap] Pass {pass_idx + 1} failed: {e}')
            if pass_idx == 0:
                return None

    # -- Step 3: luminance matching (optional) --
    # Only match scene LIGHTING, not color/identity. Uses a generous
    # bbox mask since this only affects the L channel.
    if color_match:
        try:
            luma_mask = _make_bbox_mask(h, w, target_face.bbox)
            result_bgr = _match_face_color(result_bgr, original_bgr, luma_mask)
            print('[FaceSwap] Luminance matching applied')
        except Exception as e:
            print(f'[FaceSwap] Luminance matching failed (continuing): {e}')

    # -- Step 4: GFPGAN face restoration --
    if face_restore:
        try:
            result_bgr = _restore_face_gfpgan(result_bgr, target_face.bbox)
            print('[FaceSwap] GFPGAN restoration applied')
        except Exception as e:
            print(f'[FaceSwap] GFPGAN restoration failed (continuing): {e}')

    print('[FaceSwap] Pipeline complete')
    return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)


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
