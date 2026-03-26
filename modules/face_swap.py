"""Freyra Face Swap -- post-generation face replacement using InsightFace inswapper.

Pipeline: generated image + reference face -> detect faces -> swap -> blend edges.
Runs entirely on CPU to preserve GPU VRAM for diffusion.

Requires: insightface>=0.7.3, onnxruntime>=1.16.0
Model: inswapper_128.onnx (~540MB) in models/insightface/
"""

import os
import cv2
import numpy as np

import modules.config

_analyser = None
_swapper = None

INSWAPPER_PATH = os.path.join(modules.config.paths_inpaint[0] if modules.config.paths_inpaint else 'models',
                               '..', 'insightface', 'inswapper_128.onnx')
INSWAPPER_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 'models', 'insightface', 'inswapper_128.onnx')
)


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


def swap_face(
    source_img: np.ndarray,
    target_img: np.ndarray,
    enhance_blend: bool = True,
) -> np.ndarray | None:
    """Swap the face from source_img onto target_img.

    Parameters
    ----------
    source_img : np.ndarray
        Reference face image (RGB, HWC).
    target_img : np.ndarray
        Generated image to receive the face (RGB, HWC).
    enhance_blend : bool
        Apply Gaussian feathering on face boundaries.

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

        result_bgr = swapper.get(tgt_bgr, target_face, source_face, paste_back=True)

        if enhance_blend:
            result_bgr = _smooth_face_boundary(result_bgr, tgt_bgr, target_face)

        return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    except Exception as e:
        print(f'[FaceSwap] Error: {e}')
        return None


def swap_face_batch(
    source_img: np.ndarray,
    target_imgs: list[np.ndarray],
    enhance_blend: bool = True,
) -> list[np.ndarray]:
    """Apply face swap to a batch of target images."""
    results = []
    for tgt in target_imgs:
        swapped = swap_face(source_img, tgt, enhance_blend=enhance_blend)
        results.append(swapped if swapped is not None else tgt)
    return results
