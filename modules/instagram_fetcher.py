"""Freyra Instagram Fetcher -- download images from public Instagram posts.

Takes an Instagram post/reel URL, extracts the shortcode, downloads the image
via instaloader, and returns it as a numpy RGB array for the face swap pipeline.
"""

import os
import re
import tempfile
import numpy as np

_SHORTCODE_PATTERNS = [
    re.compile(r'instagram\.com/p/([A-Za-z0-9_-]+)'),
    re.compile(r'instagram\.com/reel/([A-Za-z0-9_-]+)'),
    re.compile(r'instagram\.com/tv/([A-Za-z0-9_-]+)'),
]


def _extract_shortcode(url: str) -> str | None:
    """Extract the post shortcode from an Instagram URL."""
    url = url.strip()
    for pattern in _SHORTCODE_PATTERNS:
        match = pattern.search(url)
        if match:
            return match.group(1)
    return None


def is_instagram_url(url: str) -> bool:
    """Check whether a string looks like an Instagram post/reel URL."""
    return _extract_shortcode(url) is not None


def fetch_instagram_image(url: str) -> tuple[np.ndarray | None, str]:
    """Download the first image from a public Instagram post.

    Parameters
    ----------
    url : str
        Full Instagram post or reel URL.

    Returns
    -------
    tuple of (np.ndarray | None, str)
        (image_rgb, status_message). image is None on failure.
    """
    shortcode = _extract_shortcode(url)
    if shortcode is None:
        return None, 'Invalid Instagram URL. Expected a /p/, /reel/, or /tv/ link.'

    try:
        import instaloader
    except ImportError:
        return None, 'instaloader is not installed. Run: pip install instaloader'

    loader = instaloader.Instaloader(
        download_videos=False,
        download_video_thumbnails=False,
        download_geotags=False,
        download_comments=False,
        save_metadata=False,
        compress_json=False,
        quiet=True,
    )

    tmp_dir = tempfile.mkdtemp(prefix='freyra_ig_')

    try:
        post = instaloader.Post.from_shortcode(loader.context, shortcode)
    except instaloader.exceptions.QueryReturnedNotFoundException:
        return None, f'Post not found. It may be private or deleted (shortcode: {shortcode}).'
    except instaloader.exceptions.ConnectionException as e:
        return None, f'Connection error fetching post: {e}'
    except Exception as e:
        return None, f'Failed to load post metadata: {e}'

    try:
        loader.download_post(post, target=tmp_dir)
    except instaloader.exceptions.LoginRequiredException:
        return None, 'This post requires login (private account). Only public posts are supported.'
    except Exception as e:
        return None, f'Failed to download post image: {e}'

    image_path = _find_first_image(tmp_dir)
    if image_path is None:
        return None, 'No image found in the downloaded post (might be a video-only reel).'

    try:
        from PIL import Image
        pil_img = Image.open(image_path).convert('RGB')
        arr = np.array(pil_img)
        return arr, f'Fetched image ({arr.shape[1]}x{arr.shape[0]}) from @{post.owner_username}'
    except Exception as e:
        return None, f'Failed to read downloaded image: {e}'
    finally:
        _cleanup_temp(tmp_dir)


def _find_first_image(directory: str) -> str | None:
    """Find the first JPEG/PNG file in a directory tree."""
    image_exts = {'.jpg', '.jpeg', '.png', '.webp'}
    for root, _dirs, files in os.walk(directory):
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() in image_exts:
                return os.path.join(root, f)
    return None


def _cleanup_temp(directory: str):
    """Best-effort cleanup of temp download directory."""
    import shutil
    try:
        shutil.rmtree(directory, ignore_errors=True)
    except Exception:
        pass
