"""Freyra Instagram Fetcher -- download images from public Instagram posts.

Takes an Instagram post/reel URL, extracts the shortcode, downloads the image
via instaloader, and returns it as a numpy RGB array for the face swap pipeline.

Also supports fetching any direct image URL as a fallback.
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

_IMAGE_URL_PATTERN = re.compile(
    r'https?://.*\.(?:jpg|jpeg|png|webp)(?:\?.*)?$', re.IGNORECASE
)


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


def is_direct_image_url(url: str) -> bool:
    """Check whether a string looks like a direct image URL."""
    return bool(_IMAGE_URL_PATTERN.match(url.strip()))


def fetch_image_from_url(url: str) -> tuple[np.ndarray | None, str]:
    """Fetch an image from any URL (Instagram or direct image link).

    Tries Instagram-specific fetching first if the URL matches,
    then falls back to direct HTTP download.

    Parameters
    ----------
    url : str
        Instagram post URL or direct image URL.

    Returns
    -------
    tuple of (np.ndarray | None, str)
        (image_rgb, status_message). image is None on failure.
    """
    url = url.strip()
    if not url:
        return None, 'Please enter a URL.'

    if is_instagram_url(url):
        img, msg = _fetch_instagram_image(url)
        if img is not None:
            return img, msg
        # Instagram fetch failed -- append tip about direct URL
        return None, (
            f'{msg}\n\n'
            'Tip: Instagram often blocks automated access from cloud servers. '
            'Try one of these alternatives:\n'
            '1. Right-click the image on Instagram > "Copy image address" and paste that direct URL here\n'
            '2. Save the image to your device and use the "Upload" option instead'
        )

    if is_direct_image_url(url) or url.startswith('http'):
        return _fetch_direct_image(url)

    return None, (
        'Not a recognized URL. Supported formats:\n'
        '- Instagram post: https://www.instagram.com/p/XXXXX/\n'
        '- Instagram reel: https://www.instagram.com/reel/XXXXX/\n'
        '- Direct image URL: https://example.com/photo.jpg'
    )


def _fetch_instagram_image(url: str) -> tuple[np.ndarray | None, str]:
    """Download the first image from a public Instagram post via instaloader."""
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
    except Exception as e:
        err = str(e)
        if '401' in err or 'Unauthorized' in err or 'Please wait' in err:
            return None, (
                f'Instagram blocked the request (rate limited). '
                f'Shortcode: {shortcode}'
            )
        if 'not found' in err.lower() or '404' in err:
            return None, f'Post not found. It may be private or deleted (shortcode: {shortcode}).'
        return None, f'Failed to load post metadata: {e}'

    try:
        loader.download_post(post, target=tmp_dir)
    except Exception as e:
        err = str(e)
        if 'login' in err.lower():
            return None, 'This post requires login (private account). Only public posts are supported.'
        return None, f'Failed to download post image: {e}'

    image_path = _find_first_image(tmp_dir)
    if image_path is None:
        _cleanup_temp(tmp_dir)
        return None, 'No image found in the downloaded post (might be a video-only reel).'

    try:
        from PIL import Image
        pil_img = Image.open(image_path).convert('RGB')
        arr = np.array(pil_img)
        _cleanup_temp(tmp_dir)
        try:
            username = post.owner_username
        except Exception:
            username = 'unknown'
        return arr, f'Fetched image ({arr.shape[1]}x{arr.shape[0]}) from @{username}'
    except Exception as e:
        _cleanup_temp(tmp_dir)
        return None, f'Failed to read downloaded image: {e}'


def _fetch_direct_image(url: str) -> tuple[np.ndarray | None, str]:
    """Download an image from a direct URL using httpx or urllib."""
    try:
        import httpx
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                          '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        }
        resp = httpx.get(url, headers=headers, follow_redirects=True, timeout=30.0)
        resp.raise_for_status()
        content_type = resp.headers.get('content-type', '')
        if 'image' not in content_type and not any(
            url.lower().endswith(ext) for ext in ('.jpg', '.jpeg', '.png', '.webp')
        ):
            return None, f'URL does not appear to be an image (content-type: {content_type})'
        image_bytes = resp.content
    except ImportError:
        try:
            from urllib.request import Request, urlopen
            req = Request(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0'
            })
            with urlopen(req, timeout=30) as resp:
                image_bytes = resp.read()
        except Exception as e:
            return None, f'Failed to download image: {e}'
    except Exception as e:
        return None, f'Failed to download image: {e}'

    try:
        from PIL import Image
        from io import BytesIO
        pil_img = Image.open(BytesIO(image_bytes)).convert('RGB')
        arr = np.array(pil_img)
        return arr, f'Fetched image ({arr.shape[1]}x{arr.shape[0]}) from URL'
    except Exception as e:
        return None, f'Downloaded data is not a valid image: {e}'


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
