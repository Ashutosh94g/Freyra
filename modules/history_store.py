"""Freyra Server-Side Generation History

Persists generation history to a JSON file in the outputs directory,
so it survives across sessions regardless of browser URL or subdomain.
Capped at 200 entries.
"""

import os
import json
import time
import base64
import io
import threading

from PIL import Image

import modules.config

MAX_ENTRIES = 200
THUMB_SIZE = 128
THUMB_QUALITY = 60
_lock = threading.Lock()


def _history_path() -> str:
    return os.path.join(os.path.abspath(modules.config.path_outputs), 'freyra_history.json')


def _load_unlocked() -> list[dict]:
    """Read history file without locking (caller must hold _lock)."""
    path = _history_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def load_history() -> list[dict]:
    with _lock:
        return _load_unlocked()


def _save_unlocked(history: list[dict]):
    """Write history file without locking (caller must hold _lock)."""
    path = _history_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False)


def _make_thumbnail(image_path: str) -> str | None:
    """Create a base64 JPEG thumbnail from an image file on disk."""
    try:
        img = Image.open(image_path)
        img.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
        buf = io.BytesIO()
        img.convert('RGB').save(buf, format='JPEG', quality=THUMB_QUALITY)
        return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')
    except Exception:
        return None


def add_entry(
    prompt: str,
    seed: str | int,
    image_paths: list[str],
) -> dict | None:
    """Add a generation entry to history. Returns the entry or None on failure."""
    if not image_paths:
        return None

    thumb = _make_thumbnail(image_paths[0]) if image_paths else None

    entry = {
        'id': f"{int(time.time())}_{os.urandom(3).hex()}",
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'seed': str(seed),
        'prompt': (prompt or '')[:500],
        'thumbnail': thumb,
        'imageCount': len(image_paths),
    }

    with _lock:
        history = _load_unlocked()
        history.insert(0, entry)
        if len(history) > MAX_ENTRIES:
            history = history[:MAX_ENTRIES]
        _save_unlocked(history)

    return entry


def clear_history():
    with _lock:
        _save_unlocked([])


def export_history_json() -> str:
    return json.dumps(load_history(), indent=2, ensure_ascii=False)


def import_history_json(raw_json: str) -> int:
    """Import entries from a JSON string. Returns count of new entries added."""
    try:
        imported = json.loads(raw_json)
        if not isinstance(imported, list):
            return 0
    except Exception:
        return 0

    with _lock:
        history = _load_unlocked()
        existing_ids = {e.get('id') for e in history}
        added = 0
        for entry in imported:
            if isinstance(entry, dict) and entry.get('id') not in existing_ids:
                history.append(entry)
                existing_ids.add(entry.get('id'))
                added += 1
        history.sort(key=lambda e: e.get('timestamp', ''), reverse=True)
        if len(history) > MAX_ENTRIES:
            history = history[:MAX_ENTRIES]
        _save_unlocked(history)
    return added


def render_history_html() -> str:
    """Render the history panel HTML from server-side data."""
    history = load_history()

    html = '<div class="freyra-history-controls">'
    html += f'<span style="color:#888;font-size:13px;">{len(history)} generations saved (server-side)</span>'
    html += '<div style="display:flex;gap:8px;">'
    html += '</div></div>'

    if not history:
        html += (
            '<div style="text-align:center;padding:40px;color:#666;">'
            'No generation history yet. Generate some images to see them here.'
            '</div>'
        )
        return html

    html += '<div class="freyra-history-grid">'
    for entry in history:
        ts = entry.get('timestamp', '')
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(ts)
            date_str = dt.strftime('%b %d, %Y %H:%M')
        except Exception:
            date_str = ts

        seed_str = entry.get('seed', '?')
        prompt_raw = entry.get('prompt', '')
        prompt_short = prompt_raw[:80] + ('...' if len(prompt_raw) > 80 else '')
        prompt_escaped = prompt_raw.replace('"', '&quot;').replace('<', '&lt;').replace('>', '&gt;')
        prompt_display = prompt_short.replace('<', '&lt;').replace('>', '&gt;')
        count = entry.get('imageCount', 1)

        html += f'<div class="freyra-history-card" title="{prompt_escaped}">'
        thumb = entry.get('thumbnail')
        if thumb:
            html += f'<img src="{thumb}" class="freyra-history-thumb">'
        else:
            html += '<div class="freyra-history-thumb freyra-history-no-thumb">No preview</div>'
        html += '<div class="freyra-history-meta">'
        html += f'<div class="freyra-history-date">{date_str}</div>'
        html += f'<div class="freyra-history-seed">Seed: {seed_str}</div>'
        html += f'<div class="freyra-history-prompt">{prompt_display}</div>'
        html += f'<div class="freyra-history-count">{count} image(s)</div>'
        html += '</div></div>'

    html += '</div>'
    return html
