"""Freyra Campaign Mode -- batch generation with character consistency.

A campaign is a series of images sharing the same character identity
but varying pose, outfit, background, etc.
"""

import os
import json
import time
import uuid

CAMPAIGNS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'outputs', 'campaigns'
)


def _ensure_dir():
    os.makedirs(CAMPAIGNS_DIR, exist_ok=True)


def create_campaign(
    name: str,
    character_name: str,
    shoot_type: str,
    variations: list[dict],
) -> str:
    """Create a campaign definition.

    Args:
        name: Campaign display name
        character_name: Character profile to use
        shoot_type: Shoot type label
        variations: List of dicts, each with dimension overrides per image

    Returns:
        Campaign ID
    """
    _ensure_dir()
    campaign_id = f"{name.lower().replace(' ', '_')}_{uuid.uuid4().hex[:6]}"
    campaign_dir = os.path.join(CAMPAIGNS_DIR, campaign_id)
    os.makedirs(campaign_dir, exist_ok=True)

    meta = {
        'name': name,
        'character': character_name,
        'shoot_type': shoot_type,
        'variations': variations,
        'created_at': time.time(),
        'status': 'pending',
        'results': [],
    }

    with open(os.path.join(campaign_dir, 'campaign.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    return campaign_id


def list_campaigns() -> list[dict]:
    """List all campaigns."""
    _ensure_dir()
    campaigns = []
    for name in sorted(os.listdir(CAMPAIGNS_DIR)):
        meta_path = os.path.join(CAMPAIGNS_DIR, name, 'campaign.json')
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                meta['id'] = name
                meta['path'] = os.path.join(CAMPAIGNS_DIR, name)
                campaigns.append(meta)
            except Exception:
                continue
    return campaigns


def get_campaign(campaign_id: str) -> dict | None:
    """Load a campaign by ID."""
    meta_path = os.path.join(CAMPAIGNS_DIR, campaign_id, 'campaign.json')
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    meta['id'] = campaign_id
    return meta


def update_campaign_status(campaign_id: str, status: str, results: list | None = None):
    """Update campaign status and results."""
    meta_path = os.path.join(CAMPAIGNS_DIR, campaign_id, 'campaign.json')
    if not os.path.exists(meta_path):
        return
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    meta['status'] = status
    if results is not None:
        meta['results'] = results
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
