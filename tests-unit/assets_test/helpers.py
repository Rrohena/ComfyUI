"""Helper functions for assets integration tests."""
import time

import requests


def trigger_sync_seed_assets(session: requests.Session, base_url: str) -> None:
    """Force a fast sync/seed pass by calling the seed endpoint."""
    session.post(base_url + "/api/assets/seed", json={"roots": ["models", "input", "output"]}, timeout=30)
    time.sleep(0.2)


def get_asset_filename(asset_hash: str, extension: str) -> str:
    return asset_hash.removeprefix("blake3:") + extension
