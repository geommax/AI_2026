"""Runtime environment helpers for local execution."""

import os
from pathlib import Path


def _clear_invalid_path_env(var_name: str) -> None:
    value = os.environ.get(var_name)
    if not value:
        return

    normalized = value.strip().strip('"')
    if not normalized:
        os.environ.pop(var_name, None)
        return

    if not Path(normalized).expanduser().exists():
        os.environ.pop(var_name, None)


def sanitize_ssl_env() -> None:
    """Remove invalid SSL certificate env vars so httpx/gradio can import."""
    _clear_invalid_path_env("SSL_CERT_FILE")
    _clear_invalid_path_env("SSL_CERT_DIR")