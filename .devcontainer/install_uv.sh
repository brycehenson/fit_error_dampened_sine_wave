#!/usr/bin/env bash
set -euo pipefail

UV_VERSION="0.10.2"

if ! command -v pipx &> /dev/null; then
    echo "pipx not found, installing uv with pip..."
    python3 -m pip install --no-cache-dir "uv==${UV_VERSION}"
else
    pipx ensurepath
    # reinstall to ensure correct version is present
    if pipx list | grep -q " uv "; then
        pipx uninstall uv || true
    fi
    pipx install --global "uv==${UV_VERSION}"
fi

if command -v uv &> /dev/null; then
    echo "uv installed:" && uv --version || true
fi
