"""Resolve lossless stage outputs while retaining compatibility with old examples."""

from pathlib import Path


def find_image(folder, stem):
    for suffix in (".png", ".jpg", ".jpeg"):
        path = Path(folder) / (stem + suffix)
        if path.is_file():
            return path
    raise ValueError(f"Missing {stem}.png/.jpg/.jpeg in {folder}")
