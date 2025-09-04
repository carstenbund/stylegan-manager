from __future__ import annotations

from typing import Iterable

from PIL.Image import Image


def render_random_walk_page(images: Iterable[Image]) -> str:
    """Return a simple HTML page displaying ``images``."""
    body = ["<h1>Random Walk</h1>"]
    for i, _ in enumerate(images):
        body.append(f"<p>Frame {i}</p>")
    return "\n".join(body)
