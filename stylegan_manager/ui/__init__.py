from __future__ import annotations

import base64
import io
from typing import Iterable

from PIL.Image import Image
from flask import render_template

from stylegan_manager.videos.manager import VideoManager


def render_random_walk_page(images: Iterable[Image]) -> str:
    """Render a page showing random-walk frames."""
    frames = []
    for image in images:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        frames.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))
    return render_template("random_walk.html", frames=frames)


def render_videos_page(manager: VideoManager) -> str:
    """Render a page listing curated walks."""
    walks = manager.list_walks()
    return render_template("videos.html", walks=walks)
