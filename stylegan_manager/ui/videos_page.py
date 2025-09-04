from __future__ import annotations

from stylegan_manager.videos.manager import VideoManager


def render_videos_page(manager: VideoManager) -> str:
    """Return a simple HTML page listing curated walks."""
    body = ["<h1>Videos</h1>", "<ul>"]
    for name, meta in manager.list_walks().items():
        status = "rendered" if meta["rendered"] else "pending"
        body.append(f"<li>{name} - {status}</li>")
    body.append("</ul>")
    return "\n".join(body)
