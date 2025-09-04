from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from stylegan_manager.walks.custom_walk import CustomWalk


@dataclass
class VideoEntry:
    walk: CustomWalk
    rendered: bool = False


class VideoManager:
    """Track curated walks and their render state."""

    def __init__(self) -> None:
        self.videos: Dict[str, VideoEntry] = {}

    def add_walk(self, name: str, walk: CustomWalk) -> None:
        self.videos[name] = VideoEntry(walk=walk)

    def mark_rendered(self, name: str) -> None:
        if name in self.videos:
            self.videos[name].rendered = True

    def list_walks(self) -> Dict[str, Dict[str, bool]]:
        return {
            name: {"rendered": entry.rendered}
            for name, entry in self.videos.items()
        }
