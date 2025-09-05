from __future__ import annotations

"""Utilities for tracking video rendering state."""

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional
import threading

from stylegan_manager.walks.custom_walk import CustomWalk


@dataclass
class VideoEntry:
    """Metadata for a tracked walk."""

    walk: Optional[CustomWalk] = None
    status: str = "queued"  # queued, rendering, rendered


class VideoManager:
    """Track walks and their render state."""

    def __init__(self) -> None:
        self.videos: Dict[int, VideoEntry] = {}
        self.pending: Deque[int] = deque()
        self.rendering: Optional[int] = None
        self.lock = threading.Lock()
        self.cancelled = set()

    # ------------------------------------------------------------------
    # Queue and status management
    # ------------------------------------------------------------------
    def mark_queued(self, walk_id: int) -> None:
        """Mark a walk as queued for rendering."""
        with self.lock:
            self.pending.append(walk_id)
            self.videos.setdefault(walk_id, VideoEntry()).status = "queued"

    def mark_rendering(self, walk_id: int) -> None:
        """Mark a walk as currently being rendered."""
        with self.lock:
            self.rendering = walk_id
            try:
                self.pending.remove(walk_id)
            except ValueError:
                pass
            self.videos.setdefault(walk_id, VideoEntry()).status = "rendering"
            self.cancelled.discard(walk_id)

    def mark_rendered(self, walk_id: int) -> None:
        """Mark a walk as fully rendered."""
        with self.lock:
            if self.rendering == walk_id:
                self.rendering = None
            self.videos.setdefault(walk_id, VideoEntry()).status = "rendered"

    def remove_from_queue(self, walk_id: int) -> bool:
        """Remove a walk from the pending queue and mark cancelled."""
        with self.lock:
            try:
                self.pending.remove(walk_id)
                self.cancelled.add(walk_id)
                self.videos.pop(walk_id, None)
                return True
            except ValueError:
                return False

    def is_cancelled(self, walk_id: int) -> bool:
        """Check and clear if a walk has been cancelled."""
        with self.lock:
            if walk_id in self.cancelled:
                self.cancelled.remove(walk_id)
                return True
            return False

    def current(self) -> Optional[int]:
        """Return the walk ID currently rendering, if any."""
        with self.lock:
            return self.rendering

    def queue_status(self) -> Dict[str, Optional[int]]:
        """Return the current rendering walk and pending queue."""
        with self.lock:
            return {"rendering": self.rendering, "pending": list(self.pending)}

    # ------------------------------------------------------------------
    # Legacy helpers for curated walks/UI
    # ------------------------------------------------------------------
    def add_walk(self, name: int, walk: CustomWalk) -> None:
        """Add a walk to be tracked by name."""
        with self.lock:
            self.videos[name] = VideoEntry(walk=walk, status="queued")

    def list_walks(self) -> Dict[int, Dict[str, str]]:
        """Return metadata for all tracked walks."""
        with self.lock:
            return {name: {"status": entry.status} for name, entry in self.videos.items()}

