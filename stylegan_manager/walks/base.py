from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List
import json
import os


@dataclass
class Walk(ABC):
    """Abstract base class for latent-space walks.

    Subclasses should implement :meth:`generate` to populate ``frames`` with
    image objects (e.g. ``PIL.Image`` instances).
    """

    model: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    frames: List[Any] = field(default_factory=list)

    @abstractmethod
    def generate(self) -> None:
        """Populate :attr:`frames` with generated images."""

    def save(self, path: str) -> None:
        """Persist walk metadata to ``path`` as JSON."""
        data = {"metadata": self.metadata}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, model: Any, path: str) -> "Walk":
        """Load walk metadata from ``path`` and return a new instance."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        walk = cls(model=model)
        walk.metadata = data.get("metadata", {})
        return walk

    def export_frames(self, directory: str) -> None:
        """Export generated frames to ``directory`` as PNG files."""
        os.makedirs(directory, exist_ok=True)
        for i, frame in enumerate(self.frames):
            if hasattr(frame, "save"):
                frame.save(os.path.join(directory, f"{i:04d}.png"))
