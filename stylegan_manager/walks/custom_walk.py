from __future__ import annotations

import numpy as np
from PIL import Image
import imageio.v2 as imageio

from .base import Walk


class CustomWalk(Walk):
    """Curated walk supporting multi-point interpolation and video export."""

    def __init__(self, model, points, steps: int = 10, **kwargs):
        super().__init__(model, **kwargs)
        self.points = [np.asarray(p) for p in points]
        self.steps = steps

    def generate(self) -> None:
        self.frames = []
        if len(self.points) < 2:
            return
        for start, end in zip(self.points[:-1], self.points[1:]):
            for t in np.linspace(0, 1, self.steps, endpoint=False):
                z = (1 - t) * start + t * end
                if hasattr(self.model, "generate_image"):
                    img = self.model.generate_image(z[None, :])
                else:
                    array = np.uint8(np.random.rand(256, 256, 3) * 255)
                    img = Image.fromarray(array)
                self.frames.append(img)
        # add last point
        if hasattr(self.model, "generate_image"):
            img = self.model.generate_image(self.points[-1][None, :])
        else:
            array = np.uint8(np.random.rand(256, 256, 3) * 255)
            img = Image.fromarray(array)
        self.frames.append(img)

    def to_video(self, path: str, fps: int = 24) -> None:
        """Render generated frames to an MP4 file."""
        if not self.frames:
            raise RuntimeError("generate() must be called before to_video().")
        imageio.mimwrite(path, [np.asarray(f) for f in self.frames], fps=fps)
