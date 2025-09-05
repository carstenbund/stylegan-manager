from __future__ import annotations

import numpy as np
from PIL import Image
import imageio.v2 as imageio

from .base import Walk


def interpolate_vectors(points, steps: int) -> np.ndarray:
    """Return interpolated latent vectors for a sequence of ``points``.

    Parameters
    ----------
    points: sequence of array-like
        Latent vectors defining the keyframes of the walk.
    steps: int
        Number of interpolation samples per leg (excluding end point).

    Returns
    -------
    np.ndarray
        Array of shape ``(num_vectors, latent_dim)`` containing the
        interpolated latent vectors. If fewer than two points are provided,
        the input points are returned unchanged.
    """

    pts = [np.asarray(p, dtype=np.float32) for p in points]
    if not pts:
        return np.empty((0, 0), dtype=np.float32)
    if len(pts) == 1:
        return np.asarray(pts, dtype=np.float32)

    vectors = []
    for start, end in zip(pts[:-1], pts[1:]):
        for t in np.linspace(0, 1, steps, endpoint=False):
            vectors.append((1 - t) * start + t * end)
    vectors.append(pts[-1])
    return np.stack(vectors, axis=0).astype(np.float32)


class CustomWalk(Walk):
    """Curated walk supporting multi-point interpolation and video export."""

    def __init__(self, model, points, steps: int = 10, **kwargs):
        super().__init__(model, **kwargs)
        self.points = [np.asarray(p, dtype=np.float32) for p in points]
        self.steps = steps
        self.vectors = interpolate_vectors(self.points, self.steps)

    def generate(self) -> None:
        """Generate images for the walk's precomputed latent ``vectors``."""
        self.frames = []
        for z in self.vectors:
            if hasattr(self.model, "generate_image"):
                img = self.model.generate_image(z[None, :])
            else:
                array = np.uint8(np.random.rand(256, 256, 3) * 255)
                img = Image.fromarray(array)
            self.frames.append(img)

    def to_video(self, path: str, fps: int = 24) -> None:
        """Render generated frames to an MP4 file."""
        if not self.frames:
            raise RuntimeError("generate() must be called before to_video().")
        imageio.mimwrite(path, [np.asarray(f) for f in self.frames], fps=fps)
