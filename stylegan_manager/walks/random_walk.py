from __future__ import annotations

import numpy as np
from PIL import Image

from .base import Walk


class RandomWalk(Walk):
    """Simple latent-space exploration that produces a few random frames."""

    def __init__(self, model, steps: int = 3, **kwargs):
        super().__init__(model, **kwargs)
        self.steps = steps

    def generate(self) -> None:
        self.frames = []
        latent_dim = getattr(self.model, "latent_dim", 512)
        for _ in range(self.steps):
            z = np.random.randn(1, latent_dim)
            if hasattr(self.model, "generate_image"):
                img = self.model.generate_image(z)
            else:
                # Fallback to random noise image when no model is supplied.
                array = np.uint8(np.random.rand(256, 256, 3) * 255)
                img = Image.fromarray(array)
            self.frames.append(img)
