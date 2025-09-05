from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image

from .base import Walk
from ..utils import sample_latents


class RandomWalk(Walk):
    """Simple latent-space exploration that produces a few random frames."""

    def __init__(self, model, steps: int = 3, **kwargs):
        super().__init__(model, **kwargs)
        self.steps = steps

    def generate(
        self,
        n_vectors: Optional[int] = None,
        extent: float = 2.0,
        seed: Optional[int] = None,
    ) -> None:
        self.frames = []
        latent_dim = getattr(self.model, "latent_dim", 512)
        count = n_vectors if n_vectors is not None else self.steps
        latents = sample_latents(latent_dim, count, extent=extent, seed=seed)
        for z in latents:
            z = z[np.newaxis, :]
            if hasattr(self.model, "generate_image"):
                img = self.model.generate_image(z)
            else:
                # Fallback to random noise image when no model is supplied.
                array = np.uint8(np.random.rand(256, 256, 3) * 255)
                img = Image.fromarray(array)
            self.frames.append(img)
