import numpy as np
from typing import Optional

class LatentInterpolator:
    """Helper class to create smooth interpolations in latent space."""

    def __init__(self, z_dim: int, n_steps: int = 60):
        self.z_dim = z_dim
        self.n_steps = n_steps

    def interpolate(self, z1: np.ndarray, z2: np.ndarray) -> np.ndarray:
        """Linearly interpolate between two latent vectors."""
        ratios = np.linspace(0, 1, num=self.n_steps, dtype=np.float32)
        # Broadcasting the operation is more efficient than a list comprehension
        return (1.0 - ratios[:, np.newaxis]) * z1 + ratios[:, np.newaxis] * z2

    def random_walk(self, num_segments: int = 1, start: Optional[np.ndarray] = None, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate a random walk through latent space composed of multiple segments.

        Args:
            num_segments: Number of random interpolation segments to chain together.
            start: Optional starting latent vector. If None, a random vector is used.
            seed: Optional RNG seed for reproducibility.

        Returns:
            np.ndarray: Array of latent vectors forming the random walk with shape
                [num_segments * (n_steps - 1) + 1, z_dim].
        """
        rng = np.random.default_rng(seed)
        current = rng.standard_normal(self.z_dim, dtype=np.float32) if start is None else np.asarray(start, dtype=np.float32)
        
        walk_segments = []
        for i in range(num_segments):
            next_vec = rng.standard_normal(self.z_dim, dtype=np.float32)
            segment = self.interpolate(current, next_vec)
            
            # To avoid duplicating the connection points, we skip the first element 
            # of each new segment after the first one.
            if i > 0:
                walk_segments.append(segment[1:])
            else:
                walk_segments.append(segment)
                
            current = next_vec
            
        return np.vstack(walk_segments)
