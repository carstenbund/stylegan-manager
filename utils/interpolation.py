import numpy as np

class LatentInterpolator:
    """Helper class to create smooth interpolations in latent space."""

    def __init__(self, z_dim: int, n_steps: int = 60):
        self.z_dim = z_dim
        self.n_steps = n_steps

    def interpolate(self, z1: np.ndarray, z2: np.ndarray) -> np.ndarray:
        """Linearly interpolate between two latent vectors."""
        ratios = np.linspace(0, 1, num=self.n_steps, dtype=np.float32)
        return np.array([(1.0 - r) * z1 + r * z2 for r in ratios], dtype=np.float32)

    def random_walk(self, num_segments: int = 1, start: np.ndarray | None = None, seed: int | None = None) -> np.ndarray:
        """
        Generate a random walk through latent space composed of multiple segments.

        Args:
            num_segments: Number of random interpolation segments to chain together.
            start: Optional starting latent vector. If None, a random vector is used.
            seed: Optional RNG seed for reproducibility.

        Returns:
            np.ndarray: Array of latent vectors forming the random walk with shape
                [num_segments * n_steps, z_dim].
        """
        rng = np.random.default_rng(seed)
        current = rng.standard_normal(self.z_dim) if start is None else np.asarray(start, dtype=np.float32)
        walk = []
        for _ in range(num_segments):
            next_vec = rng.standard_normal(self.z_dim)
            segment = self.interpolate(current, next_vec)
            if walk:
                segment = segment[1:]  # Avoid duplicating boundary between segments
            walk.append(segment)
            current = next_vec
        return np.vstack(walk)
