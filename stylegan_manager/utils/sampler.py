from __future__ import annotations

from typing import Optional
import numpy as np


def sample_latents(
    z_dim: int,
    n_vectors: int,
    extent: float = 2.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate latent vectors bounded by ``[-extent, extent]``.

    Vectors are drawn from a standard normal distribution using
    :func:`numpy.random.default_rng`.  If SciPy's quasi-Monte Carlo tools are
    available, a Sobol sequence is mixed in to provide better coverage of the
    latent space.
    """
    rng = np.random.default_rng(seed)
    latents = rng.standard_normal((n_vectors, z_dim))
    latents = np.clip(latents, -extent, extent)

    # Optionally blend with a Sobol low-discrepancy sequence for coverage.
    try:  # pragma: no cover - SciPy may not be installed
        from scipy.stats import qmc

        sobol = qmc.Sobol(d=z_dim, scramble=True, seed=seed)
        quasi = sobol.random(n_vectors)
        quasi = qmc.scale(quasi, l_bounds=-extent, u_bounds=extent)
        latents = (latents + quasi) / 2.0
    except Exception:
        pass

    return latents.astype(np.float32)
