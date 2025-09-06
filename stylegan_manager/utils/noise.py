import numpy as np
from PIL import Image, ImageFilter


def compute_noise_score(img: Image.Image) -> float:
    """Compute a simple noise metric for an image.

    The metric is the variance of a high-pass filtered version of the image.
    A higher value indicates more high-frequency content (noise).
    """
    gray = img.convert("L")
    blurred = gray.filter(ImageFilter.GaussianBlur(radius=1))
    high_pass = np.asarray(gray, dtype=np.float32) - np.asarray(blurred, dtype=np.float32)
    return float(np.var(high_pass))
