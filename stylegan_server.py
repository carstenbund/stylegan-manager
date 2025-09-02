import os
from io import BytesIO

import argparse
import numpy as np
import random
from flask import Flask, send_file, jsonify, render_template

from stylegan_gen import StyleGANGenerator


class NoiseGenerator:
    """Generate latent vectors with linear interpolation."""

    def __init__(self, ns: int = 512, steps: int = 60):
        self.noise_size = ns
        self.n_steps = steps
        self.current_step = steps  # trigger new leg on first call
        self.z_start = None
        self.z_end = None
        self.vectors = None

    def _start_new_leg(self):
        if self.z_end is not None:
            self.z_start = self.z_end
        else:
            self.z_start = np.random.randn(self.noise_size)
        self.z_end = np.random.randn(self.noise_size)
        ratios = np.linspace(0, 1, num=self.n_steps, dtype=np.float32)
        self.vectors = np.array(
            [(1.0 - r) * self.z_start + r * self.z_end for r in ratios],
            dtype=np.float32,
        )
        self.current_step = 0

    def __next__(self):
        if self.current_step >= self.n_steps:
            self._start_new_leg()
        z = self.vectors[self.current_step]
        self.current_step += 1
        return z


# ----------------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------------

DEFAULT_NETWORK_PKL = (
    "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl"
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--network",
    "--network-pkl",
    dest="network_pkl",
    type=str,
    default=os.environ.get("NETWORK_PKL", DEFAULT_NETWORK_PKL),
    help="Network pickle to load.",
)
args, _ = parser.parse_known_args()

NETWORK_PKL = args.network_pkl

# ----------------------------------------------------------------------------
# Flask server setup
# ----------------------------------------------------------------------------

app = Flask(__name__)

# Load StyleGAN generator
base_generator = StyleGANGenerator(NETWORK_PKL)
noise_gen = NoiseGenerator(ns=base_generator.z_dim, steps=60)
last_vector = None

# Optional directory for logging generated images. Create a unique
# subfolder for each server instance to avoid overwriting files on
# restart.
parser = argparse.ArgumentParser()
parser.add_argument(
    "--outdir",
    type=str,
    default=os.environ.get("outdir") or os.environ.get("OUTDIR"),
    help="Directory to save generated images as JPG.",
)
args, _ = parser.parse_known_args()
base_outdir = args.outdir
if base_outdir:
    instance_id = f"{random.randint(0, 99999999):08d}"
    outdir = os.path.join(base_outdir, instance_id)
    os.makedirs(outdir, exist_ok=True)
else:
    outdir = None
image_counter = 0

# Precompute model information for the index page
model_name = os.path.basename(NETWORK_PKL)
image_size = getattr(base_generator.G, "img_resolution", "unknown")
model_params = sum(p.numel() for p in base_generator.G.parameters())
model_mode = "training" if base_generator.G.training else "eval"
device = str(base_generator.device)
precision = base_generator.precision
noise_size = noise_gen.noise_size


@app.route("/")
def index_page():
    """Serve the client-side interface."""
    return render_template(
        "index.html",
        title="Home",
        model_name=model_name,
        image_size=image_size,
        model_params=model_params,
        model_mode=model_mode,
        device=device,
        precision=precision,
        noise_size=noise_size,
        noise_step=noise_gen.current_step,
        noise_total_steps=noise_gen.n_steps,
    )


@app.route("/start", methods=["POST"])
def start_walk():
    """Start a new interpolation sequence."""
    global last_vector
    noise_gen._start_new_leg()
    last_vector = None
    return jsonify({"status": "started"})


@app.route("/next", methods=["GET"])
def get_next_image():
    """Return the next image in the latent walk."""
    global last_vector, image_counter
    z = next(noise_gen)
    last_vector = z
    img = base_generator.generate_image(z=z, truncation_psi=0.7)
    if outdir:
        img_path = os.path.join(outdir, f"{image_counter:06d}.jpg")
        img.save(img_path, format="JPEG")
        image_counter += 1
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


@app.route("/vector", methods=["GET"])
def current_vector():
    """Return the latent vector used for the last generated image."""
    if last_vector is None:
        return jsonify({"vector": None}), 404
    return jsonify({"vector": last_vector.tolist()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
