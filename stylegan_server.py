import os
from io import BytesIO

import argparse
import numpy as np
import random
import sqlite3
from flask import Flask, send_file, jsonify, render_template, request, send_from_directory

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

    def load_custom_path(self, custom_vectors):
        """Loads a pre-computed numpy array of vectors for the walk."""
        print(f"Loading custom path with {len(custom_vectors)} steps.")
        self.vectors = custom_vectors
        self.n_steps = len(custom_vectors)
        self.current_step = 0


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
parser.add_argument(
    "--outdir",
    type=str,
    default=os.environ.get("outdir") or os.environ.get("OUTDIR"),
    help="Directory to save generated images as JPG.",
)
parser.add_argument(
    "--db-file",
    type=str,
    default="image_vectors.db",
    help="Path to the SQLite database file.",
)
parser.add_argument(
    "--steps",
    type=int,
    default=60,
    help="Number of steps for each interpolation leg.",
)
args, _ = parser.parse_known_args()

NETWORK_PKL = args.network_pkl
DB_FILE = args.db_file
base_outdir = args.outdir
num_steps = args.steps

# ----------------------------------------------------------------------------
# Flask server setup
# ----------------------------------------------------------------------------

app = Flask(__name__)

# Load StyleGAN generator
base_generator = StyleGANGenerator(NETWORK_PKL)
noise_gen = NoiseGenerator(ns=base_generator.z_dim, steps=num_steps)
last_vector = None

# Create a unique instance identifier and optional output directory for
# logging generated images. Using a unique ID avoids overwriting files on
# restart.
instance_id = f"{random.randint(0, 99999999):08d}"
if base_outdir:
    outdir = os.path.join(base_outdir, instance_id)
    os.makedirs(outdir, exist_ok=True)
else:
    outdir = None
image_counter = 0

# ----------------------------------------------------------------------------
# Database setup
# ----------------------------------------------------------------------------


def init_db():
    """Initializes the database and ensures the table exists."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS generated_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            instance_id TEXT NOT NULL,
            filename TEXT NOT NULL UNIQUE,
            vector BLOB NOT NULL,
            model_pkl TEXT NOT NULL,
            step INTEGER NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()


def add_vector_record(instance_id, filename, vector, model_pkl, step):
    """Adds a record linking a generated image to its latent vector."""
    vector_blob = vector.astype(np.float32).tobytes()
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO generated_images (instance_id, filename, vector, model_pkl, step) VALUES (?, ?, ?, ?, ?)",
            (instance_id, filename, vector_blob, model_pkl, step),
        )
        conn.commit()
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        conn.close()


def get_vector_from_db(filename, db_path):
    """Retrieve a latent vector from the database given its filename."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT vector FROM generated_images WHERE filename = ?", (filename,))
    result = c.fetchone()
    conn.close()
    if result:
        return np.frombuffer(result[0], dtype=np.float32)
    return None

# Precompute model information for the index page
model_name = os.path.basename(NETWORK_PKL)
image_size = getattr(base_generator.G, "img_resolution", "unknown")
model_params = sum(p.numel() for p in base_generator.G.parameters())
model_mode = "training" if base_generator.G.training else "eval"
device = str(base_generator.device)
precision = base_generator.precision
noise_size = noise_gen.noise_size


@app.route("/gallery")
def gallery_page():
    """Renders a page showing all generated images."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT filename FROM generated_images ORDER BY id DESC")
    image_files = [row[0] for row in c.fetchall()]
    conn.close()
    return render_template("gallery.html", images=image_files, title="Image Gallery")


@app.route("/generated_images/<path:filename>")
def serve_generated_image(filename):
    """Serves a file from the configured output directory."""
    return send_from_directory(outdir, filename)


@app.route("/create_custom_walk", methods=["POST"])
def create_custom_walk():
    """Creates an interpolated walk from a list of selected image filenames."""
    data = request.get_json()
    if not data or "filenames" not in data:
        return jsonify({"status": "error", "message": "Missing filenames"}), 400

    filenames = data.get("filenames")
    steps_per_leg = data.get("steps", 60)

    if len(filenames) < 2:
        return jsonify({"status": "error", "message": "Select at least two images"}), 400

    keyframe_vectors = [get_vector_from_db(fname, DB_FILE) for fname in filenames]
    keyframe_vectors = [v for v in keyframe_vectors if v is not None]

    full_path = []
    for i in range(len(keyframe_vectors) - 1):
        z_start = keyframe_vectors[i]
        z_end = keyframe_vectors[i + 1]
        ratios = np.linspace(0, 1, num=steps_per_leg, dtype=np.float32)
        segment = np.array(
            [(1.0 - r) * z_start + r * z_end for r in ratios], dtype=np.float32
        )
        full_path.extend(segment)

    if full_path:
        noise_gen.load_custom_path(np.array(full_path, dtype=np.float32))
        global last_vector
        last_vector = None
        return jsonify({"status": "success", "total_steps": len(full_path)})
    else:
        return jsonify({"status": "error", "message": "Could not create path"}), 500


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
    step = noise_gen.current_step - 1
    img = base_generator.generate_image(z=z, truncation_psi=0.7)
    if outdir:
        filename = f"{image_counter:06d}.jpg"
        img_path = os.path.join(outdir, filename)
        img.save(img_path, format="JPEG")
        add_vector_record(instance_id, filename, z, NETWORK_PKL, step)
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
    init_db()
    app.run(host="0.0.0.0", port=5000)
