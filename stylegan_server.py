import os
from io import BytesIO
import argparse
import numpy as np
import sqlite3
import uuid
import random
from flask import Flask, send_file, jsonify, render_template, request, send_from_directory

from stylegan_gen import StyleGANGenerator
from utils import LatentInterpolator

# ----------------------------------------------------------------------------
# Argument parsing & Initial Setup
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
    help="Directory to save generated images.",
)
parser.add_argument("--db-file", type=str, default="walks.db")
parser.add_argument("--steps", type=int, default=60, help="Steps per interpolation leg.")
args, _ = parser.parse_known_args()

NETWORK_PKL = args.network_pkl
DB_FILE = args.db_file
outdir = args.outdir
num_steps = args.steps

if outdir:
    os.makedirs(outdir, exist_ok=True)

# Unique identifier for this server instance. Can be overridden via the
# INSTANCE_ID environment variable for coordinating across multiple nodes.
instance_id = os.environ.get("INSTANCE_ID", uuid.uuid4().hex)

app = Flask(__name__)

# ----------------------------------------------------------------------------
# Global State & Generator Loading
# ----------------------------------------------------------------------------
base_generator = StyleGANGenerator(NETWORK_PKL)

# This global dictionary will hold the currently active walk's data
current_walk = {
    "walk_id": None,
    "vectors": None,
    "current_step": 0
}

# ----------------------------------------------------------------------------
# Database Helper Functions
# ----------------------------------------------------------------------------
def init_db():
    """Initializes the database with the new two-table schema."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # Create the walks table
    c.execute("""
        CREATE TABLE IF NOT EXISTS walks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            num_steps INTEGER NOT NULL,
            vectors_blob BLOB NOT NULL,
            model_pkl TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Create the generated_images table with a foreign key
    c.execute("""
        CREATE TABLE IF NOT EXISTS generated_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            walk_id INTEGER NOT NULL,
            step_index INTEGER NOT NULL,
            filename TEXT NOT NULL UNIQUE,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (walk_id) REFERENCES walks (id)
        )
    """)
    conn.commit()
    conn.close()

def create_walk_record(name, type, vectors, model_pkl):
    """Saves a new walk definition to the DB and returns its ID."""
    vectors_blob = vectors.astype(np.float32).tobytes()
    num_steps = len(vectors)
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        "INSERT INTO walks (name, type, num_steps, vectors_blob, model_pkl) VALUES (?, ?, ?, ?, ?)",
        (name, type, num_steps, vectors_blob, model_pkl)
    )
    new_walk_id = c.lastrowid
    conn.commit()
    conn.close()
    return new_walk_id

def add_image_record(walk_id, step_index, relpath):
    """Links a rendered image to a step in a walk.

    The `relpath` should be the path relative to `outdir` in the format
    `<instance_id>/<rand8>/<filename>`.
    """
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        "INSERT INTO generated_images (walk_id, step_index, filename) VALUES (?, ?, ?)",
        (walk_id, step_index, relpath),
    )
    conn.commit()
    conn.close()

def get_walk_vectors(walk_id):
    """Retrieves and decodes the vectors for a given walk ID."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT vectors_blob FROM walks WHERE id = ?", (walk_id,))
    result = c.fetchone()
    conn.close()
    if result:
        return np.frombuffer(result[0], dtype=np.float32).reshape(-1, base_generator.z_dim)
    return None

def get_vector_by_image_id(image_id):
    """Gets a single vector by looking up an image's walk and step."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # Find the walk and step for the image
    c.execute("SELECT walk_id, step_index FROM generated_images WHERE id = ?", (image_id,))
    result = c.fetchone()
    if not result:
        conn.close()
        return None
    walk_id, step_index = result

    # Retrieve the entire vector blob for that walk
    c.execute("SELECT vectors_blob, num_steps FROM walks WHERE id = ?", (walk_id,))
    blob_result = c.fetchone()
    conn.close()
    if blob_result:
        vectors_blob, num_steps = blob_result
        all_vectors = np.frombuffer(vectors_blob, dtype=np.float32).reshape(num_steps, -1)
        # Return the specific vector at the correct index
        return all_vectors[step_index]
    return None

# ----------------------------------------------------------------------------
# Flask Routes
# ----------------------------------------------------------------------------

@app.route("/")
@app.route("/index")
def index_page():
    """Serve the client-side interface."""
    return render_template("index.html", title="Home")


@app.route("/start_random_walk", methods=["POST"])
def start_random_walk():
    """Defines a new random walk, saves it, and loads it for rendering."""
    segments = 1
    if request.is_json and "segments" in request.json:
        segments = int(request.json["segments"])
    else:
        segments = int(request.args.get("segments", 1))

    interpolator = LatentInterpolator(base_generator.z_dim, n_steps=num_steps)
    vectors = interpolator.random_walk(num_segments=segments)

    walk_name = f"random_{uuid.uuid4().hex[:8]}"
    walk_id = create_walk_record(walk_name, 'random', vectors, NETWORK_PKL)

    # Load this new walk as the current one
    current_walk["walk_id"] = walk_id
    current_walk["vectors"] = vectors
    current_walk["current_step"] = 0

    return jsonify({"status": "created and loaded", "walk_id": walk_id, "name": walk_name})


@app.route("/load_walk/<int:walk_id>", methods=["POST"])
def load_walk(walk_id):
    """Loads an existing walk's vectors into the active state for rendering."""
    vectors = get_walk_vectors(walk_id)
    if vectors is not None:
        current_walk["walk_id"] = walk_id
        current_walk["vectors"] = vectors
        current_walk["current_step"] = 0
        return jsonify({"status": "loaded", "walk_id": walk_id, "steps": len(vectors)})
    return jsonify({"status": "error", "message": "Walk not found"}), 404


@app.route("/next_image", methods=["GET"])
def get_next_image():
    """Renders the next image from the currently loaded walk."""
    if current_walk["walk_id"] is None or current_walk["vectors"] is None:
        return jsonify({"error": "No walk is currently loaded. Please start or load one."}), 400

    step = current_walk["current_step"]
    if step >= len(current_walk["vectors"]):
        return jsonify({"status": "walk complete"}), 404

    z = current_walk["vectors"][step]
    img = base_generator.generate_image(z=z, truncation_psi=0.7)

    if outdir:
        rand8 = random.randint(0, 99999999)
        img_subdir = os.path.join(outdir, f"{instance_id}/{rand8:08d}")
        os.makedirs(img_subdir, exist_ok=True)
        filename = f"walk_{current_walk['walk_id']:04d}_step_{step:04d}.jpg"
        img_path = os.path.join(img_subdir, filename)
        img.save(img_path, format="JPEG")
        relpath = f"{instance_id}/{rand8:08d}/{filename}"
        add_image_record(current_walk["walk_id"], step, relpath)

    current_walk["current_step"] += 1

    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


# --- Gallery and Custom Walk Creation Routes (Adapted for new schema) ---

@app.route('/gallery')
def gallery_page():
    """Renders a gallery of all defined walks and their rendered images."""
    conn = sqlite3.connect(DB_FILE)
    # Fetch all walks first
    walks_cursor = conn.cursor()
    walks_cursor.execute("SELECT id, name, type, num_steps FROM walks ORDER BY id DESC")
    all_walks = walks_cursor.fetchall()

    # Fetch all rendered images and group them by walk_id
    images_cursor = conn.cursor()
    images_cursor.execute("SELECT id, walk_id, filename FROM generated_images")

    images_by_walk = {}
    for img_id, walk_id, relpath in images_cursor.fetchall():
        if walk_id not in images_by_walk:
            images_by_walk[walk_id] = []
        inst, rand8, fname = relpath.split('/', 2)
        images_by_walk[walk_id].append({
            'id': img_id,
            'instance_id': inst,
            'rand8': rand8,
            'filename': fname,
        })

    conn.close()
    return render_template('gallery.html', walks=all_walks, images_by_walk=images_by_walk)


@app.route('/delete_walk/<int:walk_id>', methods=['DELETE'])
def delete_walk(walk_id):
    """Deletes a walk and all of its associated images from disk and database."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    # Gather image paths before deleting records
    c.execute("SELECT filename FROM generated_images WHERE walk_id = ?", (walk_id,))
    image_paths = [row[0] for row in c.fetchall()]

    # Remove database records
    c.execute("DELETE FROM generated_images WHERE walk_id = ?", (walk_id,))
    c.execute("DELETE FROM walks WHERE id = ?", (walk_id,))
    conn.commit()
    conn.close()

    # Delete image files from disk
    if outdir:
        for relpath in image_paths:
            filepath = os.path.join(outdir, relpath)
            try:
                os.remove(filepath)
                # Attempt to clean up empty directories
                dirpath = os.path.dirname(filepath)
                while dirpath.startswith(outdir) and dirpath != outdir:
                    if not os.listdir(dirpath):
                        os.rmdir(dirpath)
                        dirpath = os.path.dirname(dirpath)
                    else:
                        break
            except FileNotFoundError:
                pass

    return jsonify({"status": "success", "walk_id": walk_id})

@app.route('/generated_images/<instance_id>/<rand8>/<filename>')
def serve_generated_image(instance_id, rand8, filename):
    return send_from_directory(os.path.join(outdir, instance_id, rand8), filename)


@app.route('/create_custom_walk', methods=['POST'])
def create_custom_walk():
    """Creates a new walk definition from selected keyframe images."""
    data = request.get_json()
    image_ids = data.get('ids', [])
    steps_per_leg = data.get('steps', num_steps)

    if len(image_ids) < 2:
        return jsonify({"status": "error", "message": "Select at least two images"}), 400

    # Get the single vector for each selected keyframe image
    keyframe_vectors = [get_vector_by_image_id(img_id) for img_id in image_ids]
    keyframe_vectors = [v for v in keyframe_vectors if v is not None]

    # Interpolate between keyframes
    full_path = []
    for i in range(len(keyframe_vectors) - 1):
        z_start, z_end = keyframe_vectors[i], keyframe_vectors[i+1]
        ratios = np.linspace(0, 1, num=steps_per_leg, dtype=np.float32)
        segment = np.array([(1.0 - r) * z_start + r * z_end for r in ratios], dtype=np.float32)
        full_path.extend(segment)

    if full_path:
        walk_name = f"curated_{uuid.uuid4().hex[:8]}"
        vectors = np.array(full_path, dtype=np.float32)

        # Save the new curated walk to the database
        walk_id = create_walk_record(walk_name, 'curated', vectors, NETWORK_PKL)

        # Automatically load it
        current_walk["walk_id"] = walk_id
        current_walk["vectors"] = vectors
        current_walk["current_step"] = 0

        return jsonify({"status": "success", "walk_id": walk_id, "name": walk_name})

    return jsonify({"status": "error", "message": "Could not create path"}), 500


if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5000)
