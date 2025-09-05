import os
from io import BytesIO
import base64
import argparse
import numpy as np
import sqlite3
import uuid
import queue
import threading
import subprocess
from collections import deque
from flask import Flask, jsonify, render_template, request, send_from_directory

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
parser.add_argument("--steps", type=int, default=120, help="Steps per interpolation leg.")
args, _ = parser.parse_known_args()

NETWORK_PKL = args.network_pkl
DB_FILE = args.db_file
outdir = args.outdir
num_steps = args.steps

if outdir:
    os.makedirs(outdir, exist_ok=True)

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

# Queue for background rendering tasks
render_queue = queue.Queue()

# Deque and locks to track pending and current rendering tasks
pending_walk_ids = deque()
queue_lock = threading.Lock()
rendering_walk_id = None
abort_event = threading.Event()
cancelled_walks = set()
worker_thread = None

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
            step_rate INTEGER NOT NULL DEFAULT 60,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Ensure the step_rate column exists for older databases
    c.execute("PRAGMA table_info(walks)")
    existing_cols = [row[1] for row in c.fetchall()]
    if "step_rate" not in existing_cols:
        c.execute("ALTER TABLE walks ADD COLUMN step_rate INTEGER NOT NULL DEFAULT 60")
    # Create the generated_images table where each image stores its latent vector
    c.execute("""
        CREATE TABLE IF NOT EXISTS generated_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            walk_id INTEGER NOT NULL,
            step_index INTEGER NOT NULL,
            filename TEXT NOT NULL,
            latent_blob BLOB NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (walk_id) REFERENCES walks (id),
            UNIQUE (walk_id, step_index),
            UNIQUE (filename)
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_genimg_walk ON generated_images(walk_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_genimg_filename ON generated_images(filename)")
    conn.commit()
    conn.close()

def create_walk_record(name, type, vectors, model_pkl, step_rate):
    """Saves a new walk definition to the DB and returns its ID."""
    vectors_blob = vectors.astype(np.float32).tobytes()
    num_steps = len(vectors)
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        "INSERT INTO walks (name, type, num_steps, vectors_blob, model_pkl, step_rate) VALUES (?, ?, ?, ?, ?, ?)",
        (name, type, num_steps, vectors_blob, model_pkl, step_rate)
    )
    new_walk_id = c.lastrowid
    conn.commit()
    conn.close()
    return new_walk_id

def add_image_record(walk_id, step_index, relpath, latent):
    """Links a rendered image to a step in a walk and stores its latent vector.

    The `relpath` should be the path relative to `outdir` in the format
    `<walk_id>/<filename>`.
    """
    latent = np.asarray(latent, dtype=np.float32)
    latent_blob = latent.tobytes()

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        "INSERT OR IGNORE INTO generated_images (walk_id, step_index, filename, latent_blob) VALUES (?, ?, ?, ?)",
        (walk_id, step_index, relpath, latent_blob),
    )
    c.execute(
        "UPDATE generated_images SET latent_blob = ? WHERE walk_id = ? AND step_index = ? AND filename = ?",
        (latent_blob, walk_id, step_index, relpath),
    )
    conn.commit()
    conn.close()

def get_walk_vectors(walk_id):
    """Retrieves and decodes the planned vectors for a given walk ID."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT vectors_blob, num_steps FROM walks WHERE id = ?", (walk_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    blob, n = row
    if n <= 0:
        return None
    vec = np.frombuffer(blob, dtype=np.float32)
    if vec.size % n != 0:
        raise ValueError(
            f"Corrupt vectors for walk {walk_id}: size={vec.size} not divisible by num_steps={n}"
        )
    latent_dim = vec.size // n
    return vec.reshape(n, latent_dim)

def get_vector_by_image_id(image_id):
    """Returns the latent vector stored with a rendered image."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT latent_blob FROM generated_images WHERE id = ?", (image_id,))
    row = c.fetchone()
    conn.close()
    if not row or row[0] is None:
        return None
    return np.frombuffer(row[0], dtype=np.float32)

# ----------------------------------------------------------------------------
# Background Worker for Rendering
# ----------------------------------------------------------------------------

def render_worker():
    global rendering_walk_id
    while True:
        walk_id = render_queue.get()
        with queue_lock:
            if walk_id in cancelled_walks:
                cancelled_walks.remove(walk_id)
                render_queue.task_done()
                continue
            try:
                pending_walk_ids.remove(walk_id)
            except ValueError:
                pass
            rendering_walk_id = walk_id
            abort_event.clear()
        print(f"Starting rendering walk {walk_id}. Queue length: {render_queue.qsize()}")
        try:
            try:
                vectors = get_walk_vectors(walk_id)
                if vectors is None:
                    continue
                conn = sqlite3.connect(DB_FILE)
                c = conn.cursor()
                c.execute(
                    "SELECT step_index FROM generated_images WHERE walk_id = ?",
                    (walk_id,),
                )
                existing = {row[0] for row in c.fetchall()}
                conn.close()
            except Exception as e:
                print(f"Database error for walk {walk_id}: {e}")
                continue
            for step, z in enumerate(vectors):
                if abort_event.is_set():
                    break
                if step in existing:
                    continue
                img = base_generator.generate_image(z=z, truncation_psi=0.7)
                if outdir:
                    img_subdir = os.path.join(outdir, str(walk_id))
                    os.makedirs(img_subdir, exist_ok=True)
                    filename = f"step_{step:04d}.jpg"
                    img_path = os.path.join(img_subdir, filename)
                    img.save(img_path, format="JPEG")
                    relpath = f"{walk_id}/{filename}"
                    try:
                        add_image_record(walk_id, step, relpath, latent=z)
                    except Exception as e:
                        print(
                            f"Database error adding image record for walk {walk_id}, step {step}: {e}"
                        )
            # After rendering all steps, attempt to create a video using ffmpeg
            if outdir and not abort_event.is_set():
                img_subdir = os.path.join(outdir, str(walk_id))
                video_path = os.path.join(img_subdir, "walk.mp4")
                if os.path.exists(img_subdir) and not os.path.exists(video_path):
                    cmd = [
                        "ffmpeg", "-y",
                        "-framerate", "30",
                        "-i", os.path.join(img_subdir, "step_%04d.jpg"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        video_path,
                    ]
                    try:
                        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    except Exception as e:
                        print(f"ffmpeg failed for walk {walk_id}: {e}")
        finally:
            with queue_lock:
                rendering_walk_id = None
                abort_event.clear()
            render_queue.task_done()
            print(f"Finished rendering walk {walk_id}. Queue length: {render_queue.qsize()}")


@app.before_first_request
def start_worker():
    init_db()
    global worker_thread
    if worker_thread is None:
        worker_thread = threading.Thread(target=render_worker, daemon=True)
        worker_thread.start()

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

    step_rate = num_steps
    if request.is_json and "steps" in request.json:
        step_rate = int(request.json["steps"])
    else:
        step_rate = int(request.args.get("steps", num_steps))

    interpolator = LatentInterpolator(base_generator.z_dim, n_steps=step_rate)
    vectors = interpolator.random_walk(num_segments=segments)

    walk_name = f"random_{uuid.uuid4().hex[:8]}"
    walk_id = create_walk_record(walk_name, 'random', vectors, NETWORK_PKL, step_rate)

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


@app.route("/enqueue_walk/<int:walk_id>", methods=["POST"])
def enqueue_walk(walk_id):
    """Adds a walk ID to the background rendering queue."""
    render_queue.put(walk_id)
    with queue_lock:
        pending_walk_ids.append(walk_id)
        queue_length = render_queue.qsize()
    print(f"Enqueued walk {walk_id}. Queue length: {queue_length}")
    return jsonify({"status": "enqueued", "walk_id": walk_id})


@app.route("/queue_status", methods=["GET"])
def queue_status():
    """Returns the current rendering walk and pending queue."""
    with queue_lock:
        pending = list(pending_walk_ids)
        current = rendering_walk_id
    return jsonify({"rendering": current, "pending": pending})


@app.route("/queue_item/<int:walk_id>", methods=["DELETE"])
def delete_queue_item(walk_id):
    """Removes a walk from the queue or aborts it if currently rendering."""
    with queue_lock:
        if rendering_walk_id == walk_id:
            abort_event.set()
            return jsonify({"status": "aborting", "walk_id": walk_id})
        try:
            pending_walk_ids.remove(walk_id)
            cancelled_walks.add(walk_id)
            return jsonify({"status": "removed", "walk_id": walk_id})
        except ValueError:
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
        img_subdir = os.path.join(outdir, f"{current_walk['walk_id']}")
        os.makedirs(img_subdir, exist_ok=True)
        filename = f"step_{step:04d}.jpg"
        img_path = os.path.join(img_subdir, filename)
        img.save(img_path, format="JPEG")
        relpath = f"{current_walk['walk_id']}/{filename}"
        add_image_record(current_walk["walk_id"], step, relpath, latent=z)

    current_walk["current_step"] += 1

    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return jsonify(
        {
            "image": img_b64,
            "walk_id": current_walk["walk_id"],
            "step": step,
            "total_steps": len(current_walk["vectors"]),
        }
    )


# --- Gallery and Custom Walk Creation Routes (Adapted for new schema) ---

@app.route('/gallery')
def gallery_page():
    """Renders a gallery of all defined walks and their rendered images."""
    conn = sqlite3.connect(DB_FILE)
    # Fetch all walks first
    walks_cursor = conn.cursor()
    walks_cursor.execute("SELECT id, name, type, num_steps, step_rate FROM walks ORDER BY id DESC")
    all_walks = walks_cursor.fetchall()

    # Fetch all rendered images and group them by walk_id
    images_cursor = conn.cursor()
    images_cursor.execute("SELECT id, walk_id, filename FROM generated_images")

    images_by_walk = {}
    for img_id, walk_id, relpath in images_cursor.fetchall():
        if walk_id not in images_by_walk:
            images_by_walk[walk_id] = []
        fname = relpath.split('/', 1)[1] if '/' in relpath else relpath
        images_by_walk[walk_id].append({
            'id': img_id,
            'filename': fname,
        })

    conn.close()

    videos_by_walk = {}
    if outdir:
        for walk in all_walks:
            video_path = os.path.join(outdir, str(walk[0]), "walk.mp4")
            videos_by_walk[walk[0]] = os.path.exists(video_path)

    return render_template('gallery.html', walks=all_walks, images_by_walk=images_by_walk, videos_by_walk=videos_by_walk)


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

@app.route('/generated_images/<int:walk_id>/<filename>')
def serve_generated_image(walk_id, filename):
    if not outdir:
        return ("Outdir not configured", 404)
    return send_from_directory(os.path.join(outdir, str(walk_id)), filename)


@app.route('/generated_videos/<int:walk_id>/<filename>')
def serve_generated_video(walk_id, filename):
    if not outdir:
        return ("Outdir not configured", 404)
    return send_from_directory(os.path.join(outdir, str(walk_id)), filename)


@app.route('/create_custom_walk', methods=['POST'])
def create_custom_walk():
    data = request.get_json(silent=True) or {}
    image_ids = data.get('ids') or []
    try:
        steps_per_leg = int(data.get('steps', num_steps))
    except (TypeError, ValueError):
        return jsonify({"status": "error", "message": "`steps` must be an integer"}), 400

    if len(image_ids) < 2:
        return jsonify({"status": "error", "message": "Select at least two images"}), 400
    if steps_per_leg < 2:
        return jsonify({"status": "error", "message": "`steps` must be >= 2"}), 400

    keyframes = []
    for iid in image_ids:
        v = get_vector_by_image_id(int(iid))
        if v is None:
            return jsonify({"status": "error", "message": f"Image id {iid} not found or missing latent"}), 404
        keyframes.append(np.asarray(v, dtype=np.float32))

    dims = {v.shape[0] for v in keyframes}
    if len(dims) != 1:
        return jsonify({
            "status": "error",
            "message": f"Selected images have different latent lengths {sorted(dims)}; cannot interpolate."
        }), 400

    ratios = np.linspace(0.0, 1.0, num=steps_per_leg, dtype=np.float32)
    segments = []
    for a, b in zip(keyframes, keyframes[1:]):
        seg = (1.0 - ratios)[:, None] * a[None, :] + ratios[:, None] * b[None, :]
        segments.append(seg)

    vectors = np.vstack(segments).astype(np.float32, copy=False)

    walk_name = f"curated_{uuid.uuid4().hex[:8]}"
    walk_id = create_walk_record(walk_name, 'curated', vectors, NETWORK_PKL, steps_per_leg)

    current_walk["walk_id"] = walk_id
    current_walk["vectors"] = vectors
    current_walk["current_step"] = 0

    return jsonify({"status": "success", "walk_id": walk_id, "name": walk_name, "steps": len(vectors)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
