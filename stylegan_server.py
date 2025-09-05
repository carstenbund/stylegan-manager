import os
from io import BytesIO
import base64
import argparse
import numpy as np
import uuid
import queue
import threading
import subprocess
import shutil
from flask import Flask, jsonify, render_template, request, send_from_directory

from stylegan_manager.models import StyleGANGenerator
from stylegan_manager import RandomWalk
from stylegan_manager.walks.custom_walk import interpolate_vectors
from stylegan_manager.videos.manager import VideoManager
from stylegan_manager.db import (
    init_db,
    init_archive_db,
    create_walk,
    add_image_record,
    get_walk_vectors,
    get_vector_by_image_id,
    get_walk_metadata,
    fetch_all_walks,
    fetch_all_images,
    fetch_archived_walks,
    archive_walk as db_archive_walk,
    restore_walk as db_restore_walk,
    delete_walk as db_delete_walk,
    delete_archived_walk as db_delete_archived_walk,
    existing_step_indices,
)

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
ARCHIVE_DB_FILE = "archive.db"

if outdir:
    os.makedirs(outdir, exist_ok=True)

app = Flask(__name__)

# ----------------------------------------------------------------------------
# Global State & Generator Loading
# ----------------------------------------------------------------------------
base_generator = StyleGANGenerator(NETWORK_PKL)


class LatentTracker:
    """Wrapper that records latent vectors passed to the generator."""

    def __init__(self, model):
        self.model = model
        self.latents = []
        self.latent_dim = getattr(model, "z_dim", 512)

    def generate_image(self, z):
        self.latents.append(np.asarray(z, dtype=np.float32))
        return self.model.generate_image(z=z, truncation_psi=0.7)

# This global dictionary will hold the currently active walk's data
current_walk = {
    "walk_id": None,
    "vectors": None,
    "current_step": 0
}

# Queue for background rendering tasks
render_queue = queue.Queue()

# Video manager to track queue and rendering state
video_manager = VideoManager()
abort_event = threading.Event()
worker_thread = None

# ----------------------------------------------------------------------------
# Background Worker for Rendering
# ----------------------------------------------------------------------------

def render_worker():
    while True:
        walk_id = render_queue.get()
        if video_manager.is_cancelled(walk_id):
            render_queue.task_done()
            continue
        video_manager.mark_rendering(walk_id)
        abort_event.clear()
        print(f"Starting rendering walk {walk_id}. Queue length: {render_queue.qsize()}")
        try:
            try:
                vectors = get_walk_vectors(DB_FILE, walk_id)
                if vectors is None:
                    continue
                existing = existing_step_indices(DB_FILE, walk_id)
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
                        add_image_record(DB_FILE, walk_id, step, relpath, latent=z)
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
            video_manager.mark_rendered(walk_id)
            abort_event.clear()
            render_queue.task_done()
            print(f"Finished rendering walk {walk_id}. Queue length: {render_queue.qsize()}")


@app.before_first_request
def start_worker():
    init_db(DB_FILE)
    init_archive_db(ARCHIVE_DB_FILE)
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
    if request.is_json:
        if "keyframes" in request.json:
            segments = max(int(request.json["keyframes"]) - 1, 1)
        elif "segments" in request.json:
            segments = int(request.json["segments"])
    else:
        keyframes_arg = request.args.get("keyframes")
        if keyframes_arg is not None:
            segments = max(int(keyframes_arg) - 1, 1)
        else:
            segments = int(request.args.get("segments", 1))

    step_rate = num_steps
    if request.is_json and "steps" in request.json:
        step_rate = int(request.json["steps"])
    else:
        step_rate = int(request.args.get("steps", num_steps))

    total_steps = segments * step_rate
    tracker = LatentTracker(base_generator)
    walk = RandomWalk(tracker, steps=total_steps)
    walk.generate()
    vectors = np.concatenate(tracker.latents, axis=0).astype(np.float32, copy=False)

    walk_name = f"random_{uuid.uuid4().hex[:8]}"
    walk_id = create_walk(DB_FILE, walk_name, 'random', vectors, NETWORK_PKL, step_rate)

    # Load this new walk as the current one
    current_walk["walk_id"] = walk_id
    current_walk["vectors"] = vectors
    current_walk["current_step"] = 0

    return jsonify({
        "status": "created and loaded",
        "walk_id": walk_id,
        "name": walk_name,
        "steps": len(vectors),
    })


@app.route("/load_walk/<int:walk_id>", methods=["POST"])
def load_walk(walk_id):
    """Loads an existing walk's vectors into the active state for rendering."""
    vectors = get_walk_vectors(DB_FILE, walk_id)
    if vectors is not None:
        current_walk["walk_id"] = walk_id
        current_walk["vectors"] = vectors
        current_walk["current_step"] = 0
        return jsonify({"status": "loaded", "walk_id": walk_id, "steps": len(vectors)})
    return jsonify({"status": "error", "message": "Walk not found"}), 404


@app.route("/enqueue_walk/<int:walk_id>", methods=["POST"])
def enqueue_walk(walk_id):
    """Clone an existing walk and enqueue the new copy for rendering.

    A new step rate can be provided in the request payload or query string.
    If supplied and different from the original walk's step rate, the walk's
    vectors are re-interpolated to match the new step rate.
    """

    # Retrieve the original walk's vectors
    vectors = get_walk_vectors(DB_FILE, walk_id)
    if vectors is None:
        return jsonify({"status": "error", "message": "Walk not found"}), 404

    # Fetch metadata from the original walk
    row = get_walk_metadata(DB_FILE, walk_id)
    if not row:
        return jsonify({"status": "error", "message": "Walk not found"}), 404

    name, walk_type, model_pkl, orig_step_rate = row

    # Determine desired step rate
    new_step_rate = orig_step_rate
    if request.is_json and "steps" in request.json:
        try:
            new_step_rate = int(request.json["steps"])
        except (TypeError, ValueError):
            return (
                jsonify({"status": "error", "message": "`steps` must be an integer"}),
                400,
            )
    else:
        steps_arg = request.args.get("steps")
        if steps_arg is not None:
            try:
                new_step_rate = int(steps_arg)
            except (TypeError, ValueError):
                return (
                    jsonify({"status": "error", "message": "`steps` must be an integer"}),
                    400,
                )

    new_vectors = vectors
    # Recompute vectors if step rate changed for curated walks
    if walk_type == "curated" and new_step_rate != orig_step_rate:
        total = len(vectors)
        if orig_step_rate > 0 and (total - 1) // orig_step_rate > 0:
            segments = (total - 1) // orig_step_rate
            keyframes = [vectors[i * orig_step_rate] for i in range(segments)]
            keyframes.append(vectors[-1])
        else:
            keyframes = [vectors[0], vectors[-1]]
        new_vectors = interpolate_vectors(keyframes, new_step_rate).astype(
            np.float32, copy=False
        )

    new_name = f"{name}_render"

    # Create a cloned walk record with the new step rate
    new_walk_id = create_walk(DB_FILE, new_name, walk_type, new_vectors, model_pkl, new_step_rate)

    # Enqueue the new walk ID
    render_queue.put(new_walk_id)
    video_manager.mark_queued(new_walk_id)
    queue_length = render_queue.qsize()
    print(
        f"Cloned walk {walk_id} as {new_walk_id} and enqueued. Queue length: {queue_length}"
    )
    return jsonify({"status": "enqueued", "walk_id": new_walk_id})


@app.route("/queue_status", methods=["GET"])
def queue_status():
    """Returns the current rendering walk and pending queue."""
    return jsonify(video_manager.queue_status())


@app.route("/queue_item/<int:walk_id>", methods=["DELETE"])
def delete_queue_item(walk_id):
    """Removes a walk from the queue or aborts it if currently rendering."""
    if video_manager.current() == walk_id:
        abort_event.set()
        return jsonify({"status": "aborting", "walk_id": walk_id})
    if video_manager.remove_from_queue(walk_id):
        return jsonify({"status": "removed", "walk_id": walk_id})
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
        add_image_record(DB_FILE, current_walk["walk_id"], step, relpath, latent=z)

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
    all_walks = fetch_all_walks(DB_FILE)

    images_by_walk = {}
    for img_id, walk_id, relpath in fetch_all_images(DB_FILE):
        if walk_id not in images_by_walk:
            images_by_walk[walk_id] = []
        fname = relpath.split('/', 1)[1] if '/' in relpath else relpath
        images_by_walk[walk_id].append({
            'id': img_id,
            'filename': fname,
        })

    videos_by_walk = {}
    if outdir:
        for walk in all_walks:
            video_path = os.path.join(outdir, str(walk[0]), "walk.mp4")
            videos_by_walk[walk[0]] = os.path.exists(video_path)

    return render_template('gallery.html', walks=all_walks, images_by_walk=images_by_walk, videos_by_walk=videos_by_walk)


@app.route('/archive')
def archive_page():
    """Renders a gallery of archived walks."""
    walks = fetch_archived_walks(ARCHIVE_DB_FILE)
    return render_template('archive.html', walks=walks)


@app.route('/archive_walk/<int:walk_id>', methods=['POST'])
def archive_walk(walk_id):
    """Moves a walk to the archive database and removes its data."""
    note = ""
    if request.is_json:
        note = request.json.get('note', '').strip()
    if not db_archive_walk(DB_FILE, ARCHIVE_DB_FILE, walk_id, note):
        return jsonify({"status": "error", "message": "Walk not found"}), 404

    if outdir:
        walk_dir = os.path.join(outdir, str(walk_id))
        if os.path.isdir(walk_dir):
            shutil.rmtree(walk_dir, ignore_errors=True)

    return jsonify({"status": "success", "walk_id": walk_id})


@app.route('/delete_archived_walk/<int:walk_id>', methods=['DELETE'])
def delete_archived_walk(walk_id):
    """Deletes a walk from the archive database."""
    db_delete_archived_walk(ARCHIVE_DB_FILE, walk_id)
    return jsonify({"status": "success", "walk_id": walk_id})


@app.route('/restore_walk/<int:archived_id>', methods=['POST'])
def restore_walk(archived_id):
    """Restores a walk from the archive to the main database."""
    queue_flag = request.args.get('queue') in ('1', 'true', 'yes')

    restored_id = db_restore_walk(ARCHIVE_DB_FILE, DB_FILE, archived_id)
    if restored_id is None:
        return jsonify({"status": "error", "message": "Walk not found"}), 404

    if queue_flag:
        render_queue.put(restored_id)
        video_manager.mark_queued(restored_id)

    return jsonify({"status": "success", "walk_id": restored_id})


@app.route('/delete_walk/<int:walk_id>', methods=['DELETE'])
def delete_walk(walk_id):
    """Deletes a walk and all of its associated images from disk and database."""
    image_paths = db_delete_walk(DB_FILE, walk_id)

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

    queue_render = bool(data.get('queue', False))
    loop = bool(data.get('loop'))

    if len(image_ids) < 2:
        return jsonify({"status": "error", "message": "Select at least two images"}), 400
    if steps_per_leg < 2:
        return jsonify({"status": "error", "message": "`steps` must be >= 2"}), 400

    keyframes = []
    for iid in image_ids:
        v = get_vector_by_image_id(DB_FILE, int(iid))
        if v is None:
            return jsonify({"status": "error", "message": f"Image id {iid} not found or missing latent"}), 404
        keyframes.append(np.asarray(v, dtype=np.float32))

    dims = {v.shape[0] for v in keyframes}
    if len(dims) != 1:
        return jsonify({
            "status": "error",
            "message": f"Selected images have different latent lengths {sorted(dims)}; cannot interpolate."
        }), 400

    points = keyframes + [keyframes[0]] if loop and len(keyframes) > 1 else keyframes
    vectors = interpolate_vectors(points, steps_per_leg).astype(np.float32, copy=False)

    walk_name = f"curated_{uuid.uuid4().hex[:8]}"
    walk_id = create_walk(DB_FILE, walk_name, 'curated', vectors, NETWORK_PKL, steps_per_leg)

    if queue_render:
        render_queue.put(walk_id)
        video_manager.mark_queued(walk_id)
        return jsonify({
            "status": "queued",
            "walk_id": walk_id,
            "name": walk_name,
            "steps": len(vectors),
        })

    current_walk["walk_id"] = walk_id
    current_walk["vectors"] = vectors
    current_walk["current_step"] = 0

    return jsonify({"status": "success", "walk_id": walk_id, "name": walk_name, "steps": len(vectors)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
