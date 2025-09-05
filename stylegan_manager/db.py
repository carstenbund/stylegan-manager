import sqlite3
from typing import List, Optional, Tuple, Set, Dict

import numpy as np


# ----------------------------------------------------------------------------
# Database initialization
# ----------------------------------------------------------------------------

def init_db(db_file: str) -> None:
    """Initializes the main database with the walks and images tables."""
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS walks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            num_steps INTEGER NOT NULL,
            vectors_blob BLOB NOT NULL,
            model_pkl TEXT NOT NULL,
            step_rate INTEGER NOT NULL DEFAULT 60,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            note TEXT
        )
        """
    )
    # Ensure columns exist for older DBs
    c.execute("PRAGMA table_info(walks)")
    existing_cols = [row[1] for row in c.fetchall()]
    if "step_rate" not in existing_cols:
        c.execute("ALTER TABLE walks ADD COLUMN step_rate INTEGER NOT NULL DEFAULT 60")
    if "note" not in existing_cols:
        c.execute("ALTER TABLE walks ADD COLUMN note TEXT")

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS generated_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            walk_id INTEGER NOT NULL,
            step_index INTEGER NOT NULL,
            filename TEXT NOT NULL,
            latent_blob BLOB NOT NULL,
            liked INTEGER NOT NULL DEFAULT 0,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (walk_id) REFERENCES walks (id),
            UNIQUE (walk_id, step_index),
            UNIQUE (filename)
        )
        """
    )
    # Ensure columns exist for older DBs
    c.execute("PRAGMA table_info(generated_images)")
    existing_cols = [row[1] for row in c.fetchall()]
    if "liked" not in existing_cols:
        c.execute("ALTER TABLE generated_images ADD COLUMN liked INTEGER NOT NULL DEFAULT 0")
    c.execute("CREATE INDEX IF NOT EXISTS idx_genimg_walk ON generated_images(walk_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_genimg_filename ON generated_images(filename)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_genimg_liked ON generated_images(liked)")
    conn.commit()
    conn.close()

def init_archive_db(archive_file: str) -> None:
    """Initializes the archive database."""
    conn = sqlite3.connect(archive_file)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS walks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            num_steps INTEGER NOT NULL,
            vectors_blob BLOB NOT NULL,
            model_pkl TEXT NOT NULL,
            step_rate INTEGER NOT NULL DEFAULT 60,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            note TEXT
        )
        """
    )
    c.execute("PRAGMA table_info(walks)")
    existing_cols = [row[1] for row in c.fetchall()]
    if "step_rate" not in existing_cols:
        c.execute("ALTER TABLE walks ADD COLUMN step_rate INTEGER NOT NULL DEFAULT 60")
    if "note" not in existing_cols:
        c.execute("ALTER TABLE walks ADD COLUMN note TEXT")
    conn.commit()
    conn.close()


# ----------------------------------------------------------------------------
# Walk CRUD
# ----------------------------------------------------------------------------

def create_walk(db_file: str, name: str, walk_type: str, vectors: np.ndarray, model_pkl: str, step_rate: int) -> int:
    """Insert a new walk into the database and return its ID."""
    vectors_blob = np.asarray(vectors, dtype=np.float32).tobytes()
    num_steps = len(vectors)
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute(
        "INSERT INTO walks (name, type, num_steps, vectors_blob, model_pkl, step_rate) VALUES (?,?,?,?,?,?)",
        (name, walk_type, num_steps, vectors_blob, model_pkl, step_rate),
    )
    walk_id = c.lastrowid
    conn.commit()
    conn.close()
    return walk_id

def get_walk_vectors(db_file: str, walk_id: int) -> Optional[np.ndarray]:
    conn = sqlite3.connect(db_file)
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
    return vec.reshape(n, latent_dim).copy()

def get_walk_metadata(db_file: str, walk_id: int) -> Optional[Tuple[str, str, str, int]]:
    """Return (name, type, model_pkl, step_rate) for a walk."""
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute("SELECT name, type, model_pkl, step_rate FROM walks WHERE id = ?", (walk_id,))
    row = c.fetchone()
    conn.close()
    return row


def get_walk_info(db_file: str, walk_id: int) -> Optional[Dict[str, object]]:
    """Return walk metadata as a dictionary."""
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute(
        "SELECT id, name, type, num_steps, step_rate, note FROM walks WHERE id = ?",
        (walk_id,),
    )
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    keys = ["id", "name", "type", "num_steps", "step_rate", "note"]
    return dict(zip(keys, row))


def update_walk_info(
    db_file: str,
    walk_id: int,
    name: Optional[str] = None,
    step_rate: Optional[int] = None,
    note: Optional[str] = None,
) -> bool:
    """Update editable fields for a walk. Returns True if a row was updated."""
    updates = []
    values = []
    if name is not None:
        updates.append("name = ?")
        values.append(name)
    if step_rate is not None:
        updates.append("step_rate = ?")
        values.append(step_rate)
    if note is not None:
        updates.append("note = ?")
        values.append(note)
    if not updates:
        return False
    values.append(walk_id)
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute(f"UPDATE walks SET {', '.join(updates)} WHERE id = ?", tuple(values))
    conn.commit()
    success = c.rowcount > 0
    conn.close()
    return bool(success)

def fetch_all_walks(db_file: str) -> List[Tuple]:
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute("SELECT id, name, type, num_steps, step_rate FROM walks ORDER BY id DESC")
    walks = c.fetchall()
    conn.close()
    return walks

def delete_walk(db_file: str, walk_id: int) -> List[str]:
    """Delete a walk and associated images. Returns image file paths."""
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute("SELECT filename FROM generated_images WHERE walk_id = ?", (walk_id,))
    image_paths = [row[0] for row in c.fetchall()]
    c.execute("DELETE FROM generated_images WHERE walk_id = ?", (walk_id,))
    c.execute("DELETE FROM walks WHERE id = ?", (walk_id,))
    conn.commit()
    conn.close()
    return image_paths

def archive_walk(db_file: str, archive_file: str, walk_id: int, note: str = "") -> bool:
    """Move a walk from main DB to archive DB."""
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute(
        "SELECT id, name, type, num_steps, vectors_blob, model_pkl, step_rate, timestamp, note FROM walks WHERE id = ?",
        (walk_id,),
    )
    row = c.fetchone()
    if not row:
        conn.close()
        return False
    arch_conn = sqlite3.connect(archive_file)
    arch_c = arch_conn.cursor()
    existing_note = row[-1]
    arch_c.execute(
        "INSERT INTO walks (id, name, type, num_steps, vectors_blob, model_pkl, step_rate, timestamp, note) VALUES (?,?,?,?,?,?,?,?,?)",
        row[:-1] + ((note or existing_note),),
    )
    arch_conn.commit()
    arch_conn.close()
    c.execute("DELETE FROM generated_images WHERE walk_id = ?", (walk_id,))
    c.execute("DELETE FROM walks WHERE id = ?", (walk_id,))
    conn.commit()
    conn.close()
    return True

def fetch_archived_walks(archive_file: str) -> List[Tuple]:
    conn = sqlite3.connect(archive_file)
    c = conn.cursor()
    c.execute("SELECT id, name, type, num_steps, step_rate, note FROM walks ORDER BY id DESC")
    walks = c.fetchall()
    conn.close()
    return walks

def restore_walk(archive_file: str, db_file: str, archived_id: int) -> Optional[int]:
    """Copy a walk from the archive back to the main database.

    The walk remains in the archive after restoration.
    """
    arch_conn = sqlite3.connect(archive_file)
    arch_c = arch_conn.cursor()
    arch_c.execute(
        "SELECT id, name, type, num_steps, vectors_blob, model_pkl, step_rate, timestamp, note FROM walks WHERE id = ?",
        (archived_id,),
    )
    row = arch_c.fetchone()
    arch_conn.close()
    if not row:
        return None

    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    # Preserve original ID if it's unused; otherwise let SQLite assign a new one.
    c.execute("SELECT 1 FROM walks WHERE id = ?", (row[0],))
    exists = c.fetchone() is not None
    if exists:
        c.execute(
            "INSERT INTO walks (name, type, num_steps, vectors_blob, model_pkl, step_rate, timestamp, note) VALUES (?,?,?,?,?,?,?,?)",
            row[1:],
        )
        new_id = c.lastrowid
    else:
        c.execute(
            "INSERT INTO walks (id, name, type, num_steps, vectors_blob, model_pkl, step_rate, timestamp, note) VALUES (?,?,?,?,?,?,?,?,?)",
            row,
        )
        new_id = row[0]
    conn.commit()
    conn.close()
    return new_id


def delete_archived_walk(archive_file: str, walk_id: int) -> None:
    """Delete a walk from the archive database."""
    conn = sqlite3.connect(archive_file)
    c = conn.cursor()
    c.execute("DELETE FROM walks WHERE id = ?", (walk_id,))
    conn.commit()
    conn.close()


def existing_step_indices(db_file: str, walk_id: int) -> Set[int]:
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute("SELECT step_index FROM generated_images WHERE walk_id = ?", (walk_id,))
    existing = {row[0] for row in c.fetchall()}
    conn.close()
    return existing


# ----------------------------------------------------------------------------
# Image CRUD
# ----------------------------------------------------------------------------

def add_image_record(db_file: str, walk_id: int, step_index: int, filename: str, latent: np.ndarray) -> None:
    latent = np.asarray(latent, dtype=np.float32)
    latent_blob = latent.tobytes()
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute(
        "INSERT OR IGNORE INTO generated_images (walk_id, step_index, filename, latent_blob) VALUES (?,?,?,?)",
        (walk_id, step_index, filename, latent_blob),
    )
    c.execute(
        "UPDATE generated_images SET latent_blob = ? WHERE walk_id = ? AND step_index = ? AND filename = ?",
        (latent_blob, walk_id, step_index, filename),
    )
    conn.commit()
    conn.close()

def set_image_like(db_file: str, image_id: int, liked: bool) -> None:
    """Mark an image as liked or unliked."""
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute(
        "UPDATE generated_images SET liked = ? WHERE id = ?",
        (1 if liked else 0, image_id),
    )
    conn.commit()
    conn.close()


def fetch_images(db_file: str, liked_only: bool = False) -> List[Tuple[int, int, str, int]]:
    """Fetch image records, optionally filtering by liked status."""
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    query = "SELECT id, walk_id, filename, liked FROM generated_images"
    if liked_only:
        query += " WHERE liked = 1"
    c.execute(query)
    rows = c.fetchall()
    conn.close()
    return rows


def fetch_all_images(db_file: str) -> List[Tuple[int, int, str, int]]:
    """Backwards-compatible wrapper for fetching all images."""
    return fetch_images(db_file)

def get_vector_by_image_id(db_file: str, image_id: int) -> Optional[np.ndarray]:
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute("SELECT latent_blob FROM generated_images WHERE id = ?", (image_id,))
    row = c.fetchone()
    conn.close()
    if not row or row[0] is None:
        return None
    return np.frombuffer(row[0], dtype=np.float32).copy()
