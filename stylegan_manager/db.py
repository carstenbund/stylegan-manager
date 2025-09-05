import sqlite3
from typing import List, Optional, Tuple, Set

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
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    # Ensure step_rate column exists for older DBs
    c.execute("PRAGMA table_info(walks)")
    existing_cols = [row[1] for row in c.fetchall()]
    if "step_rate" not in existing_cols:
        c.execute("ALTER TABLE walks ADD COLUMN step_rate INTEGER NOT NULL DEFAULT 60")

    c.execute(
        """
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
        """
    )
    c.execute("CREATE INDEX IF NOT EXISTS idx_genimg_walk ON generated_images(walk_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_genimg_filename ON generated_images(filename)")
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
        "SELECT id, name, type, num_steps, vectors_blob, model_pkl, step_rate, timestamp FROM walks WHERE id = ?",
        (walk_id,),
    )
    row = c.fetchone()
    if not row:
        conn.close()
        return False
    arch_conn = sqlite3.connect(archive_file)
    arch_c = arch_conn.cursor()
    arch_c.execute(
        "INSERT INTO walks (id, name, type, num_steps, vectors_blob, model_pkl, step_rate, timestamp, note) VALUES (?,?,?,?,?,?,?,?,?)",
        row + (note,),
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
    arch_conn = sqlite3.connect(archive_file)
    arch_c = arch_conn.cursor()
    arch_c.execute(
        "SELECT id, name, type, num_steps, vectors_blob, model_pkl, step_rate, timestamp FROM walks WHERE id = ?",
        (archived_id,),
    )
    row = arch_c.fetchone()
    if not row:
        arch_conn.close()
        return None
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute(
        "INSERT INTO walks (id, name, type, num_steps, vectors_blob, model_pkl, step_rate, timestamp) VALUES (?,?,?,?,?,?,?,?)",
        row,
    )
    conn.commit()
    conn.close()
    arch_c.execute("DELETE FROM walks WHERE id = ?", (archived_id,))
    arch_conn.commit()
    arch_conn.close()
    return row[0]


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

def fetch_all_images(db_file: str) -> List[Tuple[int, int, str]]:
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute("SELECT id, walk_id, filename FROM generated_images")
    rows = c.fetchall()
    conn.close()
    return rows

def get_vector_by_image_id(db_file: str, image_id: int) -> Optional[np.ndarray]:
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute("SELECT latent_blob FROM generated_images WHERE id = ?", (image_id,))
    row = c.fetchone()
    conn.close()
    if not row or row[0] is None:
        return None
    return np.frombuffer(row[0], dtype=np.float32).copy()
