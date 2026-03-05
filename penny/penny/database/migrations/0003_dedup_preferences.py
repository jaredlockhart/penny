"""Deduplicate existing preferences using text match and embedding similarity."""

from __future__ import annotations

import sqlite3
import struct


def up(conn: sqlite3.Connection) -> None:
    """Remove duplicate preferences by text match and embedding similarity."""
    tables = [
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='preference'"
        ).fetchall()
    ]
    if not tables:
        conn.commit()
        return
    _dedup_by_text(conn)
    _dedup_by_embedding(conn)
    conn.commit()


def _dedup_by_text(conn: sqlite3.Connection) -> None:
    """Remove preferences with duplicate text (case-insensitive), keeping oldest."""
    users = conn.execute("SELECT DISTINCT user FROM preference").fetchall()

    for (user,) in users:
        rows = conn.execute(
            "SELECT id, content FROM preference WHERE user = ? ORDER BY created_at ASC",
            (user,),
        ).fetchall()

        seen: set[str] = set()
        delete_ids: list[int] = []

        for row_id, content in rows:
            key = content.strip().lower()
            if key in seen:
                delete_ids.append(row_id)
            else:
                seen.add(key)

        if delete_ids:
            placeholders = ",".join("?" for _ in delete_ids)
            conn.execute(
                f"DELETE FROM preference WHERE id IN ({placeholders})",  # noqa: S608
                delete_ids,
            )


def _dedup_by_embedding(conn: sqlite3.Connection) -> None:
    """Remove preferences with similar embeddings (cosine >= 0.85), keeping oldest."""
    users = conn.execute("SELECT DISTINCT user FROM preference").fetchall()

    for (user,) in users:
        rows = conn.execute(
            "SELECT id, embedding FROM preference "
            "WHERE user = ? AND embedding IS NOT NULL "
            "ORDER BY created_at ASC",
            (user,),
        ).fetchall()

        keep_vecs: list[tuple[int, list[float]]] = []
        delete_ids: list[int] = []

        for row_id, embedding_bytes in rows:
            vec = _deserialize(embedding_bytes)
            is_dup = any(_cosine_similarity(vec, kept_vec) >= 0.85 for _, kept_vec in keep_vecs)
            if is_dup:
                delete_ids.append(row_id)
            else:
                keep_vecs.append((row_id, vec))

        if delete_ids:
            placeholders = ",".join("?" for _ in delete_ids)
            conn.execute(
                f"DELETE FROM preference WHERE id IN ({placeholders})",  # noqa: S608
                delete_ids,
            )


def _deserialize(data: bytes) -> list[float]:
    """Deserialize a float32 embedding vector from bytes."""
    count = len(data) // 4
    return list(struct.unpack(f"{count}f", data))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
