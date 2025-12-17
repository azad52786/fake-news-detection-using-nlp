import os
import json
import sqlite3
from typing import List, Dict, Any

from config import HISTORY_DB_PATH


def _get_conn():
    os.makedirs(os.path.dirname(HISTORY_DB_PATH), exist_ok=True)
    return sqlite3.connect(HISTORY_DB_PATH, check_same_thread=False)


def init_db() -> None:
    conn = _get_conn()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id TEXT,
                title TEXT,
                content TEXT,
                label TEXT,
                probability REAL,
                model_version TEXT,
                top_tokens TEXT,
                created_at TEXT
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def insert_prediction(record: Dict[str, Any]) -> None:
    conn = _get_conn()
    try:
        conn.execute(
            """
            INSERT INTO predictions (
                prediction_id, title, content, label, probability, model_version, top_tokens, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.get("prediction_id"),
                record.get("title"),
                record.get("content"),
                record.get("label"),
                record.get("probability"),
                record.get("model_version"),
                json.dumps(record.get("top_tokens")) if record.get("top_tokens") is not None else None,
                record.get("created_at"),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def fetch_history(limit: int = 20) -> List[Dict[str, Any]]:
    conn = _get_conn()
    try:
        cur = conn.execute(
            """
            SELECT prediction_id, title, content, label, probability, model_version, top_tokens, created_at
            FROM predictions
            ORDER BY datetime(created_at) DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cur.fetchall()
        result = []
        for row in rows:
            top_tokens = json.loads(row[6]) if row[6] else None
            result.append(
                {
                    "prediction_id": row[0],
                    "title": row[1],
                    "content": row[2],
                    "label": row[3],
                    "probability": row[4],
                    "model_version": row[5],
                    "top_tokens": top_tokens,
                    "created_at": row[7],
                }
            )
        return result
    finally:
        conn.close()
