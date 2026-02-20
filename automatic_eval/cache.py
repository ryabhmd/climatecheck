import sqlite3
import hashlib
from typing import Optional, Dict, Any


def make_pair_key(claim: str, abstract: str) -> str:
    combined = claim.strip() + "|||" + abstract.strip()
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


class Ev2RCache:

    def __init__(self, db_path: str = "ev2r_cache.db"):
        self.conn = sqlite3.connect(db_path)
        self._create_table()

    def _create_table(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ev2r_cache (
                pair_key TEXT PRIMARY KEY,
                claim TEXT,
                abstract TEXT,
                S_ref REAL,
                best_gold_idx REAL,
                gold_label TEXT,
                s_proxy REAL,
                s_ev2r REAL
            )
        """)
        self.conn.commit()

    def get(self, pair_key: str) -> Optional[Dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT reference_score, proxy_score, gold_label
            FROM ev2r_cache WHERE pair_key=?
        """, (pair_key,))
        row = cursor.fetchone()

        if row:
            return {
                "reference_score": row[0],
                "proxy_score": row[1],
                "gold_label": row[2],
            }
        return None

    def insert(
        self,
        pair_key: str,
        claim: str,
        abstract: str,
        reference_score: float,
        proxy_score: float,
        gold_label: str,
    ):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO ev2r_cache
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            pair_key,
            claim,
            abstract,
            reference_score,
            proxy_score,
            gold_label,
        ))
        self.conn.commit()  # immediate commit (crash-safe)