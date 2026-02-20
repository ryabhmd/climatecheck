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
            SELECT S_ref, best_gold_idx, gold_label, s_proxy, s_ev2r
            FROM ev2r_cache WHERE pair_key=?
        """, (pair_key,))
        row = cursor.fetchone()

        if row:
            return {
                "S_ref": row[0],
                "best_gold_idx": row[1],
                "gold_label": row[2],
                "s_proxy": row[3],
                "s_ev2r": row[4],
            }
        return None

    def insert(
        self,
        pair_key: str,
        claim: str,
        abstract: str,
        S_ref: float,
        best_gold_idx: int,
        gold_label: str,
        s_proxy: float,
        s_ev2r: float,
    ):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO ev2r_cache
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pair_key,
            claim,
            abstract,
            S_ref,
            best_gold_idx,
            gold_label,
            s_proxy,
            s_ev2r
        ))
        self.conn.commit()  # immediate commit (crash-safe)
