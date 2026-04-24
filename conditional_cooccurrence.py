#!/usr/bin/env python3
"""
Efficient conditional word co-occurrence probabilities over sentence-level data.

Implements:
    P(a | b)     = count(a,b) / count(b)
    P(a | b, c)  = count(a,b,c) / count(b,c)

Architecture:
    - Inverted index word -> Roaring bitmap(sentence_ids)
    - Persisted unigram and pair counts in SQLite
    - Triple probabilities computed on demand by bitmap intersections
    - LRU cache for pairwise conditioning intersections
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
from collections import Counter, defaultdict
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterator, Optional, Sequence, Tuple

try:
    from pyroaring import BitMap
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'pyroaring'. Install with: pip install pyroaring"
    ) from exc

WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
SCHEMA_VERSION = 1


def tokenize_unique(sentence: str) -> Tuple[str, ...]:
    words = {w.lower() for w in WORD_RE.findall(sentence)}
    if not words:
        return ()
    return tuple(sorted(words))


def canonical_pair(a: str, b: str) -> Tuple[str, str]:
    if a <= b:
        return a, b
    return b, a


def batched_items(counter: Counter, chunk_size: int) -> Iterator[Sequence[Tuple]]:
    items = list(counter.items())
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


class CooccurrenceStore:
    def __init__(self, db_path: Path, cache_size: int = 100_000) -> None:
        self.db_path = db_path
        self.cache_size = cache_size
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode = WAL;")
        self.conn.execute("PRAGMA synchronous = NORMAL;")
        self.conn.execute("PRAGMA temp_store = MEMORY;")
        self.conn.execute("PRAGMA mmap_size = 30000000000;")
        self._posting_cache: Dict[str, BitMap] = {}

        @lru_cache(maxsize=cache_size)
        def _pair_intersection_cached(w1: str, w2: str) -> bytes:
            bm = self.get_posting(w1) & self.get_posting(w2)
            return bm.serialize()

        self._pair_intersection_cached = _pair_intersection_cached

    def close(self) -> None:
        self.conn.close()

    def init_schema(self, overwrite: bool = False) -> None:
        cur = self.conn.cursor()
        if overwrite:
            cur.executescript(
                """
                DROP TABLE IF EXISTS meta;
                DROP TABLE IF EXISTS unigram_counts;
                DROP TABLE IF EXISTS pair_counts;
                DROP TABLE IF EXISTS postings;
                """
            )
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS unigram_counts (
                word TEXT PRIMARY KEY,
                count INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS pair_counts (
                w1 TEXT NOT NULL,
                w2 TEXT NOT NULL,
                count INTEGER NOT NULL,
                PRIMARY KEY (w1, w2)
            );

            CREATE TABLE IF NOT EXISTS postings (
                word TEXT PRIMARY KEY,
                doc_freq INTEGER NOT NULL,
                bitmap BLOB NOT NULL
            );
            """
        )
        self.conn.commit()

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            """
            INSERT INTO meta (key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )

    def get_meta(self, key: str) -> Optional[str]:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def upsert_unigrams(self, counts: Counter) -> None:
        if not counts:
            return
        rows = [(w, int(c)) for w, c in counts.items()]
        self.conn.executemany(
            """
            INSERT INTO unigram_counts (word, count) VALUES (?, ?)
            ON CONFLICT(word) DO UPDATE SET count = count + excluded.count
            """,
            rows,
        )

    def upsert_pairs(self, counts: Counter, chunk_size: int = 100_000) -> None:
        if not counts:
            return
        for chunk in batched_items(counts, chunk_size):
            rows = [(w1, w2, int(c)) for (w1, w2), c in chunk]
            self.conn.executemany(
                """
                INSERT INTO pair_counts (w1, w2, count) VALUES (?, ?, ?)
                ON CONFLICT(w1, w2) DO UPDATE SET count = count + excluded.count
                """,
                rows,
            )

    def write_postings(self, postings: Dict[str, BitMap], chunk_size: int = 50_000) -> None:
        words = list(postings.keys())
        for i in range(0, len(words), chunk_size):
            rows = []
            for w in words[i : i + chunk_size]:
                bm = postings[w]
                rows.append((w, len(bm), bm.serialize()))
            self.conn.executemany(
                """
                INSERT INTO postings (word, doc_freq, bitmap) VALUES (?, ?, ?)
                ON CONFLICT(word) DO UPDATE SET
                    doc_freq = excluded.doc_freq,
                    bitmap = excluded.bitmap
                """,
                rows,
            )
            self.conn.commit()

    def get_unigram_count(self, word: str) -> int:
        row = self.conn.execute(
            "SELECT count FROM unigram_counts WHERE word = ?", (word,)
        ).fetchone()
        return int(row[0]) if row else 0

    def get_pair_count(self, a: str, b: str) -> int:
        w1, w2 = canonical_pair(a, b)
        row = self.conn.execute(
            "SELECT count FROM pair_counts WHERE w1 = ? AND w2 = ?", (w1, w2)
        ).fetchone()
        return int(row[0]) if row else 0

    def get_posting(self, word: str) -> BitMap:
        if word in self._posting_cache:
            return self._posting_cache[word]
        row = self.conn.execute(
            "SELECT bitmap FROM postings WHERE word = ?", (word,)
        ).fetchone()
        if not row:
            bm = BitMap()
            self._posting_cache[word] = bm
            return bm
        bm = BitMap.deserialize(row[0])
        self._posting_cache[word] = bm
        return bm

    def get_pair_intersection(self, b: str, c: str) -> BitMap:
        return BitMap.deserialize(self._pair_intersection_cached(*canonical_pair(b, c)))

    def prob_a_given_b(self, a: str, b: str) -> float:
        a = a.lower()
        b = b.lower()
        denom = self.get_unigram_count(b)
        if denom == 0:
            return 0.0
        numer = len(self.get_pair_intersection(a, b))
        return numer / denom

    def prob_a_given_b_c(self, a: str, b: str, c: str) -> float:
        a = a.lower()
        b = b.lower()
        c = c.lower()
        bc = self.get_pair_intersection(b, c)
        denom = len(bc)
        if denom == 0:
            return 0.0
        numer = len(bc & self.get_posting(a))
        return numer / denom

    def top_words(self, limit: int = 20) -> Sequence[Tuple[str, int]]:
        rows = self.conn.execute(
            "SELECT word, count FROM unigram_counts ORDER BY count DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [(str(r[0]), int(r[1])) for r in rows]


def build_index(
    sentences_path: Path,
    db_path: Path,
    max_sentences: Optional[int] = None,
    flush_every: int = 50_000,
    overwrite: bool = False,
    precompute_pairs: bool = True,
) -> None:
    if not sentences_path.exists():
        raise FileNotFoundError(f"Sentences file not found: {sentences_path}")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    store = CooccurrenceStore(db_path)
    store.init_schema(overwrite=overwrite)

    unigram_batch: Counter = Counter()
    pair_batch: Counter = Counter()
    postings: Dict[str, BitMap] = defaultdict(BitMap)
    start = time.time()
    n_sentences = 0

    with sentences_path.open("r", encoding="utf-8") as f:
        for sid, line in enumerate(f):
            if max_sentences is not None and sid >= max_sentences:
                break
            words = tokenize_unique(line)
            if not words:
                continue

            n_sentences += 1
            for w in words:
                unigram_batch[w] += 1
                postings[w].add(sid)
            if precompute_pairs:
                for w1, w2 in combinations(words, 2):
                    pair_batch[(w1, w2)] += 1

            if n_sentences % flush_every == 0:
                store.upsert_unigrams(unigram_batch)
                if precompute_pairs:
                    store.upsert_pairs(pair_batch)
                store.conn.commit()
                unigram_batch.clear()
                if precompute_pairs:
                    pair_batch.clear()
                elapsed = time.time() - start
                print(
                    f"[build] processed={n_sentences:,} unique_words={len(postings):,} elapsed={elapsed:,.1f}s"
                )

    # Final flush.
    store.upsert_unigrams(unigram_batch)
    if precompute_pairs:
        store.upsert_pairs(pair_batch)
    store.conn.commit()
    store.write_postings(postings)

    meta = {
        "schema_version": SCHEMA_VERSION,
        "sentences_path": str(sentences_path.resolve()),
        "build_time_utc_epoch": int(time.time()),
        "sentence_count": n_sentences,
        "vocab_size": len(postings),
        "pair_counts_precomputed": precompute_pairs,
    }
    for k, v in meta.items():
        store.set_meta(k, json.dumps(v))
    store.conn.commit()
    store.close()

    print(f"[build] complete sentences={n_sentences:,} vocab={len(postings):,}")
    print(f"[build] database persisted at: {db_path}")


def cmd_query(args: argparse.Namespace) -> None:
    store = CooccurrenceStore(Path(args.db))
    if args.c is None:
        p = store.prob_a_given_b(args.a, args.b)
        pair = len(store.get_pair_intersection(args.a.lower(), args.b.lower()))
        denom = store.get_unigram_count(args.b.lower())
        print(
            json.dumps(
                {
                    "query": f"P({args.a}|{args.b})",
                    "numerator_count": pair,
                    "denominator_count": denom,
                    "probability": p,
                },
                ensure_ascii=False,
            )
        )
    else:
        p = store.prob_a_given_b_c(args.a, args.b, args.c)
        bc = store.get_pair_intersection(args.b.lower(), args.c.lower())
        numer = len(bc & store.get_posting(args.a.lower()))
        denom = len(bc)
        print(
            json.dumps(
                {
                    "query": f"P({args.a}|{args.b},{args.c})",
                    "numerator_count": numer,
                    "denominator_count": denom,
                    "probability": p,
                },
                ensure_ascii=False,
            )
        )
    store.close()


def cmd_test_queries(args: argparse.Namespace) -> None:
    store = CooccurrenceStore(Path(args.db))
    top = store.top_words(limit=max(10, args.top_k))
    if len(top) < 3:
        raise RuntimeError("Not enough words in index to run test queries.")

    words = [w for w, _ in top]
    print("[tests] top words:", ", ".join(words[: args.top_k]))
    print("[tests] running pairwise conditional probability checks:")

    pair_tests = [
        (words[0], words[1]),
        (words[2], words[1]),
        (words[3], words[0]),
    ]
    for a, b in pair_tests:
        p = store.prob_a_given_b(a, b)
        print(f"  P({a}|{b}) = {p:.6f}")

    print("[tests] running triple conditional probability checks:")
    triple_tests = [
        (words[0], words[1], words[2]),
        (words[3], words[1], words[2]),
        (words[4], words[2], words[0]),
    ]
    for a, b, c in triple_tests:
        p = store.prob_a_given_b_c(a, b, c)
        print(f"  P({a}|{b},{c}) = {p:.6f}")
    store.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute conditional probabilities for sentence-level word co-occurrence."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="Build and persist inverted index + counts")
    p_build.add_argument(
        "--sentences",
        default="data/russian_sentences.txt",
        help="Path to newline-delimited sentences file",
    )
    p_build.add_argument(
        "--db",
        default="artifacts/cooccurrence.sqlite",
        help="Path to persisted SQLite artifact",
    )
    p_build.add_argument(
        "--max-sentences",
        type=int,
        default=None,
        help="Optional cap for faster experiments",
    )
    p_build.add_argument(
        "--flush-every",
        type=int,
        default=50_000,
        help="Flush count batches to SQLite every N non-empty sentences",
    )
    p_build.add_argument(
        "--overwrite",
        action="store_true",
        help="Drop and recreate existing tables before building",
    )

    p_build_no_pairs = sub.add_parser(
        "build-no-pairs",
        help="Build and persist only unigram counts + postings (no pair precompute)",
    )
    p_build_no_pairs.add_argument(
        "--sentences",
        default="data/russian_sentences.txt",
        help="Path to newline-delimited sentences file",
    )
    p_build_no_pairs.add_argument(
        "--db",
        default="artifacts/cooccurrence_no_pairs.sqlite",
        help="Path to persisted SQLite artifact",
    )
    p_build_no_pairs.add_argument(
        "--max-sentences",
        type=int,
        default=None,
        help="Optional cap for faster experiments",
    )
    p_build_no_pairs.add_argument(
        "--flush-every",
        type=int,
        default=50_000,
        help="Flush count batches to SQLite every N non-empty sentences",
    )
    p_build_no_pairs.add_argument(
        "--overwrite",
        action="store_true",
        help="Drop and recreate existing tables before building",
    )

    p_query = sub.add_parser("query", help="Run one conditional probability query")
    p_query.add_argument("--db", default="artifacts/cooccurrence.sqlite")
    p_query.add_argument("--a", required=True, help="Target word a")
    p_query.add_argument("--b", required=True, help="Conditioning word b")
    p_query.add_argument("--c", default=None, help="Optional conditioning word c")

    p_test = sub.add_parser("test-queries", help="Run a few sample queries")
    p_test.add_argument("--db", default="artifacts/cooccurrence.sqlite")
    p_test.add_argument("--top-k", type=int, default=12)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "build":
        build_index(
            sentences_path=Path(args.sentences),
            db_path=Path(args.db),
            max_sentences=args.max_sentences,
            flush_every=args.flush_every,
            overwrite=args.overwrite,
            precompute_pairs=True,
        )
        return

    if args.command == "build-no-pairs":
        build_index(
            sentences_path=Path(args.sentences),
            db_path=Path(args.db),
            max_sentences=args.max_sentences,
            flush_every=args.flush_every,
            overwrite=args.overwrite,
            precompute_pairs=False,
        )
        return

    if args.command == "query":
        cmd_query(args)
        return

    if args.command == "test-queries":
        cmd_test_queries(args)
        return

    raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
