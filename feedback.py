"""
feedback.py
mmonfar. // Semantic M&M Failure Navigation Engine

A local store for the judgements a human makes about the map, so the tool gets
better at *this hospital's* taxonomy every time someone uses it.

What this is, and what it deliberately is not
---------------------------------------------
It is **not** self-training. A model that learns from its own output entrenches
whatever it already got wrong, and in patient safety that is how a blind spot
becomes policy. Nothing here ever feeds a model's own guess back to it as truth.

It is a record of **human decisions**:

  * a group was renamed         -> that name is authoritative from then on
  * a case was reassigned       -> that case belongs where the clinician says
  * two cases are/aren't alike  -> a constraint the clustering should respect

Which is worth having because the naming is demonstrably not solvable without a
human. Measured on the demo register: TF-IDF names describe 12–29% of their
group; a 0.5B local model gets roughly one group in three right; a 1.5B model is
no better. The reliable source of a good group name is the person who chairs the
meeting. This is where that gets kept.

The PHI position
----------------
**No case text is stored.** A case is keyed by a SHA-256 of its normalised
narrative, truncated to 16 hex characters. That is enough to recognise the same
case in next quarter's export, and it cannot be read back into a narrative. The
database therefore holds opinions about cases, not cases — which keeps the
project's rule that nothing derived from a register is ever committed, and makes
the file itself far less sensitive than the register it came from.

`feedback.db` is git-ignored regardless, because "less sensitive" is not "safe".

How a name survives a re-run
----------------------------
Cluster ids are meaningless across runs: add ten cases and every id moves. So a
name is stored against two things:

  1. a **fingerprint** — the hash of the sorted case keys in the group. An exact
     match means the identical group re-formed, and the name is restored.
  2. a **centroid** — so when the group has merely shifted (a few new cases, one
     moved out) the name can still be matched by cosine similarity, above
     CARRY_OVER_MIN. That restoration is marked `carried_over` with its
     similarity, because it is an inference rather than a confirmed fact, and
     the interface should be able to say so.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "feedback.db"

# Below this cosine similarity, a stored name is not carried onto a new group.
# High on purpose: wrongly re-applying last quarter's name to a group that has
# drifted into something else is a worse failure than asking again.
CARRY_OVER_MIN = 0.92

SCHEMA = """
CREATE TABLE IF NOT EXISTS group_names (
    fingerprint TEXT PRIMARY KEY,      -- hash of the sorted member case keys
    name        TEXT NOT NULL,
    centroid    TEXT NOT NULL,         -- json list[float], for fuzzy re-matching
    n_cases     INTEGER NOT NULL,
    author      TEXT,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS case_labels (
    case_key   TEXT NOT NULL,          -- sha256 of the normalised narrative
    label      TEXT NOT NULL,          -- the house taxonomy name
    author     TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (case_key, label)
);

CREATE TABLE IF NOT EXISTS links (
    case_a     TEXT NOT NULL,
    case_b     TEXT NOT NULL,
    kind       TEXT NOT NULL CHECK (kind IN ('same', 'different')),
    author     TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (case_a, case_b)
);

-- Append-only. Governance work has to be auditable: who changed what, when.
CREATE TABLE IF NOT EXISTS events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    kind       TEXT NOT NULL,
    detail     TEXT NOT NULL,
    author     TEXT,
    created_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def case_key(summary: str) -> str:
    """A stable, non-reversible id for a case narrative.

    Normalised first — collapsed whitespace, lowercased — so trivial re-typing
    between exports does not create a second identity for the same case.
    """
    normalised = re.sub(r"\s+", " ", str(summary)).strip().lower()
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:16]


def fingerprint(keys) -> str:
    """Identity of a *group*: the hash of its sorted member keys."""
    joined = "|".join(sorted(keys))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def connect(path: Path | None = None) -> sqlite3.Connection:
    # Resolved at call time, not bound as a default argument: a default is
    # evaluated once at import, so reassigning feedback.DB_PATH afterwards —
    # which tests and any host application will do — was silently ignored and
    # every read went to the wrong file.
    conn = sqlite3.connect(path or DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def log(conn, kind: str, detail: dict, author: str | None = None) -> None:
    conn.execute(
        "INSERT INTO events (kind, detail, author, created_at) VALUES (?,?,?,?)",
        (kind, json.dumps(detail, sort_keys=True), author, _now()),
    )
    conn.commit()


# --------------------------------------------------------------------------------------
# Names
# --------------------------------------------------------------------------------------

def save_name(conn, keys, name: str, centroid, author: str | None = None) -> str:
    """Record that a human named this group. Returns the fingerprint."""
    fp = fingerprint(keys)
    conn.execute(
        "INSERT INTO group_names (fingerprint, name, centroid, n_cases, author, created_at)"
        " VALUES (?,?,?,?,?,?)"
        " ON CONFLICT(fingerprint) DO UPDATE SET"
        "   name=excluded.name, centroid=excluded.centroid,"
        "   author=excluded.author, created_at=excluded.created_at",
        (fp, name.strip(), json.dumps([round(float(v), 6) for v in centroid]),
         len(list(keys)), author, _now()),
    )
    conn.commit()
    log(conn, "name", {"fingerprint": fp, "name": name.strip()}, author)
    return fp


def restore_names(conn, groups) -> list[dict]:
    """Re-apply human names to this run's groups.

    `groups` is [{"keys": [...], "centroid": [...]}, ...] in cluster-id order.
    Returns one record per group: the name, and how it was matched.

    Exact fingerprint first, then centroid similarity. A carried-over name is
    reported as such rather than presented as though the group was re-confirmed.
    """
    import numpy as np

    stored = conn.execute(
        "SELECT fingerprint, name, centroid FROM group_names"
    ).fetchall()
    if not stored:
        return [{"name": None, "match": None, "similarity": None} for _ in groups]

    by_fp = {r["fingerprint"]: r["name"] for r in stored}
    vectors, names = [], []
    for r in stored:
        vec = np.asarray(json.loads(r["centroid"]), dtype=float)
        norm = np.linalg.norm(vec)
        if norm:
            vectors.append(vec / norm)
            names.append(r["name"])

    bank = np.stack(vectors) if vectors else None
    out = []
    taken: set[str] = set()

    for g in groups:
        fp = fingerprint(g["keys"])
        if fp in by_fp and by_fp[fp] not in taken:
            taken.add(by_fp[fp])
            out.append({"name": by_fp[fp], "match": "exact", "similarity": 1.0})
            continue

        if bank is None:
            out.append({"name": None, "match": None, "similarity": None})
            continue

        vec = np.asarray(g["centroid"], dtype=float)
        norm = np.linalg.norm(vec)
        if not norm:
            out.append({"name": None, "match": None, "similarity": None})
            continue

        sims = bank @ (vec / norm)
        order = np.argsort(-sims)
        best = None
        for i in order:
            # One stored name cannot be carried onto two groups at once.
            if names[i] in taken:
                continue
            best = (float(sims[i]), names[i])
            break

        if best and best[0] >= CARRY_OVER_MIN:
            taken.add(best[1])
            out.append({"name": best[1], "match": "carried_over",
                        "similarity": round(best[0], 4)})
        else:
            out.append({"name": None, "match": None,
                        "similarity": round(best[0], 4) if best else None})

    return out


# --------------------------------------------------------------------------------------
# Case-level judgements — the training set for a future house classifier
# --------------------------------------------------------------------------------------

def label_case(conn, summary: str, label: str, author: str | None = None) -> str:
    key = case_key(summary)
    conn.execute(
        "INSERT OR REPLACE INTO case_labels (case_key, label, author, created_at)"
        " VALUES (?,?,?,?)",
        (key, label.strip(), author, _now()),
    )
    conn.commit()
    log(conn, "label", {"case": key, "label": label.strip()}, author)
    return key


def link_cases(conn, a: str, b: str, kind: str, author: str | None = None) -> None:
    """Record that two cases are, or are not, the same failure.

    Stored as an ordered pair so the same judgement cannot be entered twice in
    mirror image and then be counted as two votes.
    """
    if kind not in ("same", "different"):
        raise ValueError("kind must be 'same' or 'different'")
    ka, kb = sorted((case_key(a), case_key(b)))
    conn.execute(
        "INSERT OR REPLACE INTO links (case_a, case_b, kind, author, created_at)"
        " VALUES (?,?,?,?,?)",
        (ka, kb, kind, author, _now()),
    )
    conn.commit()
    log(conn, "link", {"a": ka, "b": kb, "kind": kind}, author)


def training_set(conn, min_per_label: int = 5) -> dict[str, list[str]]:
    """Human-labelled cases, grouped by label, once a label has enough examples.

    This is the payoff: when a hospital has labelled enough of its own cases,
    the next register can be *classified* into its agreed taxonomy instead of
    re-clustered from scratch — and the taxonomy is theirs, not the textbook's.
    Below the threshold a label is not returned, because a classifier trained on
    two examples is worse than the clustering it would replace.
    """
    rows = conn.execute("SELECT case_key, label FROM case_labels").fetchall()
    grouped: dict[str, list[str]] = {}
    for r in rows:
        grouped.setdefault(r["label"], []).append(r["case_key"])
    return {k: v for k, v in grouped.items() if len(v) >= min_per_label}


def summary(conn) -> dict:
    """One-line state of the store, for the interface and the CLI."""
    def count(table: str) -> int:
        return conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]

    return {
        "named_groups": count("group_names"),
        "labelled_cases": count("case_labels"),
        "links": count("links"),
        "events": count("events"),
        "ready_labels": sorted(training_set(conn)),
    }
