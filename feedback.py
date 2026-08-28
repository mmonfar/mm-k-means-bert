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

-- The learned taxonomy: one running centroid per label, never one per case.
-- A mean over many cases is what makes the next register classifiable without
-- keeping the last one. Storing per-case vectors would work too and is not done
-- on purpose: an average over n cases is far less like any individual case.
CREATE TABLE IF NOT EXISTS label_centroids (
    label      TEXT PRIMARY KEY,
    vector_sum TEXT NOT NULL,          -- json list[float], summed not averaged
    n          INTEGER NOT NULL,       -- how many cases went into it
    updated_at TEXT NOT NULL
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


# --------------------------------------------------------------------------------------
# The refinement loop
# --------------------------------------------------------------------------------------
#
# This is the part that makes the tool get better with use, and it needs no
# language model. The embeddings are already computed for the map; a label is
# just the mean of the embeddings of the cases a human put under it. Classifying
# next quarter's register is then a cosine comparison — a few milliseconds, no
# download, no training run, and completely inspectable.
#
# It improves monotonically: every case a human files adds to a centroid, so the
# taxonomy sharpens as the hospital uses it, and it is *their* taxonomy rather
# than the textbook's.

# A label needs this many examples before it is allowed to classify anything.
# Below it the centroid is really just one or two cases wearing a category name.
MIN_EXAMPLES = 5

# A prediction must beat the runner-up by this much. Two labels 0.01 apart is a
# coin toss, and a coin toss presented as a classification is worse than a blank.
MIN_MARGIN = 0.02

# And it must be at least this similar at all, or the case is simply new.
MIN_CONFIDENCE = 0.25


def learn(conn, label: str, vectors, author: str | None = None) -> int:
    """Fold cases into a label's running centroid. Returns the label's new count."""
    import numpy as np

    vectors = np.asarray(vectors, dtype=float)
    if vectors.ndim == 1:
        vectors = vectors[None, :]
    if not len(vectors):
        return 0

    label = label.strip()
    row = conn.execute(
        "SELECT vector_sum, n FROM label_centroids WHERE label = ?", (label,)
    ).fetchone()

    total = np.asarray(json.loads(row["vector_sum"]), dtype=float) if row else 0.0
    count = row["n"] if row else 0
    total = total + vectors.sum(axis=0)
    count += len(vectors)

    conn.execute(
        "INSERT INTO label_centroids (label, vector_sum, n, updated_at) VALUES (?,?,?,?)"
        " ON CONFLICT(label) DO UPDATE SET"
        "   vector_sum=excluded.vector_sum, n=excluded.n, updated_at=excluded.updated_at",
        (label, json.dumps([round(float(v), 6) for v in np.atleast_1d(total)]),
         count, _now()),
    )
    conn.commit()
    log(conn, "learn", {"label": label, "added": len(vectors), "total": count}, author)
    return count


def taxonomy(conn) -> dict[str, int]:
    """Every learned label and how many cases stand behind it."""
    return {r["label"]: r["n"]
            for r in conn.execute("SELECT label, n FROM label_centroids").fetchall()}


def _usable_centroids(conn):
    import numpy as np

    rows = conn.execute(
        "SELECT label, vector_sum, n FROM label_centroids WHERE n >= ?",
        (MIN_EXAMPLES,),
    ).fetchall()
    labels, vectors = [], []
    for r in rows:
        vec = np.asarray(json.loads(r["vector_sum"]), dtype=float) / r["n"]
        norm = np.linalg.norm(vec)
        if norm:
            labels.append(r["label"])
            vectors.append(vec / norm)
    return labels, (np.stack(vectors) if vectors else None)


def classify(conn, vectors) -> list[dict]:
    """Assign each case to the house taxonomy, or to nothing.

    Returns one record per case: {"label", "confidence", "margin"}. `label` is
    None whenever the tool is not entitled to an opinion — too few examples, too
    little similarity, or two labels too close to separate. Refusing to answer is
    a feature: a governance pack full of confident mislabels is worse than one
    with blanks a human fills in.
    """
    import numpy as np

    labels, bank = _usable_centroids(conn)
    vectors = np.asarray(vectors, dtype=float)
    if bank is None or not len(vectors):
        return [{"label": None, "confidence": None, "margin": None}
                for _ in range(len(vectors))]

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    sims = (vectors / norms) @ bank.T

    out = []
    for row in sims:
        order = np.argsort(-row)
        best = float(row[order[0]])
        runner = float(row[order[1]]) if len(order) > 1 else -1.0
        margin = best - runner
        if best < MIN_CONFIDENCE or margin < MIN_MARGIN:
            out.append({"label": None, "confidence": round(best, 4),
                        "margin": round(margin, 4)})
        else:
            out.append({"label": labels[order[0]], "confidence": round(best, 4),
                        "margin": round(margin, 4)})
    return out


def evaluate(conn, vectors, truth: list[str]) -> dict:
    """How well the learned taxonomy reproduces known human labels.

    Honest about its own limits: the same cases trained these centroids, so this
    is a floor on error, not an estimate of future accuracy. It answers "has this
    taxonomy learned anything coherent yet", which is the question that decides
    whether to trust it at all.
    """
    predictions = classify(conn, vectors)
    answered = [(p["label"], t) for p, t in zip(predictions, truth) if p["label"]]
    if not answered:
        return {"answered": 0, "of": len(truth), "agreement": None}
    hits = sum(1 for pred, actual in answered if pred == actual)
    return {
        "answered": len(answered),
        "of": len(truth),
        "agreement": round(hits / len(answered), 3),
    }


def summary(conn) -> dict:
    """One-line state of the store, for the interface and the CLI."""
    def count(table: str) -> int:
        return conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]

    learned = taxonomy(conn)
    return {
        "named_groups": count("group_names"),
        "labelled_cases": count("case_labels"),
        "links": count("links"),
        "events": count("events"),
        "taxonomy": learned,
        "ready_labels": sorted(k for k, n in learned.items() if n >= MIN_EXAMPLES),
    }
