"""
engine.py
mmonfar. // Semantic M&M Failure Navigation Engine — Phase 3.2

Batch compiler. Reads a spreadsheet of M&M minutes, turns the free text into a semantic
map, and emits a static JSON payload that the Three.js canvas in `app/` renders without
ever calling back into Python.

    Excel/CSV -> MiniLM embeddings (384-d) -> KMeans(k=5) -> UMAP/PCA(3-d)
              -> app/data.json

Accepts .xlsx, .xlsm, .xls, .csv, .tsv and .txt. Column headers are matched against a
synonym table (case- and punctuation-insensitive), so a real hospital export usually
works with no flags; --text-col and friends override when it does not.

Usage:
    python engine.py
    python engine.py --input datix_export.csv
    python engine.py --input minutes.xlsx --text-col "What happened" --dept-col Specialty
    python engine.py --clusters 7 --projection pca

Everything runs locally on CPU. The only network access is the one-off download of the
`all-MiniLM-L6-v2` weights (~90 MB) into the HuggingFace cache on first run.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parent
APP_DIR = ROOT / "app"
DEFAULT_INPUT = ROOT / "mock_mm_minutes.xlsx"

MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384

REQUIRED_COLUMNS = ["Case_ID", "Date", "Department", "Case_Summary", "Severity_Score"]

# Presentation constants — see SPECIFICATION.md §4
WORLD_HALF_EDGE = 100.0
SEPARATION_GAIN = 1.65
MIN_CENTROID_GAP = 45.0
MAX_SEPARATION_PASSES = 3

CLUSTER_DIMS = 10  # PCA components used for clustering only — see denoise()

CLUSTER_PALETTE = ["#00B3B3", "#FF8A5B", "#9D8CFF", "#F2C14E", "#5BD1A0"]

PAYLOAD_START = "<!-- MM_PAYLOAD_START -->"
PAYLOAD_END = "<!-- MM_PAYLOAD_END -->"

# Clinical stopwords: high-frequency ward prose that carries no failure-mode signal.
CLINICAL_STOPWORDS = [
    "pt", "pts", "patient", "day", "days", "hour", "hours", "ward", "team", "case",
    "note", "noted", "documented", "given", "performed", "reviewed", "post", "pre",
    "later", "time", "new", "second", "record", "records", "unit", "units", "left",
    "right", "away", "took", "went", "come", "came", "did", "does", "also",
]


def _stopwords() -> list[str]:
    """English stopwords plus ward-prose filler that carries no failure-mode signal."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    return sorted(set(ENGLISH_STOP_WORDS) | set(CLINICAL_STOPWORDS))


# --------------------------------------------------------------------------------------
# 1. Ingest
# --------------------------------------------------------------------------------------

SPREADSHEET_SUFFIXES = {".xlsx", ".xlsm", ".xls"}
DELIMITED_SUFFIXES = {".csv", ".tsv", ".txt"}

# Header synonyms. Nobody's real M&M export uses our column names, and asking a
# governance lead to rename columns before they can try the tool is how a tool goes
# unused. Matching is done on a squashed key: lowercased, non-alphanumerics stripped.
COLUMN_SYNONYMS: dict[str, list[str]] = {
    "Case_ID": [
        "caseid", "case", "id", "ref", "reference", "casereference", "casenumber",
        "caseno", "incidentid", "incidentref", "datixref", "datixid", "number",
    ],
    "Date": [
        "date", "casedate", "incidentdate", "dateofincident", "eventdate",
        "dateofevent", "meetingdate", "reporteddate", "when",
    ],
    "Department": [
        "department", "dept", "specialty", "speciality", "service", "directorate",
        "division", "ward", "unit", "team", "location",
    ],
    "Case_Summary": [
        "casesummary", "summary", "narrative", "description", "details", "text",
        "freetext", "incidentdescription", "whathappened", "notes", "comment",
        "comments", "learning", "discussion", "body",
    ],
    "Severity_Score": [
        "severityscore", "severity", "harm", "harmscore", "harmlevel", "grade",
        "gradeofharm", "impact", "riskscore", "score", "outcome",
    ],
}

# pandas treats the literal string "None" as missing by default. In a harm column
# "None" means no harm — grade 1 — not a blank, so it is removed from the NA list.
NA_STRINGS = [
    "", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan",
    "1.#IND", "1.#QNAN", "<NA>", "N/A", "NA", "NULL", "NaN", "n/a", "nan", "null",
]

# Words people actually type into a harm column, mapped onto the 1-5 scale.
SEVERITY_WORDS = {
    "none": 1, "nil": 1, "nearmiss": 1, "near miss": 1, "no harm": 1, "noharm": 1,
    "negligible": 1, "insignificant": 1, "minimal": 1,
    "low": 2, "minor": 2, "slight": 2,
    "moderate": 3, "medium": 3, "significant": 3,
    "major": 4, "high": 4, "severe": 4, "serious": 4,
    "catastrophic": 5, "death": 5, "fatal": 5, "extreme": 5, "critical": 5,
}


def _squash(name: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def resolve_columns(
    df: pd.DataFrame, overrides: dict[str, str] | None = None
) -> dict[str, str | None]:
    """Map our canonical column names onto whatever the file actually calls them.

    Returns {canonical: actual_column_or_None}. Only `Case_Summary` is genuinely
    required — everything else degrades to a sensible default, because a department
    column is nice to have and a summary column is the entire product.
    """
    overrides = overrides or {}
    squashed = {_squash(c): c for c in df.columns}
    resolved: dict[str, str | None] = {}

    for canonical, synonyms in COLUMN_SYNONYMS.items():
        chosen = overrides.get(canonical)
        if chosen:
            if chosen not in df.columns:
                raise ValueError(
                    f"--{canonical.lower().replace('_', '-')} {chosen!r} is not a column "
                    f"in the file. Available: {list(df.columns)}"
                )
            resolved[canonical] = chosen
            continue

        hit = squashed.get(_squash(canonical))
        if hit is None:
            for syn in synonyms:
                if syn in squashed:
                    hit = squashed[syn]
                    break
        if hit is None:
            # Last resort for the one column we cannot do without: the widest text
            # column in the file is almost always the narrative.
            if canonical == "Case_Summary":
                text_cols = [
                    c for c in df.columns
                    if df[c].dtype == object
                    and df[c].astype(str).str.len().mean() > 40
                ]
                if text_cols:
                    hit = max(text_cols, key=lambda c: df[c].astype(str).str.len().mean())
        resolved[canonical] = hit

    return resolved


def read_tabular(path: Path) -> pd.DataFrame:
    """Read .xlsx/.xlsm/.xls or .csv/.tsv/.txt into a DataFrame.

    CSVs in the wild are exported from Excel on Windows and are frequently cp1252
    rather than UTF-8, and are as likely to be semicolon- as comma-delimited, so both
    are sniffed rather than assumed.
    """
    suffix = path.suffix.lower()

    if suffix in SPREADSHEET_SUFFIXES:
        return pd.read_excel(path)

    if suffix in DELIMITED_SUFFIXES:
        last: Exception | None = None
        # Sniffing first, then explicit delimiters: the sniffer misreads a
        # single-column file of prose, where punctuation outnumbers any real
        # delimiter. Falling through to a plain comma read recovers that case.
        for sep in (None, ",", ";", "\t"):
            for encoding in ("utf-8-sig", "cp1252", "latin-1"):
                try:
                    return pd.read_csv(
                        path,
                        encoding=encoding,
                        sep=sep,
                        engine="python",
                        skip_blank_lines=True,
                        keep_default_na=False,
                        na_values=NA_STRINGS,
                    )
                except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
                    last = exc
        raise ValueError(f"Could not parse {path.name}: {last}")

    raise ValueError(
        f"Unsupported file type {suffix!r}. Use one of: "
        + ", ".join(sorted(SPREADSHEET_SUFFIXES | DELIMITED_SUFFIXES))
    )


def coerce_severity(series: pd.Series) -> pd.Series:
    """Coerce a harm column to ints 1-5, accepting numbers or words."""
    numeric = pd.to_numeric(series, errors="coerce")
    # Always try the word map. Dtype is not a reliable guide here: pandas hands back
    # StringDtype, object or category depending on the reader and the pandas version.
    words = series.astype(str).str.strip().str.lower().map(SEVERITY_WORDS)
    return (
        numeric.astype("float64")
        .fillna(words.astype("float64"))
        .fillna(3)
        .clip(1, 5)
        .astype(int)
    )


def load_minutes(path: Path, overrides: dict[str, str] | None = None) -> pd.DataFrame:
    """Read a spreadsheet or CSV and normalise it onto the column contract."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python data_generator.py` first, or pass --input."
        )

    raw = read_tabular(path)
    if raw.empty:
        raise ValueError(f"{path.name} has no rows.")

    mapping = resolve_columns(raw, overrides)
    if mapping["Case_Summary"] is None:
        raise ValueError(
            f"{path.name}: could not find a free-text summary column. Point at it "
            f"explicitly with --text-col. Available columns: {list(raw.columns)}"
        )

    df = pd.DataFrame(index=raw.index)
    df["Case_Summary"] = raw[mapping["Case_Summary"]]

    if mapping["Case_ID"] is not None:
        df["Case_ID"] = raw[mapping["Case_ID"]].astype(str)
    else:
        df["Case_ID"] = [f"ROW-{i + 1:04d}" for i in range(len(raw))]

    df["Department"] = (
        raw[mapping["Department"]] if mapping["Department"] else "Unspecified"
    )
    df["Date"] = raw[mapping["Date"]] if mapping["Date"] else pd.NaT
    df["Severity_Score"] = (
        raw[mapping["Severity_Score"]] if mapping["Severity_Score"] else 3
    )
    df = df[REQUIRED_COLUMNS]

    renamed = {k: v for k, v in mapping.items() if v is not None and v != k}
    if renamed:
        print("      mapped cols : " + ", ".join(f"{v!r} -> {k}" for k, v in renamed.items()))
    defaulted = [k for k, v in mapping.items() if v is None]
    if defaulted:
        print(f"      defaulted   : {', '.join(defaulted)} (not present in the file)")

    df = df.dropna(subset=["Case_Summary"]).copy()
    df["Case_Summary"] = df["Case_Summary"].astype(str).str.strip()
    df = df[df["Case_Summary"].str.len() > 0].reset_index(drop=True)

    df["Severity_Score"] = coerce_severity(df["Severity_Score"])
    df["Department"] = df["Department"].fillna("Unspecified").astype(str).str.strip()
    df.loc[df["Department"] == "", "Department"] = "Unspecified"
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    if len(df) < 10:
        raise ValueError(f"Only {len(df)} usable rows — need at least 10 to cluster.")

    print(f"[1/5] ingest      : {len(df)} cases from {path.name}")
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------------------
# 2. Embed
# --------------------------------------------------------------------------------------

def embed(texts: list[str]) -> np.ndarray:
    """Encode case summaries into L2-normalised 384-d dense vectors."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise SystemExit(
            "sentence-transformers is not installed.\n"
            "  pip install -r requirements.txt"
        ) from exc

    model = SentenceTransformer(MODEL_NAME)
    matrix = model.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)

    assert matrix.shape == (len(texts), EMBED_DIM), f"unexpected shape {matrix.shape}"
    print(f"[2/5] embed       : {matrix.shape[0]} x {matrix.shape[1]} ({MODEL_NAME})")
    return matrix


# --------------------------------------------------------------------------------------
# 3. Cluster
# --------------------------------------------------------------------------------------

def denoise(matrix: np.ndarray, dims: int = CLUSTER_DIMS, seed: int = 42) -> np.ndarray:
    """Project onto the leading principal components, then re-normalise to unit length.

    Clustering the raw 384-d vectors is measurably worse: most of those dimensions carry
    writing-style variance rather than failure-mode variance, and Euclidean distance in
    high dimensions concentrates. Benchmarked on the mock corpus against its (withheld)
    ground-truth labels:

        raw 384-d      silhouette +0.024   ARI +0.31   sizes [28, 19,  4, 26, 23]
        PCA 10-d       silhouette +0.124   ARI +0.43   sizes [24, 14, 25, 13, 24]
        PCA 20-d       silhouette +0.073   ARI +0.28
        PCA 50-d       silhouette +0.040   ARI +0.27

    Re-normalising after the projection keeps k-means operating on direction rather than
    magnitude, which is what cosine-trained sentence embeddings actually encode.
    """
    d = int(min(dims, matrix.shape[1], max(2, matrix.shape[0] - 1)))
    reduced = PCA(n_components=d, random_state=seed).fit_transform(matrix)
    norms = np.linalg.norm(reduced, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return reduced / norms


def _matched_links(texts: list[str]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Recorded must-/cannot-link pairs, translated onto row indices for this run.

    Matching is by feedback.case_key, not by row position, so a link recorded
    against last quarter's export still binds after re-ordering, re-filtering
    or adding new rows. Silent-empty when there is no store, mirroring the
    other feedback readers in this module (apply_human_names, apply_house_taxonomy).
    """
    try:
        import feedback
    except ImportError:  # pragma: no cover - optional component
        return [], []
    if not feedback.DB_PATH.exists():
        return [], []

    conn = feedback.connect()
    try:
        rows = conn.execute("SELECT case_a, case_b, kind FROM links").fetchall()
    finally:
        conn.close()
    if not rows:
        return [], []

    # First occurrence wins on a duplicated narrative — an arbitrary but
    # deterministic choice, and duplicated narratives are already an edge
    # case nothing else in the pipeline handles specially.
    index_of: dict[str, int] = {}
    for i, t in enumerate(texts):
        index_of.setdefault(feedback.case_key(t), i)

    must, cannot = [], []
    for r in rows:
        ia, ib = index_of.get(r["case_a"]), index_of.get(r["case_b"])
        if ia is None or ib is None or ia == ib:
            continue  # one or both cases are not in this run's data
        (must if r["kind"] == "same" else cannot).append((ia, ib))
    return must, cannot


def _components(n: int, must_pairs: list[tuple[int, int]]) -> list[list[int]]:
    """Union-find over must-link pairs: every pair that must stay together is
    collapsed into one unit before clustering, so it never can split."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in must_pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _assign_constrained(
    comp_points: np.ndarray, centroids: np.ndarray, comp_cannot: dict[int, set[int]],
) -> tuple[np.ndarray, list[int]]:
    """One COP-KMeans assignment pass: each unit goes to its nearest centroid
    that does not put it in the same group as a cannot-link partner already
    placed there this pass.

    Units closest to a centroid are placed first — the confident cases get
    their pick, so the flexibility that is left is spent on the contested
    ones. When a unit has no legal cluster left (its cannot-link partners
    already occupy every group, which happens when a cannot-link clique is
    larger than k) it is placed at its nearest centroid anyway and recorded
    as a violation, rather than raising and aborting the whole run.
    """
    n_comp = len(comp_points)
    eff_k = len(centroids)
    all_dists = np.linalg.norm(comp_points[:, None, :] - centroids[None, :, :], axis=2)
    order = np.argsort(all_dists.min(axis=1))

    assignment = np.full(n_comp, -1, dtype=int)
    violated: list[int] = []
    for ci in order:
        dists = all_dists[ci]
        forbidden = {assignment[cj] for cj in comp_cannot.get(ci, ())
                     if assignment[cj] != -1}
        candidates = [c for c in range(eff_k) if c not in forbidden]
        if candidates:
            best = min(candidates, key=lambda c: dists[c])
        else:
            best = int(np.argmin(dists))
            violated.append(ci)
        assignment[ci] = best
    return assignment, violated


CONSTRAINED_MAX_ITER = 15  # converges in a handful of passes on corpora this size


def constrained_cluster(
    matrix: np.ndarray,
    k: int,
    seed: int,
    must_pairs: list[tuple[int, int]],
    cannot_pairs: list[tuple[int, int]],
) -> tuple[np.ndarray, dict]:
    """KMeans constrained by recorded must-/cannot-link judgements (COP-KMeans).

    Must-link pairs are merged into single units first, so they are honoured
    by construction and can never be reported as violated. Cannot-link is
    enforced during assignment, unit by unit; where it cannot be — a
    cannot-link edge that falls inside a must-link unit, or a cannot-link
    clique bigger than k — the offending edge is dropped or the placement is
    recorded as a violation, and both are counted in the returned report
    rather than hidden.
    """
    space = denoise(matrix, seed=seed)
    n = len(space)
    components = _components(n, must_pairs)
    comp_of = np.empty(n, dtype=int)
    for cid, members in enumerate(components):
        comp_of[members] = cid

    comp_points = np.stack([space[m].mean(axis=0) for m in components])
    comp_weight = np.array([len(m) for m in components], dtype=float)

    comp_cannot: dict[int, set[int]] = {c: set() for c in range(len(components))}
    contradictory = 0
    for a, b in cannot_pairs:
        ca, cb = int(comp_of[a]), int(comp_of[b])
        if ca == cb:
            # a and b are must-linked AND cannot-linked — no k-way partition
            # can satisfy both. The must-link wins (it was recorded, and
            # splitting the unit would violate it too); the cannot-link is
            # dropped and counted, not silently kept.
            contradictory += 1
            continue
        comp_cannot[ca].add(cb)
        comp_cannot[cb].add(ca)

    n_comp = len(components)
    eff_k = min(k, n_comp)
    if eff_k < k:
        print(f"      note        : {n_comp} must-linked unit(s), fewer than "
              f"k={k}; clustering into {eff_k} instead")

    km = KMeans(n_clusters=eff_k, n_init=25, random_state=seed)
    km.fit(comp_points, sample_weight=comp_weight)
    centroids = km.cluster_centers_
    assignment = km.labels_.copy()
    violated: list[int] = []

    for _ in range(CONSTRAINED_MAX_ITER):
        new_assignment, violated = _assign_constrained(comp_points, centroids, comp_cannot)
        centroids = np.stack([
            comp_points[new_assignment == c].mean(axis=0)
            if (new_assignment == c).any() else centroids[c]
            for c in range(eff_k)
        ])
        converged = np.array_equal(new_assignment, assignment)
        assignment = new_assignment
        if converged:
            break

    labels = np.empty(n, dtype=int)
    for cid, members in enumerate(components):
        labels[members] = assignment[cid]

    cannot_honoured = sum(1 for a, b in cannot_pairs if labels[a] != labels[b])
    report = {
        "must_link": {"recorded": len(must_pairs), "honoured": len(must_pairs)},
        "cannot_link": {
            "recorded": len(cannot_pairs),
            "honoured": cannot_honoured,
            "violated": len(cannot_pairs) - cannot_honoured - contradictory,
            "contradictory": contradictory,
        },
    }
    return labels, report


def cluster(
    matrix: np.ndarray, k: int, seed: int = 42, texts: list[str] | None = None,
) -> tuple[np.ndarray, float, dict | None]:
    """Plain KMeans, unless recorded links bind this run's cases.

    `texts` is optional and only used to look up recorded must-/cannot-link
    judgements; when it is None, or no store exists, or nothing recorded
    matches this run's cases, this is byte-for-byte the same KMeans call as
    before links existed — the default path is never affected by a feature
    nobody has used yet.
    """
    space = denoise(matrix, seed=seed)
    must, cannot = _matched_links(texts) if texts is not None else ([], [])

    if not must and not cannot:
        km = KMeans(n_clusters=k, n_init=25, random_state=seed)
        labels = km.fit_predict(space)
        score = float(silhouette_score(space, labels)) if len(set(labels)) > 1 else 0.0
        sizes = np.bincount(labels, minlength=k).tolist()
        print(
            f"[3/5] cluster     : k={k}  sizes={sizes}  "
            f"silhouette={score:.3f} (in {space.shape[1]}-d denoised space)"
        )
        return labels, score, None

    labels, report = constrained_cluster(matrix, k, seed, must, cannot)
    score = float(silhouette_score(space, labels)) if len(set(labels)) > 1 else 0.0
    sizes = np.bincount(labels, minlength=k).tolist()
    ml, cl = report["must_link"], report["cannot_link"]
    print(
        f"[3/5] cluster     : k={k}  sizes={sizes}  "
        f"silhouette={score:.3f} (in {space.shape[1]}-d denoised space, constrained)"
    )
    note = f"must-link {ml['honoured']}/{ml['recorded']} honoured"
    note += f", cannot-link {cl['honoured']}/{cl['recorded']} honoured"
    if cl["violated"]:
        note += f", {cl['violated']} violated (clique exceeds k)"
    if cl["contradictory"]:
        note += f", {cl['contradictory']} dropped as contradictory"
    print(f"      constraints : {note}")
    return labels, score, report


# A term has to appear in at least this share of a cluster's cases before it is
# allowed into that cluster's name. Without it, TF-IDF happily names an 11-case
# cluster "Theatre / Waited / Delay" off words that appear in 3 cases each — a
# label that misdescribes two thirds of what it points at.
LABEL_COVERAGE_MIN = 0.20

# Built once, and kept out of the f-string/escape minefield that mangled it into
# a literal backspace on the way in.
WORD_BOUNDARY = r"\b%s\b"
LABEL_TERMS = 3

# Clinical abbreviations keep their capitals. Title-casing turns "iv" into "Iv"
# and "ct" into "Ct", which reads as a typo to the only audience that matters.
LABEL_ABBREVIATIONS = {
    "iv", "ct", "pe", "ed", "mri", "gi", "cxr", "inr", "tto", "aki", "nof",
    "dnacpr", "sbar", "icu", "obs", "vte", "cpr", "gp", "hdu", "ecg", "abg",
}


def _titlecase(term: str) -> str:
    return " ".join(w.upper() if w in LABEL_ABBREVIATIONS else w.title()
                    for w in term.split())


def name_clusters(
    texts: list[str],
    labels: np.ndarray,
    k: int,
    matrix: np.ndarray | None = None,
) -> list[dict]:
    """Name each cluster from vocabulary that is distinctive AND representative.

    Three things go wrong with naive top-TF-IDF labelling, and all three showed up
    on the mock corpus:

    1. **Frequency is not distinctiveness.** A term common to the whole register
       scores well inside every cluster. Scoring is therefore the cluster's mean
       TF-IDF *minus* the corpus mean, which is the c-TF-IDF idea: what does this
       group say more than everyone else?

    2. **Distinctiveness is not coverage.** A term can be unique to a cluster and
       still appear in three of its eleven cases. Terms must now clear
       LABEL_COVERAGE_MIN of the cluster's cases, measured directly against the
       text, or they do not get to name anything.

    3. **A three-word name is a lexical artefact in a semantic tool.** "Theatre"
       means two different things in "waited 14 hours for theatre" and "theatre
       lights failed", and a bag of words cannot tell them apart — which is the
       exact failure this project exists to fix. So every cluster also carries an
       `exemplar`: the case closest to its centroid in the full embedding space.
       That is a real case, chosen semantically, and it says far more about what
       the group is than three words can.

    When too few terms survive, the cluster is honestly named "Mixed group N"
    rather than given a confident label it has not earned.
    """
    vec = TfidfVectorizer(
        stop_words=_stopwords(),
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.55,
        sublinear_tf=True,
    )
    try:
        tfidf = vec.fit_transform(texts)
        vocab = np.array(vec.get_feature_names_out())
        corpus_mean = np.asarray(tfidf.mean(axis=0)).ravel()
    except ValueError:  # corpus too small / too uniform
        tfidf, vocab, corpus_mean = None, np.array([]), np.array([])

    lowered = [t.lower() for t in texts]

    # Score every (cluster, term) pair once, so terms can be handed out globally
    # rather than each cluster independently grabbing the same popular word.
    candidates: dict[int, list[tuple[float, str]]] = {c: [] for c in range(k)}
    for cid in range(k):
        mask = labels == cid
        if tfidf is None or not mask.any():
            continue
        members = [i for i, m in enumerate(mask) if m]
        distinct = np.asarray(tfidf[mask].mean(axis=0)).ravel() - corpus_mean

        for idx in distinct.argsort()[::-1][:40]:
            term = str(vocab[idx])
            score = float(distinct[idx])
            if score <= 0:
                continue
            # "1400", "14" and "0900" are timestamps out of the narrative, not
            # names for a failure mode.
            if any(ch.isdigit() for ch in term):
                continue
            # Word boundaries, not substrings: "ct" is inside "contact" and
            # "reflect", which quietly inflated coverage for exactly the short
            # clinical abbreviations most likely to end up in a label.
            pattern = re.compile(WORD_BOUNDARY % re.escape(term))
            coverage = sum(bool(pattern.search(lowered[i])) for i in members)
            coverage /= len(members)
            if coverage < LABEL_COVERAGE_MIN:
                continue
            # A bigram is a more specific claim than a single word, so it wins ties.
            candidates[cid].append(
                (score * (1.25 if " " in term else 1.0), term, coverage))
        candidates[cid].sort(reverse=True)

    # Hand terms out greedily by score: a term names the cluster that uses it most
    # distinctively, and no other.
    taken: set[str] = set()
    chosen: dict[int, list[str]] = {c: [] for c in range(k)}
    pool = sorted(
        ((score, cid, term, cov)
         for cid, lst in candidates.items() for score, term, cov in lst),
        reverse=True,
    )
    coverage_of: dict[str, float] = {}
    for _, cid, term, cov in pool:
        if term in taken or len(chosen[cid]) >= 8:
            continue
        # Skip a term already implied by one that was taken ("theatre" after
        # "theatre list"), which otherwise pads the name with a repeat.
        if any(term in t or t in term for t in chosen[cid]):
            continue
        chosen[cid].append(term)
        coverage_of[term] = cov
        taken.add(term)

    clusters = []
    for cid in range(k):
        mask = labels == cid
        terms = chosen[cid]
        named = terms[:LABEL_TERMS]
        headline = (" / ".join(_titlecase(t) for t in named)
                    if len(named) >= 2 else f"Group {cid + 1}")

        # How much of the group the name actually describes. On this corpus the
        # best available term covers under a third of its cluster, so a label is
        # a hint about vocabulary, not a definition of the group — and the number
        # is published so the interface can say so rather than implying otherwise.
        cover = (round(min(coverage_of[t] for t in named), 3)
                 if len(named) >= 2 else 0.0)

        entry = {
            "id": cid,
            "label": headline,
            "label_coverage": cover,
            # Carried so the workbench can record a human's name for this group
            # — and match it again next quarter — without re-embedding anything.
            "centroid": ([round(float(v), 5) for v in matrix[mask].mean(axis=0)]
                         if matrix is not None and mask.any() else []),
            "color": CLUSTER_PALETTE[cid % len(CLUSTER_PALETTE)],
            "size": int(mask.sum()),
            "keywords": terms[:8],
        }

        # The exemplar: the case nearest the centroid in the full embedding space.
        if matrix is not None and mask.any():
            members = np.flatnonzero(mask)
            centroid = matrix[members].mean(axis=0)
            norm = np.linalg.norm(centroid) or 1.0
            sims = matrix[members] @ (centroid / norm)
            medoid = int(members[int(np.argmax(sims))])
            entry["exemplar"] = {
                "i": medoid,
                "text": texts[medoid][:150],
            }

        clusters.append(entry)
    return clusters


# --------------------------------------------------------------------------------------
# 4. Project to 3D
# --------------------------------------------------------------------------------------

def project(
    matrix: np.ndarray, mode: str = "auto", seed: int = 42
) -> tuple[np.ndarray, str, float | None]:
    """Compress 384-d embeddings to 3 dimensions.

    `auto` prefers UMAP (better local structure preservation) and silently falls back to
    PCA, which is always available via scikit-learn.
    """
    if mode in ("auto", "umap"):
        try:
            import umap  # type: ignore

            reducer = umap.UMAP(
                n_components=3,
                n_neighbors=min(15, max(2, len(matrix) - 1)),
                min_dist=0.12,
                metric="cosine",
                random_state=seed,
            )
            return reducer.fit_transform(matrix).astype(np.float64), "umap", None
        except Exception as exc:
            if mode == "umap":
                raise
            print(f"      note        : umap unavailable ({type(exc).__name__}); using PCA")

    pca = PCA(n_components=3, random_state=seed)
    coords = pca.fit_transform(matrix)
    var = float(pca.explained_variance_ratio_.sum())
    print(f"      pca variance: {var:.1%} retained in 3 components")
    return coords.astype(np.float64), "pca", var


def rescale(coords: np.ndarray, half_edge: float = WORLD_HALF_EDGE) -> np.ndarray:
    """Centre on the barycentre and scale isotropically into a +/-half_edge cube.

    Isotropic (single global divisor) rather than per-axis, so relative distances are
    not distorted between axes.
    """
    centred = coords - coords.mean(axis=0)
    span = float(np.abs(centred).max())
    if span == 0:
        return centred
    return centred * (half_edge / span)


def separate_clusters(
    coords: np.ndarray,
    labels: np.ndarray,
    gain: float = SEPARATION_GAIN,
) -> tuple[np.ndarray, float]:
    """Push cluster centroids radially outward so galaxies read as distinct on screen.

    Points move rigidly with their centroid, so intra-cluster geometry — the part that
    is actually measured — is untouched. See SPECIFICATION.md §4.2.
    """
    out = coords.copy()
    applied = 1.0

    for _ in range(MAX_SEPARATION_PASSES):
        gaps = _min_centroid_gap(out, labels)
        if gaps >= MIN_CENTROID_GAP:
            break
        moved = out.copy()
        bary = out.mean(axis=0)
        for cid in np.unique(labels):
            mask = labels == cid
            centroid = out[mask].mean(axis=0)
            shift = (centroid - bary) * (gain - 1.0)
            moved[mask] = out[mask] + shift
        out = moved
        applied *= gain

    out = rescale(out)
    return out, applied


def nearest_neighbours(matrix: np.ndarray, k: int = 3) -> list[list[dict]]:
    """For each case, its k most semantically similar cases in the FULL 384-d space.

    This is the honest distance channel. It is computed before any projection, so it is
    unaffected by the 3-d squash and completely unaffected by the separation transform.
    Vectors are L2-normalised, so the dot product is cosine similarity.
    """
    sim = matrix @ matrix.T
    np.fill_diagonal(sim, -np.inf)
    order = np.argsort(-sim, axis=1)[:, :k]
    return [
        [{"i": int(j), "sim": round(float(sim[i, j]), 4)} for j in order[i]]
        for i in range(len(matrix))
    ]


def cluster_metrics(matrix: np.ndarray, labels: np.ndarray, k: int) -> list[dict]:
    """Measured cohesion and separation per cluster, in cosine space.

    cohesion  = mean pairwise cosine similarity within the cluster.
                High = the same failure repeating. Low = a grab-bag.
    nearest   = the cluster whose centroid is most similar, and by how much.
                A high value here means these two galaxies are genuinely adjacent,
                whatever the on-screen gap suggests.
    """
    cents = np.stack([
        matrix[labels == c].mean(axis=0) if (labels == c).any() else np.zeros(matrix.shape[1])
        for c in range(k)
    ])
    norms = np.linalg.norm(cents, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    cents = cents / norms
    cent_sim = cents @ cents.T
    np.fill_diagonal(cent_sim, -np.inf)

    out = []
    for c in range(k):
        members = matrix[labels == c]
        if len(members) > 1:
            sim = members @ members.T
            iu = np.triu_indices(len(members), k=1)
            cohesion = float(sim[iu].mean())
        else:
            cohesion = 1.0
        nearest = int(np.argmax(cent_sim[c]))
        out.append({
            "cohesion": round(cohesion, 4),
            "nearest_cluster": nearest,
            "nearest_similarity": round(float(cent_sim[c, nearest]), 4),
        })
    return out


def _min_centroid_gap(coords: np.ndarray, labels: np.ndarray) -> float:
    ids = np.unique(labels)
    if len(ids) < 2:
        return float("inf")
    cents = np.stack([coords[labels == c].mean(axis=0) for c in ids])
    gaps = [
        float(np.linalg.norm(cents[i] - cents[j]))
        for i in range(len(cents))
        for j in range(i + 1, len(cents))
    ]
    return min(gaps)


# --------------------------------------------------------------------------------------
# 5. Emit
# --------------------------------------------------------------------------------------

# A group is named after the house taxonomy only when the taxonomy is this sure
# of it. Below the threshold the group is genuinely something else, or something
# new, and inheriting last quarter's name would hide that.
HOUSE_NAME_AGREEMENT = 0.60


def record_run(source, k, texts, labels, clusters, qc, why) -> None:
    """Log this run and every assignment in it, if a store exists.

    Silent when feedback.py is absent or no store has been created — the
    pipeline gains no hard dependency, and a first-time user sees no difference.
    """
    try:
        import feedback
    except ImportError:  # pragma: no cover - optional component
        return
    if not feedback.DB_PATH.exists():
        return

    conn = feedback.connect()
    try:
        feedback.record_run(
            conn,
            source=source,
            k=k,
            n_cases=len(texts),
            qc=qc,
            keys=[feedback.case_key(t) for t in texts],
            clusters=clusters,
            labels=labels,
            justification=why,
        )
    finally:
        conn.close()


def recorded_best_k() -> dict | None:
    """Which k the run log says has held up best, or None if it can't yet say.

    Reads feedback.best_k, which needs at least two distinct k values with
    recorded stability before it will rank anything — so a fresh store, or one
    that has only ever been run at a single k, correctly yields None rather
    than a guess dressed up as evidence.
    """
    try:
        import feedback
    except ImportError:  # pragma: no cover - optional component
        return None
    if not feedback.DB_PATH.exists():
        return None

    conn = feedback.connect()
    try:
        return feedback.best_k(conn)
    finally:
        conn.close()


def confidence_levels() -> dict:
    """The three named confidence levels and their measured trade-off, for the
    page to render — never for it to decide anything with. feedback.py is the
    one place the thresholds and the holdout figures are defined; this only
    copies them into the payload so app/index.html has no numbers of its own
    to fall out of step.
    """
    try:
        import feedback
    except ImportError:  # pragma: no cover - optional component
        return {}
    return {
        "levels": feedback.CONFIDENCE_LEVELS,
        "tradeoff": feedback.CONFIDENCE_LEVEL_TRADEOFF,
        "default": "preparing",
    }


def apply_house_taxonomy(clusters, labels, matrix, texts=None) -> list[dict]:
    """Classify every case against labels a human taught the store earlier.

    Returns one record per case, in row order, for the payload. Also names a
    cluster after the dominant house label when enough of its cases agree —
    which is how the tool converges on the hospital's own vocabulary instead of
    TF-IDF's, without anything generative involved.
    """
    blank = [{"label": None, "confidence": None, "margin": None,
              "candidate_label": None, "withheld_at_default": None}
             for _ in range(len(labels))]
    try:
        import feedback
    except ImportError:  # pragma: no cover - optional component
        return blank
    if not feedback.DB_PATH.exists():
        return blank

    conn = feedback.connect()
    try:
        ready = any(n >= feedback.MIN_EXAMPLES
                    for n in feedback.taxonomy(conn).values())
        predictions = (feedback.classify(conn, matrix) if ready else list(blank))

        # A case filed by hand is not a prediction and cannot be overruled by
        # one. The clustering proposes; the clinician disposes.
        filed = feedback.filed_cases(conn)
        if filed and texts is not None:
            for i, text in enumerate(texts):
                name = filed.get(feedback.case_key(text))
                if name:
                    predictions[i] = {"label": name, "confidence": 1.0,
                                      "margin": None, "source": "human",
                                      "candidate_label": name,
                                      "withheld_at_default": False}
            n_filed = sum(1 for p_ in predictions if p_.get("source") == "human")
            if n_filed:
                print(f"      note        : {n_filed} case(s) filed by hand")
    finally:
        conn.close()

    for c in clusters:
        members = np.flatnonzero(labels == c["id"])
        votes: dict[str, int] = {}
        for i in members:
            name = predictions[i]["label"]
            if name:
                votes[name] = votes.get(name, 0) + 1
        if not votes or not len(members):
            continue

        winner, count = max(votes.items(), key=lambda kv: kv[1])
        share = count / len(members)
        if share >= HOUSE_NAME_AGREEMENT:
            c["label"] = winner
            c["label_source"] = "taxonomy"
            c["label_agreement"] = round(share, 3)

    named = sum(1 for c in clusters if c.get("label_source") == "taxonomy")
    if named:
        print(f"      note        : {named} group(s) named from the house taxonomy")
    return predictions


# A carried-over name (feedback.CARRY_OVER_MIN <= similarity < 1.0) is still
# permitted, but similarity alone does not say how much of the group actually
# survived. Measured on the shipped corpus by resampling 5-35% of cases out and
# re-clustering, then matching each new group to its nearest old one by centroid
# and recording the real case-membership overlap (Jaccard) alongside that match's
# similarity, 40 trials:
#
#   similarity band     mean membership overlap    n
#   [0.92, 0.94)         0.483                      27
#   [0.94, 0.96)         0.557                      43
#   [0.96, 0.98)         0.666                      45
#   [0.98, 1.00)         0.836                      21
#
# (correlation between similarity and overlap: 0.92). Below 0.96 less than
# half the group's membership typically survived even though the name was
# carried; at or above it, most of it did. 0.96 is where a carried name stops
# meaning "the same group, reshaped" and starts meaning "a different group
# that happens to be nearby" — so that is where drift is flagged as material.
DRIFT_MATERIAL_MAX = 0.96


def apply_human_names(clusters, texts, labels, matrix) -> int:
    """Restore names a human gave these groups on an earlier run.

    Silent no-op when there is no feedback store, so the pipeline has no new
    hard dependency and a fresh checkout behaves exactly as before.
    """
    try:
        import feedback
    except ImportError:  # pragma: no cover - optional component
        return 0
    if not feedback.DB_PATH.exists():
        return 0

    groups = []
    for c in clusters:
        members = np.flatnonzero(labels == c["id"])
        groups.append({
            "keys": [feedback.case_key(texts[i]) for i in members],
            "centroid": (matrix[members].mean(axis=0).tolist()
                         if len(members) else [0.0] * matrix.shape[1]),
        })

    conn = feedback.connect()
    try:
        restored = feedback.restore_names(conn, groups)
    finally:
        conn.close()

    applied = 0
    drifted = 0
    for c, r in zip(clusters, restored):
        if not r["name"]:
            continue
        c["label"] = r["name"]
        c["label_source"] = "human" if r["match"] == "exact" else "human_carried"
        c["label_match"] = r["match"]
        c["label_similarity"] = r["similarity"]
        # A carried-over name whose match is weaker than the band where most
        # of a group's membership typically survives — see DRIFT_MATERIAL_MAX.
        # Exact matches (the identical group re-formed) are never flagged.
        c["label_drift"] = bool(
            r["match"] == "carried_over"
            and r["similarity"] is not None
            and r["similarity"] < DRIFT_MATERIAL_MAX
        )
        applied += 1
        if c["label_drift"]:
            drifted += 1

    if applied:
        print(f"      note        : {applied} group name(s) restored from feedback.db")
    if drifted:
        print(f"      note        : {drifted} carried name(s) show material drift "
              f"(similarity below {DRIFT_MATERIAL_MAX}) — the group has likely "
              f"changed enough that the old name is worth revisiting")
    return applied


def build_payload(
    df: pd.DataFrame,
    coords: np.ndarray,
    labels: np.ndarray,
    clusters: list[dict],
    *,
    projection: str,
    silhouette: float,
    separation: float,
    source: str,
    label_model: str = "",
    house: list[dict] | None = None,
    qc: dict | None = None,
    why: list[dict] | None = None,
    variance: float | None = None,
    true_coords: np.ndarray | None = None,
    neighbours: list[list[dict]] | None = None,
    best_k: dict | None = None,
    constraints: dict | None = None,
) -> dict:
    """Assemble the JSON the canvas renders.

    Every point carries TWO coordinate triples:

      x,  y,  z   the amplified view — galaxies pushed apart for on-screen legibility
      tx, ty, tz  the true projection — the raw manifold, nothing added

    The canvas can switch between them, so the amplification is never something the
    reader has to take on trust. `nearest` carries measured cosine similarity from the
    full 384-d space, which no projection can distort.
    """
    if true_coords is None:
        true_coords = coords

    points = []
    for i, row in df.reset_index(drop=True).iterrows():
        date = row["Date"]
        points.append(
            {
                "id": str(row["Case_ID"]),
                "x": round(float(coords[i, 0]), 3),
                "y": round(float(coords[i, 1]), 3),
                "z": round(float(coords[i, 2]), 3),
                "tx": round(float(true_coords[i, 0]), 3),
                "ty": round(float(true_coords[i, 1]), 3),
                "tz": round(float(true_coords[i, 2]), 3),
                "cluster": int(labels[i]),
                "department": str(row["Department"]),
                "severity": int(row["Severity_Score"]),
                "date": "" if pd.isna(date) else pd.Timestamp(date).strftime("%Y-%m-%d"),
                "summary": str(row["Case_Summary"]),
                "nearest": (neighbours[i] if neighbours else []),
                # What the hospital's own taxonomy makes of this case at the
                # recorded ("preparing") standard, or null where the tool is
                # not entitled to an opinion yet. This is the stored answer —
                # a chosen confidence level never changes it, only how the
                # interface presents it (see house_candidate below).
                "house": (house[i]["label"] if house and house[i]["label"] else None),
                "house_confidence": (house[i]["confidence"] if house else None),
                "house_margin": (house[i].get("margin") if house else None),
                # The classifier's best-scoring label regardless of whether it
                # cleared the recorded standard, plus whether it did not — what
                # a looser confidence level re-gates on at render time, without
                # ever re-classifying the case. None for a hand-filed case:
                # there is no candidate to reconsider, only a person's answer.
                "house_candidate": (house[i].get("candidate_label")
                                    if house and house[i].get("source") != "human"
                                    else None),
                "house_withheld": (house[i].get("withheld_at_default")
                                   if house and house[i].get("source") != "human"
                                   else None),
                # "human" where a person filed it, otherwise the classifier's.
                "house_source": (house[i].get("source", "taxonomy")
                                 if house and house[i]["label"] else None),
                # Why this case sits in this group, in numbers a reader can check.
                "why": (why[i] if why else None),
            }
        )

    departments = sorted(df["Department"].unique().tolist())
    return {
        "meta": {
            "brand": "mmonfar.",
            "title": "Semantic M&M Failure Navigation Engine",
            "source_file": source,
            "model": MODEL_NAME,
            "embedding_dim": EMBED_DIM,
            "n_cases": len(points),
            "n_clusters": len(clusters),
            "projection": projection,
            "silhouette": round(silhouette, 4),
            "cluster_space_dims": CLUSTER_DIMS,
            "label_model": label_model,
            "separation_gain": round(separation, 3),
            "separation_note": (
                "Inter-cluster spacing is a presentation transform for legibility. "
                "Intra-cluster geometry is unmodified. Gaps are not measured distances."
            ),
            "true_geometry_note": (
                "Switch the canvas to TRUE geometry to see the projection with no "
                "amplification applied. Measured semantic distance is reported per case "
                "as cosine similarity in the full "
                f"{EMBED_DIM}-dimensional space, which no projection can distort."
            ),
            "variance_retained": variance,
            "variance_note": (
                "Share of the embedding variance surviving the squash to 3 axes. The rest "
                "is lost, which is why on-screen proximity is a hint and the cosine "
                "figures on each case card are the measurement."
            ),
            "generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
            "qc": qc or {},
            # The three named confidence levels and their measured trade-off
            # (feedback.CONFIDENCE_LEVELS / CONFIDENCE_LEVEL_TRADEOFF), shipped
            # in the payload so the page never hardcodes a threshold that could
            # drift out of step with feedback.py. Choosing a level re-gates
            # house_candidate at render time only — it is never sent back, and
            # it never changes house/house_confidence/qc, which is why it can
            # live in meta rather than in anything recorded.
            "confidence": confidence_levels(),
            # Which k the run log says has held up best across runs, ranked on
            # stability then agreement — None until there is enough history
            # (feedback.best_k needs >=2 distinct k values recorded).
            "best_k": best_k,
            # How many recorded must-/cannot-link judgements bound this run, and
            # what could not be honoured (contradictions, cliques bigger than k).
            # None means no links were recorded for this run's cases — not that
            # the check failed.
            "constraints": constraints,
            # Carried-over group names whose match similarity is materially
            # weaker than DRIFT_MATERIAL_MAX (engine.py) — see that constant for
            # the measurement behind the threshold. Each such cluster also
            # carries "label_drift": true individually.
            "name_drift": sum(1 for c in clusters if c.get("label_drift")),
        },
        "departments": departments,
        "clusters": clusters,
        "points": points,
    }


def inject_into_html(payload: dict, template_path: Path, out_path: Path) -> bool:
    """Write a single-file copy of the page with the payload inlined.

    SAFETY: this writes to `out_path` (app/standalone.html, git-ignored) and never
    modifies the template. Injecting into the tracked index.html would embed every
    case narrative into a source file — with a real register, that is patient data
    committed to git history, and git history is difficult to retract. The template
    stays clean and fetches data.json when served; the standalone copy exists only
    so the page also works from a bare file:// URL, and is ignored by git for the
    same reason data.json is.
    """
    if not template_path.exists():
        print(f"      warn        : {template_path.name} not found, skipped standalone")
        return False

    html = template_path.read_text(encoding="utf-8")
    if PAYLOAD_START not in html or PAYLOAD_END not in html:
        print("      warn        : payload markers absent, skipped standalone")
        return False

    block = (
        f"{PAYLOAD_START}\n"
        '    <script id="mm-payload" type="application/json">\n'
        f"{json.dumps(payload, ensure_ascii=False, indent=1)}\n"
        "    </script>\n"
        f"    {PAYLOAD_END}"
    )
    pattern = re.compile(
        re.escape(PAYLOAD_START) + r".*?" + re.escape(PAYLOAD_END), re.DOTALL
    )
    out_path.write_text(pattern.sub(lambda _: block, html, count=1), encoding="utf-8")
    return True


# --------------------------------------------------------------------------------------

def run(
    input_path: Path = DEFAULT_INPUT,
    k: int = 5,
    projection_mode: str = "auto",
    seed: int = 42,
    overrides: dict[str, str] | None = None,
    label_model: str | None = None,
) -> dict:
    df = load_minutes(input_path, overrides)
    texts = df["Case_Summary"].tolist()

    # Someone pointing this at a 30-case departmental export should not hit a
    # scikit-learn traceback because the default k does not fit their data.
    if k > len(df):
        print(f"      note        : k={k} exceeds {len(df)} cases; using k={max(2, len(df) // 4)}")
        k = max(2, len(df) // 4)

    matrix = embed(texts)
    labels, silhouette, constraints = cluster(matrix, k=k, seed=seed, texts=texts)
    clusters = name_clusters(texts, labels, k, matrix)
    for c, m in zip(clusters, cluster_metrics(matrix, labels, k)):
        c.update(m)

    # Optional: let a small local model read each group and name it. Off unless
    # asked for, and it can only ever replace a name — never a measurement.
    model_used = ""
    if label_model:
        import labeller

        model_used = labeller.apply(clusters, texts, labels, matrix, label_model)
    else:
        for c in clusters:
            c.setdefault("label_source", "terms" if c.get("keywords") else "numbered")

    # What the hospital has already decided, applied to what it is looking at
    # now. This is the whole refinement loop: no model, no training run — the
    # embeddings are already here, and a label is the mean of the cases a person
    # filed under it. It sharpens every time somebody names something.
    house = apply_house_taxonomy(clusters, labels, matrix, texts)

    # A human's name for THIS group outranks everything above it. Applied last,
    # so it overwrites the taxonomy, the keyword label and anything a model
    # wrote — the person who chaired the meeting is the better authority.
    apply_human_names(clusters, texts, labels, matrix)

    raw, projection, variance = project(matrix, mode=projection_mode, seed=seed)
    true_coords = rescale(raw)
    coords, separation = separate_clusters(true_coords, labels)
    print(
        f"[4/5] project     : {projection}  "
        f"min centroid gap={_min_centroid_gap(coords, labels):.1f}u  gain={separation:.2f}x"
    )

    # Quality control, on every run. Without it the map is a picture somebody is
    # asked to trust; with it, the reader can see whether the grouping is real.
    import quality

    qc = quality.report(matrix, labels, k, seed)
    why = quality.per_case(matrix, labels)
    na = qc["neighbour_agreement"]
    print(f"      quality     : {qc['verdict']}  "
          f"({qc['checks_passed']}/3)  nearest-case agreement "
          f"{na['agreement']:.0%} vs {na['chance']:.0%} at chance")

    record_run(input_path.name, k, texts, labels, clusters, qc, why)

    payload = build_payload(
        df, coords, labels, clusters,
        projection=projection,
        silhouette=silhouette,
        separation=separation,
        source=input_path.name,
        variance=variance,
        true_coords=true_coords,
        neighbours=nearest_neighbours(matrix),
        label_model=model_used,
        house=house,
        qc=qc,
        why=why,
        best_k=recorded_best_k(),
        constraints=constraints,
    )

    APP_DIR.mkdir(parents=True, exist_ok=True)
    json_path = APP_DIR / "data.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    standalone = APP_DIR / "standalone.html"
    injected = inject_into_html(payload, APP_DIR / "index.html", standalone)

    print(f"[5/5] emit        : {json_path.relative_to(ROOT)}"
          + (f"  +  {standalone.relative_to(ROOT)}" if injected else ""))
    print("\n      galaxies discovered:")
    for c in clusters:
        mark = {"model": "~", "terms": " ", "numbered": "?",
                "taxonomy": "+", "human": "*", "human_carried": "*"}.get(
            c.get("label_source", " "), " ")
        print(f"        [{c['id']}] {c['size']:>3} cases {mark} {c['label']}")
    if model_used:
        print(f"\n      ~ written by {model_used}   * named by a person")
    elif any(c.get("label_source", "").startswith(("human", "taxonomy"))
             for c in clusters):
        print("\n      * named by a person   + learned from earlier decisions")
    print("\n[mmonfar.] done. Open app/index.html in a browser.")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compile M&M minutes (.xlsx/.xls/.xlsm/.csv/.tsv) into a semantic galaxy.",
        epilog=(
            "Column names are matched case- and punctuation-insensitively against a list "
            "of common synonyms, so most real exports work with no flags at all. Use the "
            "--*-col flags when a column is named something unusual."
        ),
    )
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                    help="path to a spreadsheet or delimited text file")
    ap.add_argument("--clusters", type=int, default=5)
    ap.add_argument("--projection", choices=["auto", "umap", "pca"], default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--text-col", help="column holding the free-text case summary")
    ap.add_argument("--id-col", help="column holding the case reference")
    ap.add_argument("--date-col", help="column holding the case date")
    ap.add_argument("--dept-col", help="column holding the department/specialty")
    ap.add_argument("--severity-col", help="column holding severity or harm (1-5, or words)")
    ap.add_argument("--smart-labels", action="store_true",
                    help="name each group with a small local language model "
                         "(~1GB download on first use, runs offline thereafter)")
    ap.add_argument("--label-model", default=None,
                    help="model id for --smart-labels (default: "
                         "Qwen/Qwen2.5-0.5B-Instruct)")
    args = ap.parse_args()

    overrides = {
        k: v for k, v in {
            "Case_Summary": args.text_col,
            "Case_ID": args.id_col,
            "Date": args.date_col,
            "Department": args.dept_col,
            "Severity_Score": args.severity_col,
        }.items() if v
    }

    try:
        model = None
        if args.smart_labels or args.label_model:
            import labeller

            model = args.label_model or labeller.DEFAULT_MODEL

        run(args.input, k=args.clusters, projection_mode=args.projection, seed=args.seed,
            overrides=overrides, label_model=model)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[mmonfar.] error: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
