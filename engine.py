"""
engine.py
mmonfar. // Semantic M&M Failure Navigation Engine — Phase 3.2

Batch compiler. Reads a spreadsheet of M&M minutes, turns the free text into a semantic
map, and emits a static JSON payload that the Three.js canvas in `app/` renders without
ever calling back into Python.

    Excel -> MiniLM embeddings (384-d) -> KMeans(k=5) -> UMAP/PCA(3-d) -> app/data.json

Usage:
    python engine.py
    python engine.py --input mock_mm_minutes.xlsx --clusters 5 --projection auto

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

def load_minutes(path: Path) -> pd.DataFrame:
    """Read the spreadsheet and enforce the column contract."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python data_generator.py` first, or pass --input."
        )

    df = pd.read_excel(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required column(s): {missing}")

    df = df.dropna(subset=["Case_Summary"]).copy()
    df["Case_Summary"] = df["Case_Summary"].astype(str).str.strip()
    df = df[df["Case_Summary"].str.len() > 0].reset_index(drop=True)

    df["Severity_Score"] = (
        pd.to_numeric(df["Severity_Score"], errors="coerce").fillna(3).clip(1, 5).astype(int)
    )
    df["Department"] = df["Department"].fillna("Unspecified").astype(str)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    if len(df) < 10:
        raise ValueError(f"Only {len(df)} usable rows — need at least 10 to cluster.")

    print(f"[1/5] ingest      : {len(df)} cases from {path.name}")
    return df


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


def cluster(matrix: np.ndarray, k: int, seed: int = 42) -> tuple[np.ndarray, float]:
    space = denoise(matrix, seed=seed)
    km = KMeans(n_clusters=k, n_init=25, random_state=seed)
    labels = km.fit_predict(space)
    score = float(silhouette_score(space, labels)) if len(set(labels)) > 1 else 0.0
    sizes = np.bincount(labels, minlength=k).tolist()
    print(
        f"[3/5] cluster     : k={k}  sizes={sizes}  "
        f"silhouette={score:.3f} (in {space.shape[1]}-d denoised space)"
    )
    return labels, score


def name_clusters(texts: list[str], labels: np.ndarray, k: int) -> list[dict]:
    """Derive a human-readable label per cluster from its distinctive vocabulary.

    A TF-IDF matrix is built over the whole corpus, then averaged within each cluster.
    The top-weighted terms of that mean vector are the terms the cluster uses *more*
    than the rest of the corpus does — a cheap, dependency-free stand-in for c-TF-IDF.
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
    except ValueError:  # corpus too small / too uniform
        tfidf, vocab = None, np.array([])

    clusters = []
    for cid in range(k):
        mask = labels == cid
        terms: list[str] = []
        if tfidf is not None and mask.any():
            mean_vec = np.asarray(tfidf[mask].mean(axis=0)).ravel()
            top_idx = mean_vec.argsort()[::-1][:8]
            terms = [vocab[i] for i in top_idx if mean_vec[i] > 0]

        headline = " / ".join(t.title() for t in terms[:3]) if terms else f"Cluster {cid}"
        clusters.append(
            {
                "id": cid,
                "label": headline,
                "color": CLUSTER_PALETTE[cid % len(CLUSTER_PALETTE)],
                "size": int(mask.sum()),
                "keywords": terms[:8],
            }
        )
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
    variance: float | None = None,
    true_coords: np.ndarray | None = None,
    neighbours: list[list[dict]] | None = None,
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
        },
        "departments": departments,
        "clusters": clusters,
        "points": points,
    }


def inject_into_html(payload: dict, html_path: Path) -> bool:
    """Replace the payload block inside index.html so the page works over file://."""
    if not html_path.exists():
        print(f"      warn        : {html_path.name} not found, skipped injection")
        return False

    html = html_path.read_text(encoding="utf-8")
    if PAYLOAD_START not in html or PAYLOAD_END not in html:
        print("      warn        : payload markers absent, skipped injection")
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
    html_path.write_text(pattern.sub(lambda _: block, html, count=1), encoding="utf-8")
    return True


# --------------------------------------------------------------------------------------

def run(
    input_path: Path = DEFAULT_INPUT,
    k: int = 5,
    projection_mode: str = "auto",
    seed: int = 42,
) -> dict:
    df = load_minutes(input_path)
    texts = df["Case_Summary"].tolist()

    matrix = embed(texts)
    labels, silhouette = cluster(matrix, k=k, seed=seed)
    clusters = name_clusters(texts, labels, k)
    for c, m in zip(clusters, cluster_metrics(matrix, labels, k)):
        c.update(m)

    raw, projection, variance = project(matrix, mode=projection_mode, seed=seed)
    true_coords = rescale(raw)
    coords, separation = separate_clusters(true_coords, labels)
    print(
        f"[4/5] project     : {projection}  "
        f"min centroid gap={_min_centroid_gap(coords, labels):.1f}u  gain={separation:.2f}x"
    )

    payload = build_payload(
        df, coords, labels, clusters,
        projection=projection,
        silhouette=silhouette,
        separation=separation,
        source=input_path.name,
        variance=variance,
        true_coords=true_coords,
        neighbours=nearest_neighbours(matrix),
    )

    APP_DIR.mkdir(parents=True, exist_ok=True)
    json_path = APP_DIR / "data.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    injected = inject_into_html(payload, APP_DIR / "index.html")

    print(f"[5/5] emit        : {json_path.relative_to(ROOT)}"
          + ("  + injected into app/index.html" if injected else ""))
    print("\n      galaxies discovered:")
    for c in clusters:
        print(f"        [{c['id']}] {c['size']:>3} cases  {c['label']}")
    print("\n[mmonfar.] done. Open app/index.html in a browser.")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Compile M&M minutes into a semantic galaxy.")
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--clusters", type=int, default=5)
    ap.add_argument("--projection", choices=["auto", "umap", "pca"], default="auto")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    try:
        run(args.input, k=args.clusters, projection_mode=args.projection, seed=args.seed)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[mmonfar.] error: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
