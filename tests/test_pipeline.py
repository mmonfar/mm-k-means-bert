"""
tests/test_pipeline.py
mmonfar. // Contract tests for the semantic M&M pipeline.

These are deliberately split:

  * Generator + geometry tests are fast, hermetic, and always run.
  * The one test that needs the MiniLM weights is marked `slow` and skips itself when
    sentence-transformers is not installed, so `pytest` stays green on a clean checkout.

Run everything:      pytest -q
Skip the slow one:   pytest -q -m "not slow"
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import data_generator  # noqa: E402
import engine  # noqa: E402


# ---------------------------------------------------------------- generator

def test_generator_produces_exactly_100_rows():
    df = data_generator.build_frame(rows=100, seed=42)
    assert len(df) == 100


def test_generator_column_contract():
    df = data_generator.build_frame(rows=100, seed=42).drop(columns=["_ground_truth"])
    assert list(df.columns) == engine.REQUIRED_COLUMNS


def test_generator_covers_all_five_failure_modes():
    df = data_generator.build_frame(rows=100, seed=42)
    assert df["_ground_truth"].nunique() == 5


def test_case_ids_are_unique():
    df = data_generator.build_frame(rows=100, seed=42)
    assert df["Case_ID"].is_unique


def test_summaries_are_unique_and_substantial():
    df = data_generator.build_frame(rows=100, seed=42)
    assert df["Case_Summary"].is_unique
    assert (df["Case_Summary"].str.len() > 60).all()


def test_severity_within_range():
    df = data_generator.build_frame(rows=100, seed=42)
    assert df["Severity_Score"].between(1, 5).all()


def test_semantic_traps_are_present():
    """The corpus must contain the lexically-divergent anticoagulation trio and a
    negation loop — these are what the engine is being asked to survive."""
    corpus = " ".join(data_generator.build_frame(rows=100, seed=42)["Case_Summary"]).lower()
    assert "blood thinner mistake" in corpus
    assert "heparin administration error" in corpus
    assert "warfarin dose miscalculated" in corpus
    assert "no surgical complication noted" in corpus


def test_generator_is_deterministic():
    a = data_generator.build_frame(rows=100, seed=7)["Case_ID"].tolist()
    b = data_generator.build_frame(rows=100, seed=7)["Case_ID"].tolist()
    assert a == b


# ---------------------------------------------------------------- ingest

@pytest.fixture
def corpus():
    return data_generator.build_frame(rows=100, seed=42).drop(columns=["_ground_truth"])


def _messy_export(corpus):
    """A realistic hospital export: nothing named the way we would like."""
    import pandas as pd

    return pd.DataFrame({
        "Datix Ref": corpus["Case_ID"],
        "Date of Incident": corpus["Date"],
        "Specialty": corpus["Department"],
        "What Happened?": corpus["Case_Summary"],
        "Grade of Harm": corpus["Severity_Score"].map(
            {1: "None", 2: "Low", 3: "Moderate", 4: "Major", 5: "Catastrophic"}
        ),
    })


@pytest.mark.parametrize("suffix", [".xlsx", ".csv", ".tsv"])
def test_reads_every_supported_format(tmp_path, corpus, suffix):
    target = tmp_path / f"minutes{suffix}"
    if suffix == ".xlsx":
        corpus.to_excel(target, index=False)
    else:
        corpus.to_csv(target, index=False, sep="\t" if suffix == ".tsv" else ",")

    df = engine.load_minutes(target)
    assert len(df) == 100
    assert list(df.columns) == engine.REQUIRED_COLUMNS


def test_rejects_unsupported_format(tmp_path):
    target = tmp_path / "notes.docx"
    target.write_text("nope", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported file type"):
        engine.load_minutes(target)


def test_maps_real_world_headers_without_flags(tmp_path, corpus):
    target = tmp_path / "datix_export.csv"
    _messy_export(corpus).to_csv(target, index=False, sep=";", encoding="cp1252")

    df = engine.load_minutes(target)
    assert len(df) == 100
    assert df["Case_ID"].iloc[0] == corpus["Case_ID"].iloc[0]
    assert df["Case_Summary"].iloc[0] == corpus["Case_Summary"].iloc[0]


def test_worded_harm_column_becomes_a_1_to_5_score(tmp_path, corpus):
    """'None' means no harm, not a missing value — pandas would call it NaN."""
    target = tmp_path / "worded.csv"
    _messy_export(corpus).to_csv(target, index=False)

    df = engine.load_minutes(target)
    assert set(df["Severity_Score"]) <= {1, 2, 3, 4, 5}
    assert (df["Severity_Score"] == 1).any(), "'None' was swallowed as a missing value"
    assert df["Severity_Score"].tolist() == corpus["Severity_Score"].tolist()


def test_survives_a_file_with_only_a_text_column(tmp_path, corpus):
    target = tmp_path / "bare.csv"
    corpus[["Case_Summary"]].to_csv(target, index=False)

    df = engine.load_minutes(target)
    assert len(df) == 100
    assert df["Case_ID"].iloc[0] == "ROW-0001"
    assert (df["Department"] == "Unspecified").all()
    assert (df["Severity_Score"] == 3).all()


def test_explicit_column_overrides_win(tmp_path, corpus):
    import pandas as pd

    # Two plausible text columns: synonym matching would pick the wrong one.
    frame = pd.DataFrame({
        "Summary": ["short decoy text that is long enough to look plausible"] * 100,
        "Full Narrative": corpus["Case_Summary"],
    })
    target = tmp_path / "ambiguous.csv"
    frame.to_csv(target, index=False)

    df = engine.load_minutes(target, {"Case_Summary": "Full Narrative"})
    assert df["Case_Summary"].iloc[0] == corpus["Case_Summary"].iloc[0]


def test_unknown_override_names_the_available_columns(tmp_path, corpus):
    target = tmp_path / "minutes.csv"
    corpus.to_csv(target, index=False)
    with pytest.raises(ValueError, match="not a column"):
        engine.load_minutes(target, {"Case_Summary": "Nope"})


def test_refuses_a_file_too_small_to_cluster(tmp_path, corpus):
    target = tmp_path / "tiny.csv"
    corpus.head(4).to_csv(target, index=False)
    with pytest.raises(ValueError, match="need at least 10"):
        engine.load_minutes(target)


# ---------------------------------------------------------------- geometry

@pytest.fixture
def blobs():
    """Three well-separated synthetic blobs standing in for embedded clusters."""
    rng = np.random.default_rng(0)
    centres = np.array([[0, 0, 0], [1.0, 0, 0], [0, 1.0, 0]], dtype=float)
    coords = np.vstack([c + rng.normal(0, 0.05, (20, 3)) for c in centres])
    labels = np.repeat([0, 1, 2], 20)
    return coords, labels


def test_rescale_fits_the_world_cube(blobs):
    coords, _ = blobs
    out = engine.rescale(coords)
    assert np.abs(out).max() == pytest.approx(engine.WORLD_HALF_EDGE, rel=1e-6)
    assert np.allclose(out.mean(axis=0), 0, atol=1e-9)


def test_separation_reaches_the_minimum_gap(blobs):
    coords, labels = blobs
    out, gain = engine.separate_clusters(engine.rescale(coords), labels)
    assert engine._min_centroid_gap(out, labels) >= engine.MIN_CENTROID_GAP
    assert gain >= 1.0


def test_separation_preserves_intra_cluster_shape(blobs):
    """Only inter-cluster whitespace may be manufactured. Within a cluster, the
    pairwise distance matrix must survive up to the single global rescale factor."""
    coords, labels = blobs
    start = engine.rescale(coords)
    out, _ = engine.separate_clusters(start, labels)

    mask = labels == 0
    before = np.linalg.norm(start[mask][:, None] - start[mask][None], axis=-1)
    after = np.linalg.norm(out[mask][:, None] - out[mask][None], axis=-1)

    nz = before > 1e-9
    ratios = after[nz] / before[nz]
    assert np.allclose(ratios, ratios[0], rtol=1e-6)


def test_separation_leaves_already_separated_clusters_alone(blobs):
    coords, labels = blobs
    start = engine.rescale(coords * 50)
    out, gain = engine.separate_clusters(start, labels)
    assert gain == pytest.approx(1.0)
    assert np.allclose(out, start)


# ---------------------------------------------------------------- payload

def test_payload_shape_and_bounds():
    df = data_generator.build_frame(rows=100, seed=42).drop(columns=["_ground_truth"])
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 5, len(df))
    coords = engine.rescale(rng.normal(0, 1, (len(df), 3)))
    clusters = engine.name_clusters(df["Case_Summary"].tolist(), labels, 5)

    payload = engine.build_payload(
        df, coords, labels, clusters,
        projection="pca", silhouette=0.1, separation=1.0, source="test.xlsx",
    )

    assert payload["meta"]["n_cases"] == 100
    assert payload["meta"]["embedding_dim"] == engine.EMBED_DIM
    assert len(payload["clusters"]) == 5
    assert len(payload["points"]) == 100

    for p in payload["points"]:
        assert set(p) == {
            "id", "x", "y", "z", "tx", "ty", "tz",
            "cluster", "department", "severity", "date", "summary", "nearest",
        }
        assert 0 <= p["cluster"] < 5
        assert 1 <= p["severity"] <= 5
        for axis in "xyz":
            assert abs(p[axis]) <= engine.WORLD_HALF_EDGE + 1e-6

    # Must survive a JSON round-trip — it is embedded verbatim into index.html.
    assert json.loads(json.dumps(payload))["meta"]["n_cases"] == 100


def test_payload_carries_both_geometries():
    """The canvas must be able to show the projection unamplified, so both coordinate
    sets have to survive into the payload as distinct values."""
    df = data_generator.build_frame(rows=100, seed=42).drop(columns=["_ground_truth"])
    rng = np.random.default_rng(2)
    labels = rng.integers(0, 5, len(df))
    true_coords = engine.rescale(rng.normal(0, 1, (len(df), 3)))
    view, gain = engine.separate_clusters(true_coords, labels)
    clusters = engine.name_clusters(df["Case_Summary"].tolist(), labels, 5)

    payload = engine.build_payload(
        df, view, labels, clusters,
        projection="pca", silhouette=0.1, separation=gain, source="t.xlsx",
        true_coords=true_coords,
    )

    assert payload["meta"]["separation_gain"] == pytest.approx(round(gain, 3))
    assert payload["meta"]["true_geometry_note"]
    moved = [
        p for p in payload["points"]
        if abs(p["x"] - p["tx"]) > 1e-6 or abs(p["y"] - p["ty"]) > 1e-6
    ]
    assert moved, "amplified and true geometries are identical — nothing to disclose"


def test_nearest_neighbours_are_measured_and_ranked():
    """Cosine neighbours must exclude self and come back in descending similarity."""
    rng = np.random.default_rng(3)
    m = rng.normal(0, 1, (12, 8))
    m /= np.linalg.norm(m, axis=1, keepdims=True)

    nn = engine.nearest_neighbours(m, k=3)
    assert len(nn) == 12
    for i, row in enumerate(nn):
        assert len(row) == 3
        assert all(e["i"] != i for e in row)
        sims = [e["sim"] for e in row]
        assert sims == sorted(sims, reverse=True)
        assert all(-1.0001 <= s <= 1.0001 for s in sims)


def test_cluster_metrics_report_cohesion_and_adjacency():
    rng = np.random.default_rng(4)
    tight = rng.normal(0, 0.01, (15, 8)) + np.array([1, 0, 0, 0, 0, 0, 0, 0])
    loose = rng.normal(0, 1.0, (15, 8))
    m = np.vstack([tight, loose])
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    labels = np.repeat([0, 1], 15)

    metrics = engine.cluster_metrics(m, labels, 2)
    assert metrics[0]["cohesion"] > metrics[1]["cohesion"]
    assert metrics[0]["nearest_cluster"] == 1
    assert -1.0001 <= metrics[0]["nearest_similarity"] <= 1.0001


def test_label_terms_must_actually_describe_their_group():
    """A name has to cover a real share of the cases it points at.

    This is the bug that prompted the rule: an 11-case group was being named
    "Theatre / Waited / Delay" off terms that appeared in 3 cases each, and two
    of those cases used "theatre" to mean an operating room rather than a queue
    for one. A name that misdescribes two thirds of its group is worse than no
    name, so a term now has to clear LABEL_COVERAGE_MIN of the group — measured
    on word boundaries, because "ct" is inside "contact" and "reflect".
    """
    texts = [
        "Delay to theatre for a perforated viscus, waited 14 hours for a slot.",
        "Fractured neck of femur waited 62 hours against a 36 hour standard.",
        "Upper GI bleed waited overnight for endoscopy, on-call from home.",
        "Retained swab found on the post-op chest film after laparotomy.",
        "Theatre lights failed mid-case, generator switchover took 40 seconds.",
        "Prosthetic joint infection at five weeks, antibiotic given after tourniquet.",
    ]
    labels = np.zeros(len(texts), dtype=int)
    clusters = engine.name_clusters(texts, labels, 1)
    label = clusters[0]["label"].lower()

    for term in clusters[0]["keywords"]:
        pattern = re.compile(engine.WORD_BOUNDARY % re.escape(term))
        hits = sum(bool(pattern.search(t.lower())) for t in texts)
        assert hits / len(texts) >= engine.LABEL_COVERAGE_MIN, (
            f"{term!r} names the group but appears in only {hits}/{len(texts)} cases"
        )

    # "40", "14" and "62" are timestamps out of the narrative, not failure modes.
    # Checked on the keywords, not the label: the honest fallback is "Group 1",
    # whose digit is a group number rather than a stray token.
    assert not any(ch.isdigit() for t in clusters[0]["keywords"] for ch in t)
    assert label


def test_unnamed_groups_are_numbered_not_invented():
    """With no shared vocabulary, the honest label is a number."""
    texts = [
        "Wrong dose of insulin drawn up overnight on a busy medical ward.",
        "Scaffolding collapsed in the car park during the storm on Tuesday.",
        "Interpreter unavailable so consent was taken through a family member.",
        "Freezer alarm unheard, three units of platelets discarded next morning.",
        "Porter took the patient to the wrong department for their appointment.",
        "Cleaning trolley blocked the fire door on the third floor landing.",
    ]
    labels = np.zeros(len(texts), dtype=int)
    label = engine.name_clusters(texts, labels, 1)[0]["label"]
    assert label == "Group 1"


def test_every_group_carries_a_semantic_exemplar():
    """The exemplar is the case nearest the centroid, chosen in embedding space
    rather than by keyword — it is what actually identifies a mixed group."""
    rng = np.random.default_rng(5)
    m = rng.normal(0, 1, (12, 16))
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    texts = [f"case number {i} with some narrative text attached" for i in range(12)]
    labels = np.array([0] * 6 + [1] * 6)

    clusters = engine.name_clusters(texts, labels, 2, m)
    for c in clusters:
        assert "exemplar" in c
        assert c["exemplar"]["text"]

    # It must be a genuine member of its own group, not just any case.
    assert clusters[0]["exemplar"]["i"] < 6
    assert clusters[1]["exemplar"]["i"] >= 6


def test_cluster_colors_come_from_the_brand_palette():
    df = data_generator.build_frame(rows=100, seed=42)
    labels = np.array([i % 5 for i in range(len(df))])
    clusters = engine.name_clusters(df["Case_Summary"].tolist(), labels, 5)
    assert [c["color"] for c in clusters] == engine.CLUSTER_PALETTE
    assert sum(c["size"] for c in clusters) == len(df)


def test_html_has_injection_markers():
    html = (ROOT / "app" / "index.html").read_text(encoding="utf-8")
    assert engine.PAYLOAD_START in html
    assert engine.PAYLOAD_END in html
    assert html.index(engine.PAYLOAD_START) < html.index(engine.PAYLOAD_END)


def test_injection_writes_a_copy_and_never_touches_the_template(tmp_path):
    template = ROOT / "app" / "index.html"
    before = template.read_text(encoding="utf-8")
    out = tmp_path / "standalone.html"

    payload = {"meta": {"n_cases": 3}, "clusters": [], "points": [], "departments": []}
    assert engine.inject_into_html(payload, template, out) is True

    written = out.read_text(encoding="utf-8")
    assert '"n_cases": 3' in written
    assert written.count(engine.PAYLOAD_START) == 1
    assert written.count(engine.PAYLOAD_END) == 1
    # The tracked template must come out byte-identical.
    assert template.read_text(encoding="utf-8") == before

    # Idempotent: rendering twice must not nest or duplicate blocks.
    engine.inject_into_html(payload, template, out)
    assert out.read_text(encoding="utf-8").count(engine.PAYLOAD_START) == 1


def test_tracked_template_carries_no_case_data():
    """The leak guard, as a test.

    app/index.html is tracked. If the pipeline ever writes a payload into it,
    running the tool on a real register would commit patient narratives to git
    history. The template must keep an empty payload.
    """
    html = (ROOT / "app" / "index.html").read_text(encoding="utf-8")
    start = html.index(engine.PAYLOAD_START)
    end = html.index(engine.PAYLOAD_END)
    block = html[start:end]
    assert '"points"' not in block, "a payload has been baked into the tracked template"
    assert ">null<" in block


# ---------------------------------------------------------------- semantics (slow)

@pytest.mark.slow
def test_embeddings_bind_lexically_divergent_synonyms():
    """The headline claim: 'blood thinner mistake' must sit closer to 'heparin
    administration error' than to an unrelated equipment failure."""
    pytest.importorskip("sentence_transformers")

    texts = [
        "Blood thinner mistake on the ward, pt restarted on apixaban without holding.",
        "Heparin administration error, infusion running at the wrong rate overnight.",
        "Defibrillator failed self-test at the start of the arrest call.",
    ]
    v = engine.embed(texts)
    sim = v @ v.T  # vectors are L2-normalised, so this is cosine similarity
    assert sim[0, 1] > sim[0, 2]


@pytest.mark.slow
def test_the_headline_claim_is_scoped_to_the_default_grouping():
    """The claim holds at k=5 and does NOT hold at k=6, so it is scoped, not absolute.

    Measured on the shipped corpus: at k=5 the three anticoagulation phrasings share
    a group (ARI against ground truth +0.43); at k=6 "blood thinner mistake" splits
    away from the other two (+0.36); at k=7 it splits again (+0.27). The app lets a
    user change the group count, so a reader who regroups can find the headline claim
    false — which is why the interface says so when the count is changed, and why the
    README and the post both say "the default grouping" rather than stating it flat.

    This test exists to stop the claim being quietly widened later.
    """
    pytest.importorskip("sentence_transformers")

    df = data_generator.build_frame(rows=100, seed=42)
    texts = df["Case_Summary"].tolist()
    matrix = engine.embed(texts)
    phrases = ["blood thinner mistake", "heparin administration error",
               "warfarin dose miscalculated"]

    def groups_at(k):
        labels, _ = engine.cluster(matrix, k=k)
        return {p: int(labels[next(i for i, t in enumerate(texts)
                                   if p in t.lower())]) for p in phrases}

    assert len(set(groups_at(5).values())) == 1, "the default grouping must hold"
    assert len(set(groups_at(6).values())) > 1, (
        "k=6 now keeps the trio together — the scoping language in the README and "
        "the post is out of date and should be revisited"
    )


@pytest.mark.slow
def test_anticoagulation_trio_shares_a_galaxy():
    """The claim the README and the LinkedIn post both make, pinned as a test.

    Three anticoagulation failures written with no content words in common must end up
    in the SAME cluster. Note what is deliberately *not* asserted: that they are each
    other's top-ranked neighbours. They are not — each summary is a full narrative, so
    the ranking is driven by more than the failure type. Co-assignment is the robust
    signal, and it is the only thing the copy is allowed to promise.
    """
    pytest.importorskip("sentence_transformers")

    df = data_generator.build_frame(rows=100, seed=42)
    texts = df["Case_Summary"].tolist()
    labels, _ = engine.cluster(engine.embed(texts), k=5)

    phrases = [
        "blood thinner mistake",
        "heparin administration error",
        "warfarin dose miscalculated",
    ]
    found = {}
    for phrase in phrases:
        idx = [i for i, t in enumerate(texts) if phrase in t.lower()]
        assert idx, f"corpus no longer contains {phrase!r}"
        found[phrase] = int(labels[idx[0]])

    assert len(set(found.values())) == 1, f"trio split across clusters: {found}"
