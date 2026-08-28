# SPECIFICATION.md
## mmonfar. // Semantic M&M Failure Navigation Engine

**Owner:** mmonfar. — Clinical Systems Architect
**Status:** v1.0 (SDD baseline)
**Class:** Cheap, open-access, fully-local clinical analytics. No API keys. No cloud inference. No PHI egress.

---

## 1. Problem Statement

Mortality & Morbidity (M&M) meeting minutes are the highest-signal safety artefact a hospital produces, and
they are almost universally stored as free-text cells in a spreadsheet. Free-text is not queryable by
keyword: `"blood thinner mistake"`, `"heparin administration error"` and `"warfarin dose miscalculation"`
are the *same failure mode* and share *zero* keywords. Conventional pivot tables therefore under-count the
dominant systemic failure in most departments.

This engine replaces lexical matching with **semantic matching**, then renders the resulting semantic
topology as a navigable 3D star field so that a clinician — not a data scientist — can *see* where the
system is breaking.

---

## 2. Data Pipeline Flow

```
┌──────────────────────────────────────────────┐
│ Local tabular file                           │
│ .xlsx .xlsm .xls .csv .tsv .txt              │
│ delimiter + encoding sniffed for delimited   │
│ headers resolved against a synonym table,    │
│ overridable per column from the CLI          │
│ -> Case_ID | Date | Department |             │
│    Case_Summary | Severity_Score             │
└──────────┬───────────────────────────────────┘
           │  Case_Summary column, list[str], n = 100
           ▼
┌──────────────────────────────────────────────┐
│ Embedding Matrix                             │
│ sentence-transformers 'all-MiniLM-L6-v2'     │
│ runs 100% locally on CPU                     │
│ output: float32 ndarray  shape (n, 384)      │
│ normalize_embeddings=True -> unit hypersphere│
└──────────┬───────────────────────────────────┘
           │  X in R^(n x 384)
           ▼
┌──────────────────────────────────────────────┐
│ Topic Clusters                               │
│ denoise: PCA(10) + re-normalise (clustering  │
│          only — measurably better than       │
│          clustering the raw 384-d vectors)   │
│ sklearn.cluster.KMeans(n_clusters=5,         │
│                        n_init=25,            │
│                        random_state=42)      │
│ output: labels in {0..4}^n                   │
│ + auto-labelled cluster names via TF-IDF     │
│   top-term extraction per cluster            │
└──────────┬───────────────────────────────────┘
           │  labels, centroids
           ▼
┌──────────────────────────────────────────────┐
│ Spatial Projection                           │
│ primary : umap-learn (n_components=3)        │
│ fallback: sklearn.decomposition.PCA(3)       │
│ -> rescale to a cube of edge +/-100 units    │
│    = the TRUE geometry (tx, ty, tz)          │
│ -> per-cluster centroid repulsion pass       │
│    = the AMPLIFIED geometry (x, y, z)        │
│ + cosine nearest neighbours from the full    │
│   384-d space (measured, projection-free)    │
└──────────┬───────────────────────────────────┘
           │  coords in R^(n x 3)
           ▼
┌──────────────────────────────────────────────┐
│ JSON Payload  ->  app/data.json              │
│ { meta, departments[], clusters[], points[] }│
│ + app/standalone.html: the same payload      │
│   inlined into a COPY of the page, so it     │
│   works over file:// with no server          │
│ Neither is tracked: both carry case text.    │
│ app/index.html is never written to.          │
└──────────┬───────────────────────────────────┘
           ▼
┌──────────────────────────────────────────────┐
│ Independent Three.js Web Canvas              │
│ app/index.html — zero build step, CDN ESM    │
│ OrbitControls + Raycaster + HUD overlay      │
└──────────────────────────────────────────────┘
```

**Hard rule:** the web canvas is *independent*. It never calls Python at runtime. `engine.py` is a
batch compiler; `app/` is a static deliverable that can be dropped on GitHub Pages, an intranet share,
or opened directly from disk.

---

## 3. Project Directory Tree

```
mm-k-means-bert/
├── .gitignore
├── SPECIFICATION.md            # this document
├── README.md                   # setup, stack, execution scripts
├── requirements.txt
├── data_generator.py           # Phase 3.1 — mock clinical corpus
├── engine.py                   # Phase 3.2 — the pipeline compiler
├── mock_mm_minutes.xlsx        # generated, git-ignored
├── serve.py                    # local workbench: static server + ingest endpoint
├── app/
│   ├── index.html              # Phase 3.3 — the app; tracked, empty payload
│   ├── standalone.html         # generated, payload inlined, git-ignored
│   ├── data.json               # generated, git-ignored
│   └── assets/
│       ├── brand.css           # canonical mmonfar. brand layer, fonts inlined
│       └── styles.css          # app layer, composes brand tokens only
├── marketing/                  # git-ignored in full
│   ├── linkedin_post.txt
│   ├── generate_promo_viz.py
│   └── visual_preview.gif
└── tests/
    └── test_pipeline.py        # pytest contract tests
```

---

## 4. 3D Cartesian Space Rendering Guidelines

### 4.1 Coordinate contract

| Axis | Range | Meaning |
|------|-------|---------|
| `x`  | −100 … +100 | UMAP/PCA component 1, amplified view |
| `y`  | −100 … +100 | UMAP/PCA component 2, amplified view |
| `z`  | −100 … +100 | UMAP/PCA component 3, amplified view |
| `tx`, `ty`, `tz` | −100 … +100 | the same three components with **no** separation transform applied |

Right-handed Y-up world space (Three.js default). Origin `(0,0,0)` is the semantic barycentre of the
entire corpus — the "centre of the hospital's failure universe". Camera home is `(150, 90, 190)`
looking at origin, giving a three-quarter view where all five galaxies are simultaneously visible.

### 4.2 Cluster separation

Raw manifold projections routinely place two clusters within a few units of each other, which reads as
a single blob on screen. After rescaling, the engine runs a **centroid repulsion pass**:

1. Compute each cluster centroid `c_k`.
2. Push every centroid radially outward from the global barycentre by factor `SEPARATION_GAIN = 1.65`.
3. Translate each point rigidly with its centroid — *intra*-cluster geometry (the real semantic
   structure) is preserved exactly; only *inter*-cluster whitespace is manufactured.
4. Assert minimum centroid pair distance ≥ 45 world units; escalate the gain up to 3 times if violated.

This is a **presentation transform, not an analytical one**. Declaring it in the payload is not
sufficient — a clinical governance audience reads the screen, not the JSON — so the disclosure is
enforced in three places:

1. **The payload** carries `meta.separation_gain` and `meta.true_geometry_note`.
2. **Every point carries both geometries.** `x, y, z` is the amplified view; `tx, ty, tz` is the raw
   projection with nothing added.
3. **The canvas exposes a Geometry switch** (Amplified / True) directly beneath the legend, with a
   standing caveat paragraph that names the applied gain in plain language. Switching morphs the
   point cloud between the two coordinate sets over ~0.5 s, so the reader sees exactly which galaxies
   were pushed apart and by how much rather than being asked to take it on trust.

### 4.2.1 Reporting measured semantic distance

Because the 3-d layout is lossy (typically ~14% of embedding variance retained under PCA), on-screen
proximity is treated throughout as a *hint*, and the measurement is reported separately in numbers
computed in the full 384-d space, before any projection or amplification:

| Where | Field | Meaning |
|-------|-------|---------|
| Case card | `points[i].nearest` | the 3 most similar cases, with cosine similarity to 2 d.p. |
| Case card | `nn-cross` flag | marks a near neighbour that K-Means filed in a *different* cluster — the point where the clustering and the measurement disagree, and usually the most instructive case in the room |
| Legend | `clusters[k].cohesion` | mean pairwise cosine similarity within the galaxy. High = one failure repeating; low = a grab-bag |
| Legend | `clusters[k].nearest_similarity` | cosine similarity between this galaxy's centroid and its closest neighbouring galaxy — genuinely adjacent galaxies say so numerically, whatever the on-screen gap suggests |
| Stats strip | `meta.variance_retained` | share of embedding variance surviving the squash to 3 axes |

Clicking a nearest-neighbour row flies the camera to that case, which makes the headline claim
falsifiable by hand.

**What the engine claims, and what it does not.** On the shipped mock corpus, *"blood thinner
mistake"*, *"heparin administration error"* and *"warfarin dose miscalculated"* share no content words
and are assigned to the same cluster. That co-assignment is the claim, and it is asserted in
`tests/test_pipeline.py::test_anticoagulation_trio_shares_a_galaxy`.

The engine does **not** claim those cases will be each other's top-ranked neighbours. Each summary is
a full clinical narrative, so cosine similarity is driven by the whole account rather than by the
failure type alone; in the shipped run the other two anticoagulation cases rank 17th and 33rd of 99
against the first (0.35 and 0.28, versus a corpus mean of 0.26). Cluster co-assignment is the robust
signal; the neighbour ranking is a lead for a human to follow, not a verdict. Any copy written about
this tool must respect that distinction.

### 4.3 Visual depth

- **Fog:** `THREE.FogExp2(0x1E2A32, 0.0022)` — far stars desaturate into the canvas colour, producing
  honest depth cueing without a skybox.
- **Size attenuation:** `PointsMaterial.sizeAttenuation = true`; near stars ~14 px, far stars ~3 px.
- **Glow map:** each star is a radial-gradient sprite generated on a 128×128 canvas at runtime
  (no external texture files → single-file portability).
- **Severity → radius:** point size scales `6 + Severity_Score * 2.2`, so critical cases are physically
  larger stars. Severity is 1–5.
- **Additive blending** with `depthWrite = false` so overlapping stars bloom rather than z-fight.
- **Ambient starfield:** 1,400 dim cream micro-points at radius 400–900 provide parallax reference so
  orbital rotation is legible.

### 4.4 Colour

Cluster colour is categorical, never sequential. Palette is a brand-extended, high-contrast set validated
against the `#1E2A32` canvas:

| Cluster | Hex | Role |
|---------|-----|------|
| 0 | `#00B3B3` | brand teal, lifted for dark-canvas legibility |
| 1 | `#FF8A5B` | warm coral |
| 2 | `#9D8CFF` | violet |
| 3 | `#F2C14E` | amber |
| 4 | `#5BD1A0` | mint |

The palette is deliberately not red/green paired, and each hue differs in luminance as well as chroma, so
it survives deuteranopia and monochrome print.

### 4.5 Interaction contract

| Input | Behaviour |
|-------|-----------|
| Left-drag | orbital rotation, damped (`enableDamping`, factor 0.06) |
| Wheel / pinch | smooth dolly zoom, clamped `minDistance 40`, `maxDistance 700` |
| Hover | raycast (threshold 5 units) → star highlights, HUD shows summary |
| Click | pins the case card until dismissed or another star is clicked |
| Legend click | isolates a single galaxy (others drop to low opacity) |
| Geometry switch | morphs the field between the amplified and true coordinate sets |
| Nearest-case click | flies the camera to that case and pins it |
| `R` key | returns camera to home position |

### 4.6 Accessibility & responsiveness

- Canvas resizes on `resize`; HUD collapses to a bottom sheet below 820 px.
- All HUD text meets WCAG AA on `#1E2A32`.
- `prefers-reduced-motion` disables the idle auto-rotation.
- The full case list is also emitted into the DOM as a visually-hidden `<ul>` so the payload is
  reachable by screen readers and by Ctrl-F.

---

## 4.6.1 Naming a group honestly

Naming clusters from top-TF-IDF terms is the obvious approach and it was wrong here in a
way worth recording, because the failure is instructive: **it is keyword counting, inside
a tool built to show that keyword counting misleads.**

Measured on the shipped corpus at k=5, the most distinctive term in each cluster appears
in this share of that cluster's cases (word-boundary matched):

| Cluster | Best term | Coverage |
|---|---|---|
| 0 | transfer | 12% |
| 1 | ct | 29% |
| 2 | prescribed | 16% |
| 3 | waited | 23% |
| 4 | op | 21% |

A three-word name built from those describes, at best, a quarter of what it points at. An
earlier build named an 11-case cluster "Theatre / Waited / Delay" from terms appearing in
3 cases each — and two of those cases used *theatre* to mean an operating room rather than
a queue for one, which is the exact lexical collision the product exists to defeat.

The rules that follow from that:

1. A term must clear `LABEL_COVERAGE_MIN` (0.20) of its cluster, matched on **word
   boundaries** — substring matching put `ct` inside *contact* and *reflect* and silently
   inflated coverage for precisely the short abbreviations most likely to be chosen.
2. Terms containing digits are rejected; `1400` and `14` are timestamps, not failure modes.
3. Scoring is cluster-mean TF-IDF **minus** corpus mean, and each term is assigned to only
   one cluster, so a term common to the whole register cannot name several groups at once.
4. Fewer than two surviving terms means the group is numbered — `Group 3` — rather than
   given a label it has not earned.
5. Every cluster carries an `exemplar`: the case nearest its centroid **in the full
   embedding space**. That is a real case chosen semantically, and it identifies a mixed
   group far better than any three words can. It is what the interface shows on hover.
6. `label_coverage` is published in the payload, and the interface states it, so a name
   cannot imply more completeness than it has.

---

## 4.6.2 Naming authority

Three sources may name a group. They are ranked, and the winner is recorded in
`clusters[].label_source` so the interface never presents a guess and a judgement
as though they carry equal weight.

| Rank | `label_source` | Origin |
|---|---|---|
| 1 | `human`, `human_carried` | a person renamed it; stored in `feedback.db` |
| 2 | `model` | a local instruct model read the group's central cases |
| 3 | `terms` | TF-IDF terms clearing 20% coverage |
| 4 | `numbered` | nothing earned a name |

**The model tier is gated on measured cohesion (≥ 0.25), not on model size.**
Evidence, from the demo register:

| Group | Cohesion | Qwen2.5-0.5B said | Qwen2.5-1.5B said | Truth |
|---|---|---|---|---|
| 1 | 0.35 | Delays reaching theatre | Documentation oversight | CT reporting delays |
| 3 | 0.31 | Delayed reporting | Time delays | delays to treatment |
| 2 | 0.28 | Errors in medication administration | Dosage calculation errors | medication errors |
| 0 | 0.24 | Errors in documentation | Communication failures | handover + equipment |
| 4 | 0.23 | Delays reaching theatre | Inadequate communication | surgical complications |

Tripling the parameter count did not help, and both models gave two different
groups the same name. A group with nothing in common has no name; a fluent model
will supply one regardless. Generated names are therefore validated, de-duplicated
across groups, gated on cohesion, and always shown next to the exemplar so a
reader can check the claim in one glance.

## 4.6.3 The feedback store

`feedback.db` (SQLite, stdlib, git-ignored) records human judgements:
`group_names`, `case_labels`, `links` (same / different), and an append-only
`events` audit trail carrying author and timestamp.

**No case text is stored.** A case is keyed by `sha256(normalised narrative)[:16]`
— sufficient to recognise the same case in a later export, insufficient to
reconstruct one. This is what allows a persistent store to exist at all under
§4.8's egress rules.

Name persistence across runs, since cluster ids are not stable:

1. **Fingerprint** — `sha256` of the group's sorted case keys. Exact match
   restores the name as `human`.
2. **Centroid** — cosine similarity against the stored centroid. Above
   `CARRY_OVER_MIN` (0.92) the name is restored as `human_carried`, with the
   similarity published, because it is an inference. Below it, nothing is
   restored: mislabelling a drifted group with last quarter's name is a worse
   failure than asking again.
3. One stored name can claim only one group per run.

**Explicitly not self-training.** No model output is ever recycled as truth. The
intended endpoint is supervised: once `training_set()` reports enough examples per
label, a register can be classified into the hospital's own agreed taxonomy
instead of re-clustered — a taxonomy owned by the people who wrote it.

---

## 4.7 Ingest contract

Only `Case_Summary` is genuinely required; every other column degrades to a documented
default (`ROW-nnnn` ids, `NaT` dates, `Unspecified` department, severity `3`). This is
deliberate: a department column is a nice-to-have and a narrative column is the entire
product, so the tool should run on whatever a governance lead can export today rather
than on a schema they have to build first.

Resolution order per column: explicit CLI override → exact name → synonym table → (for
`Case_Summary` only) the widest free-text column in the file → default. Every mapping and
every defaulted column is printed at ingest, so the operator can see what was assumed.

`Severity_Score` accepts integers or harm words. Note that pandas treats the literal
string `"None"` as a missing value by default; in a harm column `"None"` means grade 1,
so the reader overrides the NA list rather than silently defaulting those rows to 3.

---

## 4.8 Data-egress contract

The failure this project has to design against is not an attacker; it is a clinician in a
hurry committing their own register. Accordingly:

- **No tracked file is ever written by the pipeline.** The payload is injected into
  `app/standalone.html` (ignored), never into the tracked `app/index.html` template. The
  template ships with an empty payload and fetches `data.json` at runtime. A test asserts
  this and fails the build if it regresses.
- **Every tabular format is ignored by git**, including the synthetic mock. The mock is
  deterministic and regenerates in about a second, so tracking it buys nothing and the
  blanket rule cannot be defeated by a badly-named file.
- **`serve.py` binds to `127.0.0.1` by default.** Uploads are written to `.uploads/`
  (ignored), capped at 32 MB, restricted to extensions the engine can read, and named
  from a fixed stem so a crafted filename cannot traverse out of the directory. Binding
  beyond localhost is possible via `--host` and prints a warning.
- **No outbound request exists in the running application.** The only network access in
  the whole project is the one-off model download on first run, and the Three.js CDN
  import in the page. `app/assets/brand.css` inlines both typefaces, so the interface
  itself makes no font request and renders identically on an air-gapped machine.

---

## 5. Non-Goals

- No PHI. The generator produces synthetic text only; real deployments must de-identify upstream.
- No causal inference. Clusters are *descriptive* semantic neighbourhoods, not root-cause findings.
- No server. If it needs a server, it is out of scope for v1.
