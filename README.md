# mmonfar. // Semantic M&M Failure Navigation Engine

**Turn a spreadsheet of Mortality & Morbidity minutes into a navigable 3D map of how your
system actually fails.**

Free-text M&M summaries are the richest safety data a hospital produces and the least
usable, because a keyword search cannot tell that *"blood thinner mistake"*,
*"heparin administration error"* and *"warfarin dose miscalculated"* are one failure mode.

This engine embeds every case with a local sentence-transformer, clusters the embeddings
with K-Means, projects them into three dimensions, and renders the result as an
interactive star field where each case is a star and each failure mode is a galaxy.

Everything runs on a laptop. No API keys. No cloud inference. No PHI leaves the building.

---

## Stack

| Layer | Choice | Why |
|---|---|---|
| Ingest | `pandas` + `openpyxl` | reads the spreadsheet the hospital already has |
| Semantics | `sentence-transformers` `all-MiniLM-L6-v2` | 384-d, ~90 MB, CPU-fast, permissive licence |
| Clustering | PCA(10) + re-normalise, then `sklearn.cluster.KMeans` (k=5) | transparent, deterministic, explainable in a meeting; the PCA step roughly doubles agreement with ground truth vs. clustering the raw 384-d vectors |
| Labelling | `sklearn` TF-IDF top-terms per cluster | names each galaxy from its own distinctive vocabulary |
| Projection | `umap-learn` if present, else `sklearn` PCA | 384-d → x, y, z |
| Render | Three.js r160 via CDN ESM | zero build step, single static folder |
| Workbench | `serve.py`, stdlib `http.server` only | drop a register on the page; nothing to install, ~200 lines an IG team can read |
| Brand | `app/assets/brand.css`, typefaces inlined as base64 | no font request; renders identically on an air-gapped machine |
| Promo | `matplotlib` + `imageio` | headless-renderable GIF/MP4 for social |
| Tests | `pytest` | geometry and payload contracts |

---

## Quick start

```bash
python -m venv venv
```

```bash
venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

```bash
python data_generator.py
```

```bash
python serve.py
```

That builds the demo galaxy if there isn't one and opens the workbench at
http://127.0.0.1:8000. From there you can drop your own register onto the page.

On macOS/Linux the activate step is `source venv/bin/activate`.

**Two ways to run it**

| | Workbench — `python serve.py` | Static — `python engine.py` |
|---|---|---|
| Upload a register from the page | yes | no |
| Needs Python running | yes | only to build |
| Output | live at `127.0.0.1:8000` | `app/standalone.html`, opens from disk |
| Good for | using it | sending it to someone, GitHub Pages |

The workbench binds to `127.0.0.1` — this machine only. Nothing is ever sent anywhere;
the upload endpoint writes to a local scratch folder and runs the same `engine.py`
pipeline the CLI runs.

---

## Execution scripts

| Command | What it does |
|---|---|
| `python data_generator.py` | writes `mock_mm_minutes.xlsx` — 100 synthetic M&M cases across 5 failure modes, with semantic traps baked in |
| `python engine.py` | the pipeline: ingest → embed → cluster → project → `app/data.json` (+ injection into `app/index.html`) |
| `python engine.py --input export.csv` | run it against your own de-identified extract (.xlsx/.xls/.xlsm/.csv/.tsv) |
| `python engine.py --input x.csv --text-col "What happened"` | name a column explicitly when the synonym table misses it |
| `python engine.py --clusters 7 --projection pca` | change k, or force a projection backend |
| `python marketing/generate_promo_viz.py --mp4` | render `marketing/visual_preview.gif` (and `.mp4`) for LinkedIn |
| `pytest -q` | full contract suite |
| `pytest -q -m "not slow"` | skip the test that needs the MiniLM weights |
| `python serve.py` | the workbench: serve the canvas and accept register uploads from the page |
| `python serve.py --port 9000 --no-browser` | same, on another port, without opening a browser |

First run of `engine.py` downloads the MiniLM weights (~90 MB) into the HuggingFace cache.
Every run after that is fully offline.

---

## Using your own data

Point `--input` at a spreadsheet or a delimited text file:

```bash
python engine.py --input "C:/exports/datix_q3.csv"
```

**Accepted formats:** `.xlsx`, `.xlsm`, `.xls`, `.csv`, `.tsv`, `.txt`. CSVs are sniffed
for delimiter (`,` `;` tab `|`) and read as UTF-8, then cp1252, then latin-1 — so an
export straight out of Excel on Windows opens without you converting anything.

**You do not need to rename your columns.** Headers are matched case- and
punctuation-insensitively against a synonym table, so `Datix Ref`, `Date of Incident`,
`Specialty`, `What Happened?` and `Grade of Harm` all map themselves:

| Wanted | Also accepts | If absent |
|---|---|---|
| `Case_ID` | ref, reference, case no, incident id, datix ref, number | auto-numbered `ROW-0001` |
| `Date` | incident date, event date, date of incident, reported date | left blank |
| `Department` | dept, specialty, service, directorate, division, ward | `Unspecified` |
| `Case_Summary` | summary, narrative, description, what happened, free text, details | **required** — falls back to the widest text column, or use `--text-col` |
| `Severity_Score` | severity, harm, grade of harm, harm level, impact, risk score | `3` |

`Severity_Score` accepts numbers **or** words — `None`/`Low`/`Moderate`/`Major`/
`Catastrophic` map to 1–5. (`None` means no harm, and is deliberately not treated as a
blank cell.)

When a column is named something the synonym table has never seen, name it yourself:

```bash
python engine.py --input minutes.xlsx --text-col "Discussion notes" --dept-col Directorate
```

`--id-col`, `--date-col` and `--severity-col` work the same way. If `k` is larger than
your row count, the engine reduces it rather than failing.

> **De-identify upstream.** The engine has no scrubbing stage and makes no claim to
> anonymise anything. `.gitignore` ignores every spreadsheet and CSV — including the
> synthetic mock — so nothing tabular can be committed by accident.

---

## Reading the galaxy

| What you see | What it means |
|---|---|
| A large galaxy | your most common failure mode |
| A tight, dense galaxy | the same failure repeating — systemic, and fixable |
| A diffuse galaxy | varied one-offs |
| A star drifting between two galaxies | a case that failed in two ways at once — usually the most instructive one in the room |
| A big star | high severity (`Severity_Score` 4–5) |

### Loading your own register from the page

With `python serve.py` running, drop an `.xlsx` or `.csv` onto the **Case register**
panel in the rail (or click it to browse). The page posts the file to the local
endpoint, the pipeline re-runs, and the galaxy rebuilds — usually a few seconds for a
few hundred cases. **Restore demo** puts the synthetic dataset back and deletes the
uploaded copy.

Opened as a bare `file://` page there is nothing to upload *to*, so the dropzone says so
rather than failing silently. Embedding needs the model, and the model lives in Python.

### The control rail

The left rail is the working surface, not decoration:

| Control | What it does |
|---|---|
| **Search** (`/` to focus, `Esc` to clear) | live filter across case text, ID and department |
| **Failure galaxies** | click one to isolate it; shows measured size and cohesion |
| **Minimum severity** | floor the view at 1–5 |
| **Departments** | click a chip to mute that department |
| **Geometry** | Amplified ⇄ True, with the standing caveat |
| **View** | auto-rotate, ambient starfield, depth fade, size-by-severity |
| **Export CSV** | downloads exactly what is on screen, cluster label included |
| **‹ toggle** | collapses the rail for a clean full-screen field |

Filters compose: search + severity floor + muted departments + isolated galaxy all apply
at once, and the footer reads `N of 100 cases` so you always know what you are looking at.
A case that has been filtered out cannot be hovered, clicked or exported.

**Canvas:** drag to orbit · scroll to zoom · hover to preview · click to pin a case ·
click a nearest-case row to fly to it · `R` to reset the view.

### Measured distance vs. drawn distance

The 3D layout is lossy — only ~14% of the embedding variance survives the squash to three
axes — and the spacing between galaxies is deliberately amplified so they read as distinct.
Neither of those is hidden in the docs; both are stated on the page itself.

**Geometry switch** (under the legend): flip between **Amplified** and **True**. True is
the raw projection with nothing added. The field morphs between the two so you can see
exactly which galaxies were pushed apart and by how much. The applied factor is named in
the caveat text and published as `meta.separation_gain`.

**The numbers are the measurement, the picture is the hint.** Every figure below is
computed in the full 384-dimensional space, before any projection or amplification, so no
layout choice can distort it:

| Where | What it tells you |
|---|---|
| Case card → *Closest cases* | the 3 most semantically similar cases, as cosine similarity |
| A neighbour flagged **different galaxy** | the clustering and the measurement disagree here — usually the most instructive case in the room |
| Legend → *cohesion* | mean within-galaxy similarity. High = one failure repeating. Low = a grab-bag |
| Legend → *nearest galaxy N similar* | how close this galaxy really is to its neighbour, regardless of the on-screen gap |
| Stats → *variance kept* | how much of the semantics survived the projection |

Click any nearest-case row and the camera flies to it, which makes the headline claim
checkable by hand. In the shipped mock corpus, *"blood thinner mistake"*, *"heparin
administration error"* and *"warfarin dose miscalculated"* share **no content words** and
land in the **same galaxy** — that grouping is what the engine is claiming, and
`tests/test_pipeline.py` asserts it.

Be careful not to over-read the *ranking*, though. Within a galaxy, "closest case" is
driven by the whole narrative, not only the failure type, so those three are not
necessarily each other's top-3 matches — in the shipped run they sit at ranks 17 and 33 of
99. **The grouping is the reliable signal; the ranking is a lead, not a verdict.**

See [SPECIFICATION.md](SPECIFICATION.md) §4.2 and §4.2.1.

Clusters are descriptive semantic neighbourhoods. They are not root causes, and K-Means
will always return exactly the *k* clusters you ask for, whether or not they exist. Treat
the map as a way to structure the conversation, not as a finding.

---

## Project layout

```
mm-k-means-bert/
├── SPECIFICATION.md          architecture, data contract, render guidelines
├── requirements.txt
├── data_generator.py         synthetic 100-case M&M corpus
├── engine.py                 the pipeline compiler
├── mock_mm_minutes.xlsx      generated (git-ignored)
├── serve.py                  local workbench (upload + rebuild from the page)
├── app/                      the static deliverable — deploy this folder alone
│   ├── index.html            the app; tracked, ships with an empty payload
│   ├── standalone.html       generated, payload inlined (git-ignored)
│   ├── data.json             generated payload (git-ignored)
│   └── assets/
│       ├── brand.css         canonical mmonfar. brand layer, fonts inlined
│       └── styles.css        app layer, composes brand tokens only
├── marketing/
│   ├── linkedin_post.txt
│   ├── generate_promo_viz.py
│   └── visual_preview.gif    generated (git-ignored)
└── tests/test_pipeline.py
```

`app/` is self-contained. Drop it on GitHub Pages, an intranet share, or a USB stick.

---

## Deploying the canvas to GitHub Pages

Push the repo, then in **Settings → Pages** select the `main` branch and the `/app`
folder (or push `app/` to a `gh-pages` branch). Nothing needs building.

---

## Where data can leak, and what stops it

The realistic accident is: someone runs this on a real register, then commits.

| Artefact | Contains case text? | Tracked? |
|---|---|---|
| `mock_mm_minutes.xlsx`, any `.xlsx`/`.csv` | yes | **no** — all tabular files are ignored |
| `.uploads/` (registers dropped on the page) | yes | **no** |
| `app/data.json` | yes | **no** |
| `app/standalone.html` (payload inlined) | yes | **no** |
| `app/index.html` | **no** — empty payload, fetches `data.json` | yes |
| `engine.py`, `serve.py`, `tests/`, `SPECIFICATION.md`, `README.md` | no | yes |

Two design decisions do the work:

1. **The pipeline never writes into a tracked file.** `engine.py` injects the payload
   into `app/standalone.html`, which is ignored, and leaves the tracked `app/index.html`
   template untouched. `tests/test_pipeline.py::test_tracked_template_carries_no_case_data`
   fails the build if that ever stops being true.
2. **Every tabular format is ignored, with no exceptions** — including the synthetic
   mock, which regenerates in about a second. A blanket rule is the only rule that
   survives a real extract being dropped into the folder in a hurry.

Tests, the spec, the README and the marketing copy stay tracked. They contain no case
data, and the tests are the evidence that the claims in the README hold — removing them
would remove the proof, not a risk.

Two things this does *not* do: it has no de-identification stage, so registers must be
de-identified upstream; and `--host` can bind the workbench beyond localhost, which
prints a warning and should not be used with real data.

---

## Licence & provenance

Synthetic data only — every case in `mock_mm_minutes.xlsx` is invented. `all-MiniLM-L6-v2`
is Apache-2.0. Three.js is MIT.

**mmonfar.** — Clinical Systems Architect
