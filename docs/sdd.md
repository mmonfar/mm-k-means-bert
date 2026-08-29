# docs/sdd.md — master delivery contract

**Project:** mmonfar. // Semantic M&M Failure Navigation Engine
**Design document:** [`../SPECIFICATION.md`](../SPECIFICATION.md) — the *why* and *how*.
This file is the *what is done, what is next, and what counts as done*.

**Overall completion: 90%** (28 of 31 acceptance criteria across Phases 1–7 met) · **Active phase:** Phase 6 — trust & refinement
**Last audited:** 2026-08-29 · **Tests:** 84 passing

---

## How to read this

A task is `Done` only when its **acceptance criterion** is met and pinned by a
test. "It works when I try it" is not done. Percentages are the share of
acceptance criteria met, not a feeling.

---

## Phase 1–4 — shipped

| Task | Status | Acceptance criterion | Pinned by |
|---|---|---|---|
| Synthetic corpus, 100 cases, 5 failure modes, semantic traps | Done | trio of anticoagulation phrasings present, no shared content words | `test_semantic_traps_are_present` |
| Ingest xlsx/xls/xlsm/csv/tsv/txt | Done | reads all six; sniffs delimiter and encoding | `test_reads_every_supported_format` |
| Header synonym resolution | Done | a Datix-style export maps with no flags | `test_maps_real_world_headers_without_flags` |
| Embed → cluster → project → payload | Done | payload validates and round-trips JSON | `test_payload_shape_and_bounds` |
| Amplified vs true geometry, both published | Done | both coordinate sets differ and are declared | `test_payload_carries_both_geometries` |
| Measured cosine neighbours | Done | excludes self, ranked descending | `test_nearest_neighbours_are_measured_and_ranked` |
| Three.js canvas on the brand layer | Done | boots with zero page errors | `test_the_app_actually_boots_with_no_javascript_errors` |
| Local workbench: upload, regroup, rename, file | Done | endpoints return 200 and rebuild | manual + `serve.py` routes |
| Apache-2.0 licence and NOTICE | Done | LICENSE and NOTICE tracked | audit |

## Phase 5 — honest naming

| Task | Status | Acceptance criterion | Pinned by |
|---|---|---|---|
| Coverage-gated keyword labels | Done | a naming term covers ≥20% of its group, word-boundary matched | `test_label_terms_must_actually_describe_their_group` |
| Numbered fallback | Done | fewer than two qualifying terms → `Group N` | `test_unnamed_groups_are_numbered_not_invented` |
| Semantic exemplar per group | Done | nearest case to centroid, drawn from its own group | `test_every_group_carries_a_semantic_exemplar` |
| Optional local model naming | Done | opt-in, cohesion-gated, validated, de-duplicated | `test_generated_labels_are_validated_not_trusted` |

## Phase 6 — trust & refinement *(active)*

| Task | Status | Acceptance criterion | Pinned by |
|---|---|---|---|
| QC: neighbour agreement vs chance | Done | chance baseline accounts for uneven group sizes | `test_the_chance_baseline_accounts_for_uneven_groups` |
| QC: stability under resampling | Done | reported per run | `test_a_run_is_recorded_with_a_reason_for_every_case` |
| QC: versus shuffled null | Done | refuses to endorse noise | `test_quality_is_not_fooled_by_noise` |
| Per-case justification | Done | every case has a reason; borderline cases flagged | `test_a_borderline_case_is_reported_as_borderline` |
| Run log with per-case reasons | Done | one row per case per run, no case text | `test_the_run_log_stores_no_case_text` |
| `best_k` from recorded evidence | Done | ranks on stability then agreement | `test_best_k_is_decided_on_recorded_evidence` |
| Human names survive re-runs | Done | exact by fingerprint, fuzzy by centroid ≥0.92 | `test_a_name_carries_to_a_shifted_group_and_says_so` |
| Taxonomy classifies unseen cases | Done | 78% correct on a 30-case holdout, declines when unsure | `test_a_learned_label_classifies_a_case_it_has_never_seen` |
| Per-case filing overrides prediction | Done | a hand-filed case is never overruled | `filed_cases` in `engine.apply_house_taxonomy` |
| Surface `best_k` in the UI | Done | Method panel shows which k has held up best, read from the run log via `engine.recorded_best_k()` → payload `meta.best_k`; says "not enough run history yet" rather than inventing a k when `feedback.best_k` returns `None` | `test_engine_recorded_best_k_reads_the_run_log`, `test_payload_carries_best_k_or_none`, `test_method_panel_surfaces_the_best_performing_k_from_the_run_log` |
| Bulk-file from a filtered view | Done | new `PATCH /api/cases` route files every visible case under one label in a single action, reusing the single-case filing semantics so a bulk-filed case is a human filing and is never overruled by prediction; empty selection rejected | `test_bulk_filing_files_every_visible_case_under_one_label` |
| Stability ≥ 0.60 at the default k | Done (fallback branch) | **The bar is not reached.** The criterion was "either reach it, or state in the UI that it is not reached," and it closes on the second branch: the UI states the position from the measured payload value against a `STABILITY_BAR` constant, no hardcoded figure. Stability remains 0.57 against a 0.60 bar; the number has not moved | `test_method_panel_states_plainly_that_the_stability_bar_is_not_met`, `test_method_panel_states_plainly_when_the_stability_bar_is_met` |
| `must-link` / `cannot-link` honoured | Done | `engine.cluster()` now takes `texts` and detours into a lite in-module COP-KMeans path when recorded links match this run's cases; must-link is honoured by construction via union-find, cannot-link is enforced during assignment; unsatisfiable sets degrade with a report rather than crashing (contradictory and violated counted and printed, surfaced in payload `meta.constraints`); with no recorded links the clustering is byte-identical to before | `test_no_recorded_links_leaves_clustering_unchanged`, `test_a_must_link_pair_lands_in_the_same_group`, `test_a_cannot_link_pair_is_split_apart`, `test_unsatisfiable_constraints_are_reported_not_crashed`, `test_a_cannot_link_clique_bigger_than_k_degrades_without_crashing` |
| Drift report | Done | carried-over human names are flagged when their group has moved materially (`engine.DRIFT_MATERIAL_MAX = 0.96`, surfaced as `clusters[i].label_drift` and payload `meta.name_drift`, plus a console line); the 0.96 threshold is measured, not invented — 40 resampling trials on the shipped corpus relating centroid similarity to real case-membership overlap (correlation 0.92), below 0.96 typically less than half the group's membership survives even though the name carries; the derivation table is recorded in `engine.py` above the constant | `test_a_carried_name_below_the_material_drift_threshold_is_flagged` |

### Phase 6 — closed since last audit

| Task | Status | Acceptance criterion | Pinned by |
|---|---|---|---|
| User-chosen confidence level | Done | three named presets (Exploring / Preparing / Acting on it) live in the rail under Failure galaxies, not behind Settings — the discoverability failure the owner reported ("I knew it was there and took me a while to understand"). The chosen level's plain-English consequence is always visible as text, not only on hover; a guess (a label the recorded 'preparing' standard would have withheld) is marked distinctly from a confident label via a dedicated `.by-guess` style, never identically. The setting persists via localStorage (wrapped in try/catch) and never enters a request body or the stored record; "Preparing" still reproduces today's exact behaviour, so the pinned 78% holdout claim is unaffected | `test_confidence_preset_renders_in_the_rail_with_plain_english_text`, `test_a_guess_is_marked_distinctly_from_a_confident_label`, `test_confidence_level_persists_and_never_reaches_the_stored_record` |
| Replace `window.prompt` filing dialogs | Done | the single-case and bulk-file dialogs are an in-app modal that states what is in scope (the count and that it is the visible/filtered set), offers existing labels as clickable choices, states the filing-is-final consequence in plain language before commit, and offers Cancel/Escape/backdrop-click as an obvious way out; the empty-selection guard is unchanged. The rename dialog at :847 was left on `prompt()` — out of scope per the brief, which prioritised the filing dialogs | `test_filing_modal_offers_existing_labels_and_files_the_visible_set` |
| Method panel layout | Done | First pass widened `#method-pop` to `min(560px, 100vw - 24px)` and packed two label/value pairs per row — measured wrong on the live app: the longest label ("Best-performing group count") claimed ~202px of the panel, leaving each value column ~90px, and `overflow-wrap: break-word` split ordinary words ("automatically") mid-syllable at that width, the exact defect the task was raised to fix. Corrected to one pair per row with the label column capped at `minmax(0, 180px)`, giving each value ~320px at 560px panel width (measured live, both at 1280px and down to 420px viewport) — no word breaks, no scroll. A separate defect found in the same review, the three confidence-preset buttons splitting 2-then-1 across rows on a narrow rail, is fixed by giving `.conf-seg` a fixed 3-column grid instead of `flex-wrap`, so all three always share one row (checked at 800/1000/1440px) | `test_method_panel_value_column_is_wide_enough_not_to_break_words`, `test_confidence_presets_never_split_two_and_one_across_rows` |

### Open — Phase 6

No open rows at last audit.

## Standing — periodic tooling review

Not a phase; a recurring check. Authorised 2026-08-28. Criteria and the watch
list are in [`../CLAUDE.md`](../CLAUDE.md). A proposal without a benchmark is
rejected at audit.

| Candidate | Why it would matter | Bar to clear |
|---|---|---|
| Alternative embedding model | stability is 0.57 against a 0.60 bar — this is the one number worth moving | beats 0.70 / 0.57 on the shipped corpus |
| `umap-learn` when it installs cleanly | engine already prefers it, falls back to PCA | no new build toolchain |
| Vendored Three.js | removes the last runtime network dependency | app/ stays under a few hundred KB |

## Phase 7 — not started

| Task | Acceptance criterion | Effort |
|---|---|---|
| Multi-register history | compare quarters; show what grew | High |
| Per-department view | the same map filtered to one directorate, with its own KPIs | Medium |
| Printable governance pack | one page per group: exemplar, count, trend, actions | Medium |

---

## Standing constraints

Any task that violates one of these is rejected at audit regardless of whether it
works. Full reasoning in [`../CLAUDE.md`](../CLAUDE.md).

1. Nothing derived from a register is committed.
2. The pipeline never writes into a tracked file.
3. Every number in the docs is measured and pinned by a test.
4. The default path imports neither `torch` nor `transformers`.
5. UI and doc language must match the QC verdict, not flatter it.
6. The corpus stays synthetic.

## Current quality position — do not overstate

Measured at k=5 on the shipped corpus:

| Check | Value | Passing? |
|---|---|---|
| Neighbour agreement | 0.70 vs 0.21 at chance (3.3×) | yes |
| Stability | 0.57 | no (bar 0.60) |
| Versus shuffled | 0.124 vs 0.098 | no (bar 1.5×) |

**Verdict: 1 of 3 — "weak; treat the grouping as a prompt, not a finding."**

The grouping is clearly not random and is not reliable enough to settle a
question alone. Every claim in the README already reflects this. Do not let it
drift upward in tone without the numbers moving first.

Closing the five Phase 6 rows above — including the constrained clustering and
the drift report — did not move any of these numbers. `must-link`/`cannot-link`
and drift labelling change how groups are formed and named after the fact;
they were not re-run through `quality.py`, and nothing in this session's work
claims to have improved neighbour agreement, stability, or the shuffled-null
margin. The verdict stays 1 of 3.
