# docs/sdd.md — master delivery contract

**Project:** mmonfar. // Semantic M&M Failure Navigation Engine
**Design document:** [`../SPECIFICATION.md`](../SPECIFICATION.md) — the *why* and *how*.
This file is the *what is done, what is next, and what counts as done*.

**Overall completion: 92%** · **Active phase:** Phase 6 — trust & refinement
**Last audited:** 2026-08-28 · **Tests:** 65 passing

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

### Open — Phase 6

| Task | Status | Acceptance criterion | Effort | Agent |
|---|---|---|---|---|
| Surface `best_k` in the UI | Pending | Method panel shows which k has held up best, from the run log | Medium | `coder` |
| Bulk-file from a filtered view | Pending | file every visible case under one label in a single action | Medium | `coder` |
| Stability ≥ 0.60 at the default k | Pending | either reach it, or state in the UI that it is not reached | High | `coder` |
| `must-link` / `cannot-link` honoured | Pending | recorded links constrain the next clustering | High | `coder` |
| Drift report | Pending | flag when a carried-over name's group has moved materially | Medium | `coder` |

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
