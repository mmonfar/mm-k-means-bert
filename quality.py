"""
quality.py
mmonfar. // Semantic M&M Failure Navigation Engine

Is this grouping real, or is it random? Answered with numbers, cheaply.

An unlabelled clustering cannot be checked against a right answer — there isn't
one. But three things can be measured without any ground truth, and together
they answer the question a governance lead actually has, which is "why should I
believe any of this?"

  1. NEIGHBOUR AGREEMENT — of every case, does its single most similar case sit
     in the same group? Under random assignment this would happen about 1/k of
     the time. This is the most legible check there is: no statistics needed to
     understand "84% of cases have their closest match in the same group, and
     chance would be 17%".

  2. STABILITY — re-cluster on repeated random subsamples and count how often
     two cases that were grouped together stay together. Real structure survives
     resampling; a partition of noise does not. This is the standard test for
     whether clusters are an artefact of the algorithm.

  3. SEPARATION vs A NULL — the same pipeline run on shuffled data, which has
     the same shape and no structure. If the real silhouette is not clearly
     above the shuffled one, the grouping is decoration.

And per case, a plain-language reason it sits where it does: how close it is to
its own group, how close to the next one, and whether its nearest neighbours
agree. A borderline case should look borderline, not authoritative.

Everything here is numpy over embeddings already computed for the map. No model,
no new dependency, about a second for a hundred cases.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Enough resamples to be stable to ~0.01 without making the run feel slow.
STABILITY_ROUNDS = 12
STABILITY_FRACTION = 0.8

# A case closer than this to the next group along is genuinely on a border, and
# the interface should say so rather than implying a clean assignment.
BORDERLINE_MARGIN = 0.02


def _unit(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def neighbour_agreement(matrix: np.ndarray, labels: np.ndarray) -> dict:
    """How often a case's most similar case is in its own group.

    The single most interpretable check available: it needs no statistics to
    read, and it is computed in the full embedding space, so the projection
    cannot flatter it.
    """
    unit = _unit(matrix)
    sims = unit @ unit.T
    np.fill_diagonal(sims, -np.inf)
    nearest = sims.argmax(axis=1)
    agree = float((labels[nearest] == labels).mean())

    # Chance is not 1/k when groups differ in size: a case is more likely to
    # land beside a member of a big group. The honest baseline is the sum of
    # squared group shares.
    shares = np.bincount(labels, minlength=labels.max() + 1) / len(labels)
    chance = float((shares ** 2).sum())

    return {
        "agreement": round(agree, 4),
        "chance": round(chance, 4),
        "lift": round(agree / chance, 2) if chance else None,
    }


def stability(matrix: np.ndarray, k: int, seed: int = 42) -> dict:
    """Do the same cases keep company when the data is resampled?

    For each of several random 80% subsamples, re-cluster and record whether
    each pair of cases that shared a group in the full run shares one again.
    Structure survives; an arbitrary partition does not.
    """
    from engine import denoise

    rng = np.random.default_rng(seed)
    n = len(matrix)
    if n < k * 3:
        return {"stability": None, "rounds": 0}

    base = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(
        denoise(matrix, seed=seed))

    together = np.zeros((n, n), dtype=float)
    seen = np.zeros((n, n), dtype=float)

    for r in range(STABILITY_ROUNDS):
        idx = rng.choice(n, size=int(n * STABILITY_FRACTION), replace=False)
        sub = denoise(matrix[idx], seed=seed + r)
        sub_labels = KMeans(n_clusters=k, n_init=5,
                            random_state=seed + r).fit_predict(sub)

        same = (sub_labels[:, None] == sub_labels[None, :]).astype(float)
        grid = np.ix_(idx, idx)
        together[grid] += same
        seen[grid] += 1.0

    pair_base = (base[:, None] == base[None, :])
    mask = (seen > 0) & pair_base
    np.fill_diagonal(mask, False)
    if not mask.any():
        return {"stability": None, "rounds": STABILITY_ROUNDS}

    kept = (together[mask] / seen[mask]).mean()
    return {"stability": round(float(kept), 4), "rounds": STABILITY_ROUNDS}


def against_shuffled(matrix: np.ndarray, k: int, seed: int = 42) -> dict:
    """The same pipeline on data with the structure destroyed.

    Each feature column is shuffled independently, which keeps every marginal
    distribution and removes the relationships between them. If the real
    silhouette does not clearly beat this, there is nothing to see.
    """
    from engine import denoise

    rng = np.random.default_rng(seed)
    real = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(
        denoise(matrix, seed=seed))
    real_score = float(silhouette_score(denoise(matrix, seed=seed), real))

    noise = matrix.copy()
    for col in range(noise.shape[1]):
        rng.shuffle(noise[:, col])
    null = denoise(noise, seed=seed)
    null_score = float(silhouette_score(
        null, KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(null)))

    return {
        "silhouette": round(real_score, 4),
        "shuffled_silhouette": round(null_score, 4),
        "beats_noise": bool(real_score > null_score * 1.5),
    }


def per_case(matrix: np.ndarray, labels: np.ndarray) -> list[dict]:
    """Why each case sits where it does, in numbers a person can check.

    Returns, per case: similarity to its own group's centre, similarity to the
    nearest other group, the margin between them, how many of its three closest
    cases share its group, and a one-line reason.
    """
    unit = _unit(matrix)
    k = int(labels.max()) + 1
    centroids = _unit(np.stack([
        unit[labels == c].mean(axis=0) if (labels == c).any()
        else np.zeros(unit.shape[1]) for c in range(k)
    ]))
    sims = unit @ centroids.T

    neighbour_sims = unit @ unit.T
    np.fill_diagonal(neighbour_sims, -np.inf)
    top3 = np.argsort(-neighbour_sims, axis=1)[:, :3]

    out = []
    for i, row in enumerate(sims):
        own = float(row[labels[i]])
        others = np.delete(row, labels[i])
        rival = float(others.max()) if len(others) else -1.0
        rival_id = int(np.argsort(-row)[1]) if k > 1 else None
        margin = own - rival
        agree = int((labels[top3[i]] == labels[i]).sum())

        if margin < BORDERLINE_MARGIN:
            reason = (f"borderline — nearly as close to group {rival_id} "
                      f"({rival:.2f}) as to its own ({own:.2f})")
        elif agree == 3:
            reason = (f"all three of its closest cases are in this group; "
                      f"sits {own:.2f} from its centre")
        elif agree == 0:
            reason = (f"none of its three closest cases are in this group — "
                      f"placed on overall similarity ({own:.2f}), worth a look")
        else:
            reason = (f"{agree} of its three closest cases are in this group; "
                      f"{own:.2f} from its centre, {margin:.2f} clear of the next")

        out.append({
            "to_own": round(own, 4),
            "to_next": round(rival, 4),
            "next_group": rival_id,
            "margin": round(margin, 4),
            "neighbours_agreeing": agree,
            "borderline": bool(margin < BORDERLINE_MARGIN),
            "reason": reason,
        })
    return out


def report(matrix: np.ndarray, labels: np.ndarray, k: int, seed: int = 42) -> dict:
    """The whole quality picture for one run."""
    agreement = neighbour_agreement(matrix, labels)
    versus = against_shuffled(matrix, k, seed)
    stab = stability(matrix, k, seed)

    # One honest headline. Two of three must hold: neighbours agree well above
    # chance, groups survive resampling, and the split beats shuffled data.
    checks = [
        agreement["lift"] is not None and agreement["lift"] >= 2.0,
        stab["stability"] is not None and stab["stability"] >= 0.60,
        versus["beats_noise"],
    ]
    passed = sum(bool(c) for c in checks)
    verdict = ("structure found" if passed == 3 else
               "usable, with caveats" if passed == 2 else
               "weak — treat the grouping as a prompt, not a finding")

    return {
        "neighbour_agreement": agreement,
        "stability": stab,
        "versus_shuffled": versus,
        "checks_passed": passed,
        "verdict": verdict,
    }
