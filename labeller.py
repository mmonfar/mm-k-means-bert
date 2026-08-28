"""
labeller.py
mmonfar. // Semantic M&M Failure Navigation Engine

Optional: name each cluster with a small local language model.

Why this exists
---------------
Naming a cluster from its top TF-IDF terms does not work on this data, and the
project measured that rather than assuming it: the most distinctive term in each
cluster appears in 12–29% of that cluster's cases. A three-word name built from
those describes at best a quarter of what it points at, and on one 11-case group
it produced "Theatre / Waited / Delay" from terms present in 3 cases each — two
of which used "theatre" to mean an operating room rather than a queue for one.

A small instruct model reads the actual cases and names the thing they share.
That is the job it is genuinely good at, and it needs no more than about a
billion parameters to do it.

Why it is OFF by default
------------------------
1. It is a ~1 GB download, and this project's promise is that it runs on a
   laptop with no ceremony. Opt in with `--smart-labels`.
2. A generated label is a *claim about a group of clinical cases*. When it is
   wrong it is wrong fluently, which is worse than "Group 3". So every label is
   validated, marked in the payload as model-generated, and shown alongside the
   exemplar case so a reader can check it in one glance.

No new dependency: `transformers` and `torch` already arrive with
sentence-transformers. Only the weights are new, and they are Apache-2.0.

Nothing leaves the machine. The model is downloaded once to the local
HuggingFace cache and runs on CPU thereafter.
"""

from __future__ import annotations

import re

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# A model can only name a group that has one thing to name. Measured on the demo
# register, the two groups a model got WRONG were the two loosest:
#
#   group  cohesion  Qwen2.5-0.5B            Qwen2.5-1.5B
#   1      0.35      Delays reaching theatre  Documentation oversight
#   3      0.31      Delayed reporting        Time delays
#   2      0.28      Medication administration errors  Dosage calculation errors
#   0      0.24      Errors in documentation  Communication failures
#   4      0.23      Delays reaching theatre  Inadequate communication   <- surgical
#
# Group 4 is surgical complications; both models called it something else, and
# both duplicated a name they had already used. Scaling the model 3x did not fix
# it, because the fault is not capacity — a group with nothing in common has no
# name, and a fluent model will invent one anyway. So the gate is cohesion, not
# model size: below this, the group keeps its number.
LABEL_MIN_COHESION = 0.25

MAX_LABEL_WORDS = 6
MAX_LABEL_CHARS = 46
CASES_PER_PROMPT = 6
CASE_CHARS = 220

SYSTEM = (
    "You name groups of hospital incident reports. You reply with a short noun "
    "phrase naming the failure the reports share, and nothing else."
)

INSTRUCTION = (
    "These {n} incident reports were grouped together because they are similar.\n\n"
    "{cases}\n\n"
    "Reply with a 2 to 5 word noun phrase naming the failure they share, in the "
    "style of 'Medication dosing errors' or 'Delays reaching theatre'. "
    "No preamble, no punctuation, no explanation."
)

# Refusals, restatements and other non-answers a small model emits under pressure.
_REJECT = re.compile(
    r"\b(i (cannot|can't|am|will)|as an ai|sure|here('| i)s|the (group|reports|"
    r"answer)|these reports|noun phrase|sorry)\b",
    re.I,
)


def clean(raw: str) -> str | None:
    """Validate and normalise a generated label, or return None to reject it.

    A small model will occasionally answer with a sentence, a refusal, a
    restatement of the prompt, or a label with a case number welded on. None of
    those may reach a governance meeting wearing the authority of a group name,
    so anything that is not a short clean noun phrase is thrown away and the
    caller falls back to the deterministic label.
    """
    if not raw:
        return None

    text = raw.strip().splitlines()[0].strip()
    text = text.strip("\"'“”‘’ \t.:;-–—*#")
    text = re.sub(r"\s+", " ", text)

    if not text or _REJECT.search(text):
        return None
    # Digits in a group name are case numbers or timestamps out of the source.
    if any(ch.isdigit() for ch in text):
        return None
    if len(text) > MAX_LABEL_CHARS:
        return None

    words = text.split()
    if not 2 <= len(words) <= MAX_LABEL_WORDS:
        return None
    # A trailing full stop is fine; a sentence is not.
    if text.count(",") > 1 or "." in text[:-1]:
        return None

    return text[0].upper() + text[1:]


def representative_cases(texts, members, matrix, limit=CASES_PER_PROMPT):
    """The cases closest to the cluster centroid, which is what the group *is*.

    Feeding a random sample would let one outlier steer the name; the cases
    nearest the centre are the ones the group is actually about.
    """
    import numpy as np

    members = list(members)
    if matrix is None or len(members) <= limit:
        chosen = members[:limit]
    else:
        centroid = matrix[members].mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) or 1.0)
        order = np.argsort(-(matrix[members] @ centroid))
        chosen = [members[i] for i in order[:limit]]
    return [texts[i][:CASE_CHARS] for i in chosen]


class LocalLabeller:
    """Wraps a small instruct model. Loaded once, reused for every cluster."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()

    def name_group(self, cases: list[str], avoid: set[str] | None = None) -> str | None:
        import torch

        listing = "\n".join(f"- {c}" for c in cases)
        prompt = INSTRUCTION.format(n=len(cases), cases=listing)
        # Two groups may not share a name. On the first run the 0.5B model gave
        # "Delays reaching theatre" to both the diagnostic-delay group and the
        # surgical-complication group — worse than two numbers, because it
        # asserts the two are the same thing.
        if avoid:
            prompt += ("\n\nDo not use any of these names, they belong to other "
                       "groups: " + "; ".join(sorted(avoid)) + ".")
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=16,
                # Greedy, so two runs of the same register produce the same names.
                # A governance pack that changes wording between runs is not
                # trustworthy even when every individual name is defensible.
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = out[0][inputs["input_ids"].shape[-1]:]
        return clean(self.tokenizer.decode(generated, skip_special_tokens=True))


def apply(clusters, texts, labels, matrix, model_name: str = DEFAULT_MODEL) -> str:
    """Re-name clusters in place. Returns the model actually used, or "".

    Every cluster keeps `label_source`, so the interface can say where its name
    came from rather than presenting a generated phrase and a measured one as
    though they carry the same weight.
    """
    import numpy as np

    for c in clusters:
        c.setdefault("label_source", "terms" if c.get("keywords") else "numbered")

    try:
        engine = LocalLabeller(model_name)
    except Exception as exc:
        print(f"      note        : smart labels unavailable ({type(exc).__name__}: "
              f"{str(exc)[:80]}) — keeping deterministic names")
        return ""

    used: set[str] = set()
    # Tightest groups first, so the clearest names claim their wording before a
    # looser group can take it.
    order = sorted(clusters, key=lambda c: -(c.get("cohesion") or 0))

    for c in order:
        members = np.flatnonzero(labels == c["id"])
        if not len(members):
            continue

        cohesion = c.get("cohesion")
        if cohesion is not None and cohesion < LABEL_MIN_COHESION:
            c["label_source"] = "numbered"
            print(f"      note        : group {c['id']} too mixed to name "
                  f"(cohesion {cohesion:.2f} < {LABEL_MIN_COHESION})")
            continue

        cases = representative_cases(texts, members, matrix)
        named = None
        for attempt in range(2):
            try:
                candidate = engine.name_group(cases, avoid=used if attempt else None)
            except Exception as exc:  # a failure must not lose the whole run
                print(f"      note        : group {c['id']} not named ({exc})")
                break
            if candidate and candidate.casefold() not in used:
                named = candidate
                break
            # Duplicate: say what is taken and let it try once more.

        if named:
            c["label"] = named
            c["label_source"] = "model"
            used.add(named.casefold())
        # A rejected or duplicate generation keeps the deterministic label.

    return model_name
