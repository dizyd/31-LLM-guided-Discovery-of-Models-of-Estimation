"""
Scientific regret minimisation: turning the Centaur/model gap into a usable prompt.

Two things went wrong in v1 and both are addressed here.

**Selection.** ``Delta = model_nll - centaur_nll`` was thresholded absolutely (``> 4`` or
``> 5``), which flagged 1418 of 3264 Mammals trials and overflowed the context window. Worse,
Centaur's NLL is smallest exactly on round and repeated responses (2.98 vs 5.03 nats for
multiples of 100; 3.08 vs 5.85 for values the participant reused), so an absolute threshold
selects trials by *response format*, not by misfit. Selection is now by rank within
participant, on top of a reporting layer that has already absorbed the format effect.

**Content.** The LLM was shown item names (``Item: Zebra``) although the model only ever sees
anonymous MDS coordinates, and was shown neither the model's prediction nor the true value.
It also read raw criterion values in the prompt while the fit ran on z-standardised data.
Each flagged trial now carries its cue vector, the model's prediction, the participant's
estimate, the true value and the nearest exemplar; and a block of aggregate residual
diagnostics is included, because a list of individual trials is far weaker evidence in a
continuous task than in a binary one.
"""

import numpy as np

from asmr_fit import predictions

DOMAIN_TEXT = {
    "Mammals": {
        "criterion": "the number of days until female maturity",
        "unit":      "days",
        "objects":   "mammals",
    },
    "Food": {
        "criterion": "the carbohydrate content in grams per 100 g",
        "unit":      "grams per 100 g",
        "objects":   "food items",
    },
    "Countries": {
        "criterion": "life expectancy in years",
        "unit":      "years",
        "objects":   "countries",
    },
}


def regret(fit, ds):
    """Per-trial ``Delta = model_nll - centaur_nll``; ``nan`` where not comparable."""
    d = fit.trial_nll - ds.centaur_nll
    d[~fit.ok] = np.nan
    return d


def format_covariates(ds):
    """``(n_subs, n_items, n_cov)`` response-format regressors.

    These describe *how the participant wrote the number down*, not what they judged:
    whether it is a round number and whether they had used that value elsewhere. Centaur
    is very good at both -- it is autoregressive over the participant's own answers -- and
    a cognitive model that predicts a single quantity per item structurally cannot be.
    """
    y   = ds.estimates.T                                   # (n_subs, n_items)
    cov = [np.ones_like(y),
           (y % 100 == 0).astype(float),
           (y % 10 == 0).astype(float)]

    reused = np.zeros_like(y, dtype=float)
    for p in range(ds.n_subs):
        v, c = np.unique(ds.estimates[ds.valid[:, p], p], return_counts=True)
        dup  = set(v[c > 1])
        reused[p] = [e in dup for e in ds.estimates[:, p]]
    cov.append(reused)
    return np.stack(cov, axis=-1)


def residualize(delta, ds):
    """Remove the part of Delta that response formatting alone explains.

    Regresses Delta on :func:`format_covariates` within each participant and returns the
    residual. Without this, the worst-Delta trials are largely trials where Centaur got a
    round or repeated number right -- a gap no item-level cognitive model can close, so
    handing those to the LLM as "here is where your model fails" is misleading.
    """
    cov = format_covariates(ds)
    out = np.full_like(delta, np.nan)
    for p in range(delta.shape[0]):
        m = np.isfinite(delta[p])
        if m.sum() <= cov.shape[-1]:
            out[p, m] = delta[p, m]
            continue
        X, d = cov[p][m], delta[p][m]
        beta, *_ = np.linalg.lstsq(X, d, rcond=None)
        out[p, m] = d - X @ beta
    return out


def select_points(delta, top_k_per_sub=8, max_points=60, rng=None, ds=None,
                  residualize_format=False):
    """Worst ``top_k_per_sub`` trials per participant, then thinned to ``max_points``.

    Ranking within participant removes any participant-specific offset (differences in
    overall fit, in response variability, in how predictable that person is to Centaur)
    and caps the prompt size deterministically. Set ``residualize_format=True`` (and pass
    ``ds``) to rank on the format-residualised Delta instead.

    ``rng`` is accepted for call-site symmetry; selection is deterministic given ``delta``.
    """
    score = residualize(delta, ds) if (residualize_format and ds is not None) else delta
    picks = []
    for p in range(delta.shape[0]):
        row = score[p]
        ok  = np.flatnonzero(np.isfinite(row))
        if ok.size == 0:
            continue
        order = ok[np.argsort(-row[ok])][:top_k_per_sub]
        picks.extend((p, int(i), float(row[i])) for i in order)

    if len(picks) > max_points:
        # Keep the globally worst points but spread them over participants: take them in
        # descending Delta after interleaving participants by their within-subject rank.
        picks.sort(key=lambda t: -t[2])
        by_sub, out = {}, []
        for p, i, d in picks:
            by_sub.setdefault(p, []).append((p, i, d))
        rank = 0
        while len(out) < max_points and any(len(v) > rank for v in by_sub.values()):
            tier = [v[rank] for v in by_sub.values() if len(v) > rank]
            tier.sort(key=lambda t: -t[2])
            out.extend(tier[: max_points - len(out)])
            rank += 1
        picks = out

    picks.sort(key=lambda t: (t[0], -t[2]))
    return picks


def _fmt(v, nd=2):
    return np.array2string(np.asarray(v), precision=nd, separator=", ",
                           suppress_small=True, max_line_width=10_000)


def exemplar_block(ds):
    lines = [f"The {len(ds.ex_ids)} training exemplars, their true criterion values, and "
             "their cues (these are `ex_crit` and `ex_cues` in the code):"]
    for name, crit, cue in zip(ds.ex_names, ds.ex_crit, ds.ex_cues):
        lines.append(f"* {name} (true value: {crit:g}) cues: {_fmt(cue)}")
    return "\n".join(lines)


def point_block(picks, preds, ds):
    """One line per flagged trial, with everything needed to reason about it."""
    d_ex   = np.linalg.norm(ds.cues[:, None, :] - ds.ex_cues[None, :, :], axis=-1)
    near   = d_ex.argmin(axis=1)

    lines = []
    for p, i, _ in picks:
        lines.append(
            f"* {ds.item_names[i]}: the participant estimated "
            f"{ds.estimates[i, p]:g}, the model predicted {preds[p][i]:.0f}, "
            f"the true value is {ds.true_crit[i]:g}. "
            f"cues: {_fmt(ds.cues[i])}. "
            f"Nearest exemplar: {ds.ex_names[near[i]]} "
            f"(distance {d_ex[i, near[i]]:.2f}, true value {ds.ex_crit[near[i]]:g})."
        )
    return "\n".join(lines)


def diagnostic_block(preds, ds, fit):
    """Aggregate residual structure -- the part the LLM can actually generalise from."""
    P = np.stack([preds[p] for p in range(ds.n_subs)], axis=1)      # (n_items, n_subs)
    R = np.where(ds.valid, ds.estimates - P, np.nan)                # signed residual
    R[:, ~fit.ok] = np.nan
    item_res = np.nanmean(R, axis=1)

    d_ex = np.linalg.norm(ds.cues[:, None, :] - ds.ex_cues[None, :, :], axis=-1)
    dmin = d_ex.min(axis=1)
    pbar = np.nanmean(P[:, fit.ok], axis=1)

    def corr(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 2 else np.nan

    lines = [
        "Aggregate structure of the model's errors "
        "(residual = participant's estimate minus the model's prediction, "
        "averaged over participants for each item):",
        f"* mean residual over all items: {np.nanmean(item_res):+.1f}",
        f"* correlation of the residual with the distance to the nearest exemplar: "
        f"{corr(item_res, dmin):+.2f}",
        f"* correlation of the residual with the true criterion value: "
        f"{corr(item_res, ds.true_crit):+.2f}",
        f"* correlation of the residual with the model's own prediction: "
        f"{corr(item_res, pbar):+.2f}",
    ]

    cue_r = [corr(item_res, ds.cues[:, d]) for d in range(ds.n_dim)]
    lines.append("* correlation of the residual with each cue dimension "
                 f"(cue 1 to {ds.n_dim}): {_fmt(cue_r)}")

    # Regression to the mean: does the model over-predict low items and under-predict
    # high ones, or the other way round?
    order = np.argsort(pbar)
    thirds = np.array_split(order, 3)
    labels = ("lowest-predicted third", "middle third", "highest-predicted third")
    for lab, idx in zip(labels, thirds):
        lines.append(f"* mean residual for the {lab} of items "
                     f"(model predicts {pbar[idx].mean():.0f} on average): "
                     f"{np.nanmean(item_res[idx]):+.1f}")
    return "\n".join(lines)


def history_block(history):
    if not history:
        return ""
    lines = ["Performance of the models tried so far "
             "(AIC is lower-is-better and charges 2 per free parameter per participant; "
             "held-out log-likelihood is higher-is-better and is the criterion that "
             "actually decides which model is kept):"]
    for h in history:
        cv = f"{h['cv_ll']:.1f}" if h.get("cv_ll") is not None else "n/a"
        lines.append(f"* iteration {h['iteration']}: {h['k']} free parameters, "
                     f"AIC = {h['aic']:.1f}, held-out log-likelihood = {cv}")
    return "\n".join(lines)


INSTRUCTIONS = """
Can you suggest an improved model that is able to capture human behavior in the listed
situations?

Please structure your answer as follows:
* Keep the function signature exactly the same:
  `def model(parameters, cues, ex_cues=None, ex_crit=None):`
* Return ONLY `pred_crit`, a 1-D array of length n_trials. Do NOT return a tuple and do
  NOT return a sigma; the response noise is handled outside your function.
* State the number of free parameters before the function using the NUM_PARAMETERS
  variable. Do NOT define a BOUNDS variable.
* `parameters` is an unconstrained array of real numbers of order 1. Apply any constraint
  yourself inside the function (for example `np.exp(...)` for a positive quantity, a
  sigmoid for something in [0, 1], a softmax for weights that must sum to a constant).
* Your predictions must be on the raw criterion scale. Set that scale from `ex_crit` and
  `ex_cues` statistics, as the current model does, so that `parameters = 0` already gives
  predictions of a plausible magnitude.
* Every parameter from 0 to NUM_PARAMETERS - 1 must be used and must change the
  prediction. Unused parameters are rejected, and each extra parameter costs AIC, so add
  one only if it earns its keep.
* You are encouraged to change the internal mathematical logic substantially, including
  proposing a different psychological process, rather than only retuning the existing
  equation.
* Output ONLY valid, executable Python code. No markdown fences, no commentary.
""".strip()


def compose(model_source, ds, fit, preds, picks, history=None):
    """Full prompt text for one ASMR iteration."""
    txt   = DOMAIN_TEXT[ds.domain]
    lo, hi = np.nanmin(ds.estimates), np.nanmax(ds.estimates)

    parts = [
        f"I am studying human behavior in an estimation experiment.\n"
        f"In this experiment, participants estimate {txt['criterion']} for various "
        f"{txt['objects']}.\n",
        "Experiment structure:\n"
        f"1. Training phase: participants repeatedly estimated {txt['criterion']} for 12 "
        "specific exemplars and received immediate feedback with the true value after "
        "each estimate.\n"
        f"2. Testing phase: participants estimated {txt['criterion']} for "
        f"{ds.n_items} new objects, in random order, without feedback.\n",
        "Feature representation (important):\n"
        f"The {ds.n_dim} numbers in each row of the `cues` matrix are coordinates from a "
        "multidimensional scaling analysis of human similarity judgments. They form a "
        "psychological map: objects that people perceive as similar lie close together in "
        f"this {ds.n_dim}-dimensional space. The individual dimensions have no a priori "
        "meaning, and distances in this space are the psychologically meaningful "
        "quantity.\n",
        f"Scale: estimates and predictions are in {txt['unit']}, on the raw scale. "
        f"Participants' estimates in this data set range from {lo:g} to {hi:g}, with the "
        f"true criterion values ranging from {ds.true_crit.min():g} to "
        f"{ds.true_crit.max():g}.\n",
        exemplar_block(ds),
        "\nI have the following computational model that is currently my best guess for "
        "how people make estimates in this experiment. Its parameters are fitted "
        "separately for each participant by maximum likelihood:\n",
        model_source,
        "\nThis model captures human behavior reasonably well overall, but the following "
        "data points are ones a strong reference model predicts well while this model "
        "does not:\n",
        point_block(picks, preds, ds),
        "\n" + diagnostic_block(preds, ds, fit),
    ]
    hist = history_block(history)
    if hist:
        parts.append("\n" + hist)
    parts.append("\n" + INSTRUCTIONS)
    return "\n".join(p for p in parts if p)


def check_prompt_fits(prompt, tokenizer, max_new_tokens, max_seq_length):
    """Fail loudly before generation rather than deep inside the pipeline.

    The v1 run died at ``input length 25902 + max_new_tokens 16384 exceeds the maximum
    sequence length of 40960``.
    """
    n = len(tokenizer(prompt).input_ids)
    if n + max_new_tokens > max_seq_length:
        raise ValueError(
            f"prompt is {n} tokens and max_new_tokens is {max_new_tokens}, which exceeds "
            f"the {max_seq_length}-token context. Lower max_points in select_points()."
        )
    return n
