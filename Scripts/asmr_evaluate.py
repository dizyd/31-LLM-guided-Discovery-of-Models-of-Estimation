"""
Evaluating the library of discovered models.

Individual differences are recovered by **model selection**, not by discovering a bespoke
model per person: every model that ASMR validated, across seeds and iterations, goes into a
library, each library member is scored against every participant on held-out items, and
each participant is assigned their best member. That yields a counts table directly
comparable to Table 2 of the manuscript, without the overfitting of fitting one model form
per person.

Held-out log-likelihood is the primary criterion. AIC is reported too, but the LLM can
lower AIC simply by proposing fewer parameters and raise it by proposing more, so it is not
what decides.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from asmr_codegen import validate_model_code
from asmr_fit import fit_all, granularity_weights
from asmr_likelihood import coarsest_grid
from asmr_models_seed import REFERENCE_MODELS, SEED_MODELS
from asmr_pipeline import OUT_DIR, model_hash


def load_library(out_dir=OUT_DIR, domain=None, include_seeds=True,
                 include_references=True):
    """All discovered models, plus the seeds and reference models for comparison."""
    lib = {}
    if include_seeds:
        for name, src in SEED_MODELS.items():
            lib[model_hash(src)] = {"source": src, "name": f"seed:{name}", "origin": "seed"}
    if include_references:
        for name, src in REFERENCE_MODELS.items():
            lib[model_hash(src)] = {"source": src, "name": f"ref:{name}",
                                    "origin": "reference"}

    path = Path(out_dir) / "library.jsonl"
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                e = json.loads(line)
                if domain is not None and e.get("domain") != domain:
                    continue
                lib.setdefault(e["model_id"], {}).update(
                    source=e["source"],
                    name=e.get("name", f"{e.get('seed', '?')}/run{e.get('run_id', '?')}"
                                       f"/it{e.get('iteration', '?')}"),
                    origin="discovered",
                )
    return lib


def per_participant_heldout(source, ds, pis=None, dist="normal", n_folds=5,
                            n_restarts=3, seed=0, n_jobs=1):
    """Held-out log-likelihood per participant for one model.

    ``seed`` fixes the item folds, so every library member is scored on the same splits
    and the comparison is paired. Re-scoring one model under different splits moves the
    number by hundreds of nats, which would swamp real differences between models.
    """
    rep = validate_model_code(source, ds.cues, ds.ex_cues, ds.ex_crit)
    if not rep.ok:
        return None, rep.error

    pis   = granularity_weights(ds) if pis is None else pis
    rng   = np.random.default_rng(seed)
    order = rng.permutation(ds.n_items)
    folds = np.array_split(order, n_folds)

    from asmr_likelihood import ModelEvaluationError, nll_reported

    held = np.full((ds.n_subs, ds.n_items), np.nan)
    for f, test_idx in enumerate(folds):
        train_idx = np.setdiff1d(order, test_idx)
        fit = fit_all(rep.model_fn, rep.num_parameters, ds, pis=pis, dist=dist,
                      n_restarts=n_restarts, items=train_idx, seed=seed + f,
                      n_jobs=n_jobs, source=rep.source)
        for p in range(ds.n_subs):
            mask = ds.valid[test_idx, p]
            if not fit.ok[p] or mask.sum() == 0:
                continue
            scale = float(np.nanstd(ds.estimates[:, p])) or 1.0
            try:
                nll = nll_reported(fit.theta[p], rep.model_fn, ds.estimates[test_idx, p],
                                   ds.cues[test_idx], ds.ex_cues, ds.ex_crit, pis[p],
                                   ds.ub, sigma_scale=scale, dist=dist, agg="none",
                                   mask=mask)
            except ModelEvaluationError:
                continue
            held[p, test_idx] = -nll

    return held, ""


def evaluate_library(ds, library=None, dist="normal", n_folds=5, n_restarts=3, seed=0,
                     n_jobs=1, verbose=True):
    """Score every library member on every participant. Returns a tidy frame."""
    library = load_library(domain=ds.domain) if library is None else library
    rows    = []
    for mid, entry in library.items():
        held, err = per_participant_heldout(entry["source"], ds, dist=dist,
                                            n_folds=n_folds, n_restarts=n_restarts,
                                            seed=seed, n_jobs=n_jobs)
        if held is None:
            if verbose:
                print(f"  {entry['name']}: rejected ({err.splitlines()[0]})")
            continue
        rep = validate_model_code(entry["source"], ds.cues, ds.ex_cues, ds.ex_crit)
        per_sub = np.nansum(held, axis=1)
        n_sub   = np.isfinite(held).sum(axis=1)
        for p in range(ds.n_subs):
            rows.append({"model_id": mid, "name": entry["name"],
                         "origin": entry.get("origin", "discovered"),
                         "k": rep.num_parameters + 1, "participant": p,
                         "heldout_ll": per_sub[p], "n_trials": int(n_sub[p])})
        if verbose:
            tot = float(np.nansum(held))
            print(f"  {entry['name']:>28}: k={rep.num_parameters + 1:2d} "
                  f"heldout LL={tot:10.1f} "
                  f"({tot / max(np.isfinite(held).sum(), 1):+.3f} nats/trial)")
    return pd.DataFrame(rows)


def winner_table(scores):
    """Best model per participant, and the counts by model (cf. Manuscript Table 2)."""
    best = scores.loc[scores.groupby("participant")["heldout_ll"].idxmax()]
    counts = (best.groupby("name").size().sort_values(ascending=False)
              .rename("n_participants").reset_index())
    return best, counts


def reference_rows(ds):
    """Reference points every discovered model should be read against."""
    valid  = ds.valid.T
    cent   = float(ds.centaur_nll[valid].mean())

    # Descriptive, in-sample and therefore optimistic: the entropy of each participant's
    # own empirical distribution of test responses, i.e. what a model that knows only the
    # person's response style (and gets to see the answers) would pay per trial. This is
    # the quantity that motivated the reporting layer.
    ent = []
    for p in range(ds.n_subs):
        y = ds.estimates[ds.valid[:, p], p]
        _, c = np.unique(y, return_counts=True)
        q = c / c.sum()
        ent.append(float(-(q * np.log(q)).sum()))

    return pd.DataFrame([
        {"reference": "Centaur (0 fitted parameters)", "nats_per_trial": cent},
        {"reference": "in-sample entropy of own responses (optimistic bound)",
         "nats_per_trial": float(np.mean(ent))},
    ])


def _reused_mask(ds):
    """``(n_items, n_subs)`` -- did this participant give this value more than once?"""
    rep = np.zeros_like(ds.estimates, dtype=bool)
    for p in range(ds.n_subs):
        v, c = np.unique(ds.estimates[ds.valid[:, p], p], return_counts=True)
        dup  = set(v[c > 1])
        rep[:, p] = [e in dup for e in ds.estimates[:, p]]
    return rep


def response_style_summary(ds):
    """The response-format facts that make raw NLL comparisons misleading."""
    # Everything is flattened in (participant, item) order to match `centaur_nll`;
    # mixing that with the (item, participant) layout of `estimates` silently pairs each
    # response with another trial's NLL.
    valid = ds.valid.T
    y     = ds.estimates.T[valid]
    nll   = ds.centaur_nll[valid]

    rows = []
    for g in (10, 100):
        on = (y % g == 0)
        rows.append({"split": f"multiple of {g}", "share": float(on.mean()),
                     "centaur_nll_yes": float(nll[on].mean()),
                     "centaur_nll_no": float(nll[~on].mean())})

    r = _reused_mask(ds).T[valid]
    rows.append({"split": "value reused by that participant", "share": float(r.mean()),
                 "centaur_nll_yes": float(nll[r].mean()),
                 "centaur_nll_no": float(nll[~r].mean())})
    return pd.DataFrame(rows)


def regret_diagnostics(delta, ds, preds=None):
    """Is Delta tracking misfit, or still tracking response format?

    In v1 the answer was "format": Centaur's NLL differed by ~2 nats between round and
    non-round responses and by ~2.8 between reused and unique values, so thresholding
    Delta selected trials by how the participant wrote the number down.

    The quantity Delta *should* track is the cognitive model's own residual -- how far its
    prediction sits from what the participant actually said. Pass ``preds`` (a dict of
    per-participant prediction arrays, as the pipeline builds) to measure that directly;
    the participant's distance from the true value is only a weak proxy for it.
    """
    # (participant, item) order throughout, matching `delta` and `centaur_nll`.
    valid  = ds.valid.T
    d      = delta[valid]
    y      = ds.estimates.T[valid]
    true   = np.tile(ds.true_crit, (ds.n_subs, 1))[valid]
    err    = np.abs(np.log(np.clip(y, 1, None)) - np.log(true))

    round100 = (y % 100 == 0).astype(float)
    reused   = _reused_mask(ds).T[valid].astype(float)

    def corr(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 2 else np.nan

    rows = []
    if preds is not None:
        P    = np.stack([preds[p] for p in range(ds.n_subs)])          # (n_subs, n_items)
        resid = np.abs(np.log(np.clip(ds.estimates.T, 1, None))
                       - np.log(np.clip(P, 1, None)))[valid]
        rows.append({
            "quantity": "corr(Delta, |log estimate - log model prediction|)",
            "value": corr(d, resid),
            "wanted": "high -- this is the signal SRM is supposed to select on"})

    rows += [
        {"quantity": "corr(Delta, |log estimate - log true|)", "value": corr(d, err),
         "wanted": "weak proxy for misfit; not the target"},
        {"quantity": "corr(Delta, response is a multiple of 100)",
         "value": corr(d, round100), "wanted": "near zero -- format should not drive it"},
        {"quantity": "corr(Delta, participant reused this value)",
         "value": corr(d, reused),
         "wanted": "near zero; residual here is Centaur's autoregressive advantage"},
    ]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys

    from asmr_data import load_aligned

    domain = sys.argv[1] if len(sys.argv) > 1 else "Mammals"
    ds = load_aligned(domain)

    print("\nResponse-format structure of the data:")
    print(response_style_summary(ds).to_string(index=False))
    print("\nReference points:")
    print(reference_rows(ds).to_string(index=False))
